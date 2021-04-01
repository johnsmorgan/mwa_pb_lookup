#!/usr/bin/env python
import os
import logging
import numpy as np
from h5py import File
from optparse import OptionParser  # NB zeus does not have argparse!

from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Longitude, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u

N_POL = 2
POLS = ("XX", "YY")
# MWA location from CONV2UVFITS/convutils.h
LAT = -26.703319
LON = 116.67081
ALT = 377.0

LOCATION = EarthLocation.from_geodetic(
    lat=LAT * u.deg, lon=LON * u.deg, height=ALT * u.m
)

try:
    PB_FILE = os.environ["MWA_PB_BEAM"]
except:
    try:
        PB_FILE = os.environ["MWA_PB_LOOKUP"]
    except KeyError:
        PB_FILE = ""


def trap(n):
    """
    return normalised trapezium
    """
    x = np.ones((n,), dtype=np.float32)
    x[0] = x[-1] = 0.5
    x /= sum(x)
    return x


def coarse_range(chans, coarse_str):
    int_chans = [int(c) for c in coarse_str.split("-")]
    edge_freq_hz = (int_chans[0] * 1280000 - 640000, int_chans[1] * 1280000 + 640000)

    lower = np.argwhere(chans == edge_freq_hz[0]).flatten()
    upper = np.argwhere(chans == edge_freq_hz[1]).flatten()
    if len(lower) == 0:
        raise IndexError("No match for lower coarse chan %d" % int_chans[0])
    if len(upper) == 0:
        raise IndexError("No match for upper coarse chan %d" % int_chans[1])
    return lower[0], upper[0] - lower[0] + 1


def mhz_to_index_weight(chans, freq_mhz):
    freq = 1e6 * freq_mhz
    i = np.searchsorted(chans, freq)
    if i == 0:
        raise ValueError("Frequency %f below lowest channel %f" % (freq, chans[0]))
    if i == len(chans):
        raise ValueError("Frequency %f above highest channel %f" % (freq, chans[-1]))
    weight1 = 1 - (freq - chans[i - 1]) / np.float(chans[i] - chans[i - 1])
    return i - 1, weight1


def get_meta_lst(obsid_str):
    meta_hdus = fits.open("%s.metafits" % obsid_str)
    gridnum = meta_hdus[0].header["GRIDNUM"]
    start_lst = meta_hdus[0].header["LST"]
    lst = start_lst + 360.0 * meta_hdus[0].header["Exposure"] / 2 / 86164.1
    return gridnum, lst


def get_meta(obsid_str, metafits_suffix=".metafits"):
    meta_hdus = fits.open("%s%s" % (obsid_str, metafits_suffix))
    gridnum = meta_hdus[0].header["GRIDNUM"]
    start_time = meta_hdus[0].header["DATE-OBS"]
    duration = meta_hdus[0].header["EXPOSURE"] * u.s
    t = Time(start_time, format="isot", scale="utc") + 0.5 * duration
    return gridnum, t


def tidy_spline(spline, dtype):
    """
    RectBivariateSpline does not respect precision (always promotes to float32)
    RectBivariateSpline also returns flattened array.
    
    This function factory will return a spline which when called will
    - set grid to false
    - shape spline to 'shape' and convert to dtype

    dtype is set when this function factory is called
    shape is set when spline function is called

    shape is set explicitly (rather than implicitly via e.g. dec.shape)
    because a different, equally-sized shape may be required due to extra FITS dimensions
    """

    def inner(dec, ha, shape):
        return dtype(spline(dec, ha, grid=False).reshape(shape))

    return inner


def get_avg_beam_spline(beam_file, gridnum, low_index, n_freq, weights):
    assert (
        beam_file["beams"].shape[2] == N_POL
    ), "Beam file does not contain 2 polarisations. Not an XX/YY file!"
    beam_xy = np.sum(
        np.nan_to_num(beam_file["beams"][gridnum, low_index : low_index + n_freq, ...])
        * weights.reshape(n_freq, 1, 1, 1),
        axis=0,
    )
    # Note that according to the docs, x, y should be
    # "1-D arrays of coordinates in strictly ascending order."
    # However this seems to work
    beams = {}
    for p, pol in enumerate(POLS):
        b = RectBivariateSpline(
            x=beam_file["alt_scale"][...], y=beam_file["az_scale"][...], z=beam_xy[p]
        )
        beams[pol] = tidy_spline(b, np.float32)
    return beams


def header_to_pixel_radec(header):
    wcs = WCS(header)
    # Get RA and Dec of each pixel in hdf5 file
    n_x, n_y = header["NAXIS1"], header["NAXIS2"]
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    ra, dec = wcs.celestial.wcs_pix2world(x, y, 0)
    return ra, dec


def ra_to_ha(ra, lst):
    return Longitude((lst - ra) * u.deg, wrap_angle=180 * u.deg).deg


def radec_to_altaz(ra, dec, t, location=LOCATION):
    radec = SkyCoord(ra * u.deg, dec * u.deg)
    altaz = radec.transform_to(AltAz(obstime=t, location=location))
    return altaz.alt.deg, altaz.az.deg


def beam_lookup_1d(ras, decs, gridnum, time, freq):
    """Return the attenuation of sources at RA/Dec for a given time/delay/freq in a 
    manner similar to the mwapy.pb function. 

    x,y = beam_value(data[args.racol][indices], data[args.decol][indices], t, delays, freq)

    Args:
        ras (np.ndarray): RA positions of sources
        decs (np.ndarray): Dec positions of sources
        gridnum (int): The MWA gridnum of the pointing position
        time (astropy.time.Time): Central time of the observation
        freq (float): Frequency of observation, in Hertz
    """
    assert (
        PB_FILE != ""
    ), "MWA Beam HDF5 file not configure, ensure either MWA_PB_BEAM or MWA_PB_LOOKUP is set"

    df = File(PB_FILE, "r")

    if not isinstance(ras, np.ndarray):
        ras = np.array(ras)
    if not isinstance(decs, np.ndarray):
        decs = np.array(decs)

    low_index, weight1 = mhz_to_index_weight(df["chans"][...], freq / 1_000_000)
    weights = np.array((weight1, 1 - weight1))
    beams = get_avg_beam_spline(df, gridnum, low_index, N_POL, weights)

    alt, az = radec_to_altaz(ras, decs, time)
    xx = beams["XX"](alt, az, ras.shape)
    yy = beams["YY"](alt, az, ras.shape)

    return xx, yy


if __name__ == "__main__":
    parser = OptionParser(
        usage="usage: obsid suffix [out_prefix] [out_suffix]"
        + """
                          read input fits image and metafits and produce XX and YY beams with same dimensions as fits image

                          --chan_str *or* --freq_mhz must be specified.

                          A path to the lookup table must also be specified, either via --beam_path or via the global variable MWA_PB_LOOKUP.

                          If chan_str is specified, all beams within the frequency range specified are averaged together, with
                          edge channels given half weighting

                          if --freq_mhz is specified, the two beams closest to the given frequency are linearly interpolated.


                          input fits image is 
                            [obsid][suffix]
                          input metafits is 
                            [obsid].metafits

                          output files will be 
                            [out_prefix]XX[out_suffix]
                          and
                            [out_prefix]YY[out_suffix]

                          obsid and suffix must be given

                          out_prefix defaults to obsid-
                          out_suffix defaults to "-beam.fits"
                          """
    )
    parser.add_option(
        "-c",
        "--chan_str",
        dest="chan_str",
        default=None,
        type="str",
        help="coarse channel string (e.g. 121-132)",
    )
    parser.add_option(
        "-f",
        "--freq_mhz",
        dest="freq_mhz",
        default=None,
        type="float",
        help="frequency in MHz",
    )
    parser.add_option(
        "-v", "--verbose", action="count", default=0, dest="verbose", help="-v info, -vv debug"
    )
    parser.add_option(
        "--beam_path",
        default=PB_FILE,
        dest="beam_path",
        type="str",
        help="path to hdf5 file containing beams",
    )
    parser.add_option(
        "--metafits_suffix",
        default=".metafits",
        dest="metafits_suffix",
        help="metafits suffix (default %default)",
    )
    parser.add_option(
        "--delete",
        action="store_true",
        dest="delete",
        help="delete output files if they already exist",
    )

    opts, args = parser.parse_args()

    if len(args) < 2:
        parser.error("incorrect number of arguments")
    obsid = args[0]
    suffix = args[1]
    if len(args) > 2:
        out_prefix = args[2]
    else:
        out_prefix = obsid + "-"
    if len(args) > 3:
        out_suffix = args[3]
    else:
        out_suffix = "-beam.fits"

    if opts.chan_str is None and opts.freq_mhz is None:
        parser.error("Either chan_str or freq_mhz must be set")

    if opts.chan_str is not None and opts.freq_mhz is not None:
        parser.error("Either chan_str *or* freq_mhz must be set")

    if opts.verbose == 1:
        logging.basicConfig(
            format="%(asctime)s-%(levelname)s %(message)s", level=logging.INFO
        )

    # have gotten a few type errors here about '>' being incompatible between str and int
    # not obvious to me as to why, and dont want to figure out optparse
    elif opts.verbose > 1:
        logging.basicConfig(
            format="%(asctime)s-%(levelname)s %(message)s", level=logging.DEBUG
        )

    for pol in POLS:
        out = "%s%s%s" % (out_prefix, pol, out_suffix)
        if os.path.exists(out):
            if opts.delete:
                os.remove(out)
            else:
                raise RuntimeError("%s exists" % out)

    # get metadata
    logging.debug("getting metadata")
    gridnum, t = get_meta(obsid, opts.metafits_suffix)
    logging.info("using centroid time %s", t.isot)

    # open beam file
    logging.debug("generate spline from beam file")
    df = File(opts.beam_path, "r")
    if opts.chan_str is not None:
        low_index, n_chan = coarse_range(df["chans"][...], opts.chan_str)
        weights = trap(n_chan)
        logging.info(
            "averaging channels %s Hz with weights %s",
            df["chans"][low_index : low_index + n_chan],
            weights,
        )
        beams = get_avg_beam_spline(df, gridnum, low_index, n_chan, weights)
    else:
        low_index, weight1 = mhz_to_index_weight(df["chans"][...], opts.freq_mhz)
        weights = np.array((weight1, 1 - weight1))
        logging.info(
            "averaging channels %s Hz with weights %s",
            df["chans"][low_index : low_index + 2],
            weights,
        )
        beams = get_avg_beam_spline(df, gridnum, low_index, N_POL, weights)

    hdus = fits.open(obsid + suffix)
    header = hdus[0].header
    data = hdus[0].data
    logging.debug("calculate pixel ra, dec")
    ra, dec = header_to_pixel_radec(header)
    logging.debug("convert to az el")
    alt, az = radec_to_altaz(ra, dec, t)

    # store metadata in fits header
    try:
        hdus[0].header["PBVER"] = df.attrs["VERSION"]
    except:
        hdus[0].header["PBVER"] = df.attrs["VERSION"].decode("utf-8")

    hdus[0].header["PBPATH"] = opts.beam_path
    hdus[0].header["PBTIME"] = t.isot
    hdus[0].header["PBGRIDN"] = gridnum

    # get values for each fits image pix
    for pol in POLS:
        logging.debug("interpolating beams for %s", pol)
        hdus[0].data = beams[pol](alt, az, data.shape)
        logging.debug("writing %s beam to disk", pol)
        hdus.writeto("%s%s%s" % (out_prefix, pol, out_suffix))
    logging.debug("finished")
