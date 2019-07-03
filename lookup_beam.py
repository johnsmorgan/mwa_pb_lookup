#!/usr/bin/env python
import os
import logging
import numpy as np
from h5py import File
from optparse import OptionParser #NB zeus does not have argparse!

from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Longitude
from astropy import units as u


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
    edge_freq_hz = (int_chans[0]*1280000-640000, int_chans[1]*1280000+640000)

    lower = np.argwhere(chans == edge_freq_hz[0]).flatten()
    upper = np.argwhere(chans == edge_freq_hz[1]).flatten()
    if len(lower) == 0:
        raise IndexError, "No match for lower coarse chan %d" % int_chans[0]
    if len(upper) == 0:
        raise IndexError, "No match for upper coarse chan %d" % int_chans[1]
    return lower[0], upper[0]-lower[0]+1

def mhz_to_index_weight(chans, freq_mhz):
    freq = 1e6*freq_mhz
    i = np.searchsorted(chans, freq)
    if i == 0:
        raise ValueError, "Frequency %f below lowest channel %f" % (freq, chans[0])
    if i == len(chans):
        raise ValueError, "Frequency %f above highest channel %f" % (freq, chans[-1])
    weight1 = 1-(freq-chans[i-1])/np.float(chans[i]-chans[i-1])
    return i-1, weight1

def get_meta(obsid_str):
    meta_hdus = fits.open("%s.metafits" % obsid_str)
    gridnum = meta_hdus[0].header['GRIDNUM']
    start_lst = meta_hdus[0].header['LST']
    lst = start_lst + 360*meta_hdus[0].header['Exposure']/86164.1
    return gridnum, lst

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

def get_avg_beam_spline(beam_file, low_index, n_freq, weights):
    beam_xy = np.sum(np.nan_to_num(beam_file['beams'][gridnum, low_index:low_index+n_freq, ...])*weights.reshape(n_freq, 1, 1, 1),
                     axis=0)
    # Note that according to the docs, x, y should be
    # "1-D arrays of coordinates in strictly ascending order."
    # However this seems to work
    beam_x = RectBivariateSpline(x=beam_file['dec_scale'][...], y=beam_file['ha_scale'][...], z=beam_xy[0])
    beam_y = RectBivariateSpline(x=beam_file['dec_scale'][...], y=beam_file['ha_scale'][...], z=beam_xy[1])
    return tidy_spline(beam_x, np.float32), tidy_spline(beam_y, np.float32)

def header_to_pixel_radec(header):
    wcs = WCS(header)
    # Get RA and Dec of each pixel in hdf5 file
    n_x, n_y = header['NAXIS1'], header['NAXIS2']
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    ra, dec = wcs.celestial.wcs_pix2world(x, y, 0)
    return ra, dec

def ra_to_ha(ra, lst):
    return Longitude((lst-ra)*u.deg, wrap_angle=180*u.deg).deg

if __name__ == '__main__':
    try:
        PB_FILE = os.environ['MWA_PB_LOOKUP']
    except KeyError:
        PB_FILE = ""

    parser = OptionParser(usage="usage: obsid suffix [out_prefix] [out_suffix]" +
                          """
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
                          """)
    parser.add_option("-c", "--chan_str", dest="chan_str", default=None, type="str", help="coarse channel string (e.g. 121-132)")
    parser.add_option("-f", "--freq_mhz", dest="freq_mhz", default=None, type="float", help="frequency in MHz")
    parser.add_option("-v", "--verbose", action="count", dest="verbose", help="-v info, -vv debug")
    parser.add_option("--beam_path", default=PB_FILE, dest="beam_path", type="str", help="path to hdf5 file containing beams")
    parser.add_option("--metafits_suffix", default=".metafits", action="count", dest="verbose", help="-v info, -vv debug (default %default")
    parser.add_option("--delete", action="store_true", dest="delete", help="delete output files if they already exist")

    opts, args = parser.parse_args()

    if len(args) < 2:
        parser.error("incorrect number of arguments")
    obsid = args[0]
    suffix = args[1]
    if len(args) > 2:
        out_prefix = args[2]
    else:
        out_prefix = obsid + '-'
    if len(args) > 3:
        out_suffix = args[3]
    else:
        out_suffix = "-beam.fits"

    if opts.chan_str is None and opts.freq_mhz is None:
        parser.error("Either chan_str or freq_mhz must be set")

    if opts.chan_str is not None and opts.freq_mhz is not None:
        parser.error("Either chan_str *or* freq_mhz must be set")

    if opts.verbose == 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
    elif opts.verbose > 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)

    out_x = "%sXX%s" % (out_prefix, out_suffix)
    if os.path.exists(out_x):
        if opts.delete:
            os.remove(out_x)
        else:
            raise RuntimeError, "%s exists" % out_x
    out_y = "%sYY%s" % (out_prefix, out_suffix)
    if os.path.exists(out_y):
        if opts.delete:
            os.remove(out_y)
        else:
            raise RuntimeError, "%s exists" % out_y

    # get metadata
    logging.debug("getting metadata")
    gridnum, lst = get_meta(obsid)

    #open beam file
    logging.debug("generate spline from beam file")
    df = File(opts.beam_path)
    if opts.chan_str is not None:
        low_index, n_chan = coarse_range(df['chans'][...], opts.chan_str)
        weights = trap(n_chan)
        logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+n_chan], weights)
        beam_x, beam_y = get_avg_beam_spline(df, low_index, n_chan, weights)
    else:
        low_index, weight1 = mhz_to_index_weight(df['chans'][...], opts.freq_mhz)
        weights = np.array((weight1, 1-weight1))
        logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+2], weights)
        beam_x, beam_y = get_avg_beam_spline(df, low_index, 2, weights)

    hdus = fits.open(obsid+suffix)
    header = hdus[0].header
    data = hdus[0].data
    logging.debug("calculate pixel ra, dec")
    ra, dec = header_to_pixel_radec(header)
    logging.debug("convert to ha")
    ha = ra_to_ha(ra, lst)

    # store metadata in fits header
    hdus[0].header['PBVER'] = df.attrs['VERSION']
    hdus[0].header['PBPATH'] = opts.beam_path
    hdus[0].header['PBLST'] = lst
    hdus[0].header['PBGRIDN'] = gridnum

    # get values for each fits image pix
    logging.debug("interpolating beams for XX")
    hdus[0].data = beam_x(dec, ha, data.shape)
    logging.debug("writing XX beam to disk")
    hdus.writeto(out_x)
    logging.debug("interpolating beams for YY")
    hdus[0].data = beam_y(dec, ha, data.shape)
    logging.debug("writing YY beam to disk")
    hdus.writeto(out_y)
    logging.debug("finished")
