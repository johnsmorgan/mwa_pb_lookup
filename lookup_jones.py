#!/usr/bin/env python
import os
import logging
import numpy as np
from h5py import File
from optparse import OptionParser #NB zeus does not have argparse!

from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from lookup_beam import trap, coarse_range, mhz_to_index_weight, get_meta, tidy_spline, header_to_pixel_radec, radec_to_altaz

N_POL = 4
POLS = ("xx", "xy", "yx", "yy")
#https://archive.stsci.edu/fits/users_guide/node87.html
STOKES = (-5, -7, -8, -6)

def get_avg_beam_spline(beam_file, gridnum, low_index, n_freq, weights):
    assert beam_file['beams'].shape[2] == N_POL, "Beam file does not contain 4 polarisations. Not a Jones matrix file!"
    beam_xy = np.sum(np.nan_to_num(beam_file['beams'][gridnum, low_index:low_index+n_freq, ...])*weights.reshape(n_freq, 1, 1, 1),
                     axis=0)
    # Note that according to the docs, x, y should be
    # "1-D arrays of coordinates in strictly ascending order."
    # However this seems to work
    beams = {}
    for p, pol in enumerate(POLS):
        beams[pol] = {}
        for comp in ('r', 'i'):
            if comp == 'r':
                b = RectBivariateSpline(x=beam_file['alt_scale'][...], y=beam_file['az_scale'][...], z=beam_xy[p].real)
            else:
                b = RectBivariateSpline(x=beam_file['alt_scale'][...], y=beam_file['az_scale'][...], z=beam_xy[p].imag)
            beams[pol][comp] = tidy_spline(b, np.float32)
    return beams


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
    parser.add_option("--metafits_suffix", dest="metafits_suffix", default=".metafits",  help="metafits suffix (default %default")
    parser.add_option("--delete", action="store_true", dest="delete", help="delete output files if they already exist")
    parser.add_option("-w", "--wsclean_names", action="store_true", dest="wsclean_names", help="use upper case polarisation designations ('XX', 'XY' etc.) and no 'r' for real")

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

    for pol in POLS:
        for c in ('r', 'i'):
            if not opts.wsclean_names:
                out = "%s%s%s%s" % (out_prefix, pol, c, out_suffix)
            else:
                out = "%s%s%s%s" % (out_prefix, pol.upper(), c if c=='i' else '', out_suffix)
            if os.path.exists(out):
                if opts.delete:
                    os.remove(out)
                else:
                    raise RuntimeError, "%s exists" % out

    # get metadata
    logging.debug("getting metadata")
    gridnum, t = get_meta(obsid, opts.metafits_suffix)

    #open beam file
    logging.debug("generate spline from beam file")
    df = File(opts.beam_path)
    if opts.chan_str is not None:
        low_index, n_chan = coarse_range(df['chans'][...], opts.chan_str)
        weights = trap(n_chan)
        logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+n_chan], weights)
        beams = get_avg_beam_spline(df, gridnum, low_index, n_chan, weights)
    else:
        low_index, weight1 = mhz_to_index_weight(df['chans'][...], opts.freq_mhz)
        weights = np.array((weight1, 1-weight1))
        logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+2], weights)
        beams = get_avg_beam_spline(df, gridnum, low_index, 2, weights)

    hdus = fits.open(obsid+suffix)
    header = hdus[0].header
    data = hdus[0].data
    logging.debug("calculate pixel ra, dec")
    ra, dec = header_to_pixel_radec(header)
    logging.debug("convert to az el")
    alt, az = radec_to_altaz(ra, dec, t)

    # attempt to locate stokes axis
    stokes_axis = None
    try:
        stokes_axis = [header['CTYPE%d' % (i+1)] for i in range(header['NAXIS'])].index('STOKES') + 1
    except ValueError:
        logging.warn("STOKES axis can't be found")

    # store metadata in fits header
    hdus[0].header['PBVER'] = df.attrs['VERSION']
    hdus[0].header['PBPATH'] = opts.beam_path
    hdus[0].header['PBTIME'] = t.isot
    hdus[0].header['PBGRIDN'] = gridnum

    # get values for each fits image pix
    for p, pol in enumerate(POLS):
        for comp in ('r', 'i'):
            if stokes_axis is not None:
                     hdus[0].header['CRVAL%d' % stokes_axis] = STOKES[p]
            hdus[0].header['COMPLEX'] = 'REAL' if comp=='r' else 'IMAG'
            logging.debug("interpolating beams for %s%s", pol, comp)
            beam = beams[pol][comp](alt, az, data.shape)
            logging.debug("writing %s%s beam to disk", pol, comp)
            hdus[0].data = beam
            if not opts.wsclean_names:
                out = "%s%s%s%s" % (out_prefix, pol, comp, out_suffix)
            else:
                out = "%s%s%s%s" % (out_prefix, pol.upper(), comp if comp=='i' else '', out_suffix)
            hdus.writeto(out)
    logging.debug("finished")
