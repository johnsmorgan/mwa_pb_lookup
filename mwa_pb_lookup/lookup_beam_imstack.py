#!/usr/bin/env python
import os
import logging
import numpy as np
from h5py import File
from optparse import OptionParser #NB zeus does not have argparse!

from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Longitude, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u

from lookup_beam import trap, coarse_range, mhz_to_index_weight, get_meta, get_avg_beam_spline, header_to_pixel_radec, radec_to_altaz, LOCATION

N_POL = 2
POLS = ("XX", "YY")

if __name__ == '__main__':
    try:
        PB_FILE = os.environ['MWA_PB_LOOKUP']
    except KeyError:
        PB_FILE = ""

    parser = OptionParser(usage="usage: image_stack metafits chanstr" +
                          """
                            produce beam for  image stack
                          """)
    parser.add_option("-v", "--verbose", action="count", default=0, dest="verbose", help="-v info, -vv debug")
    parser.add_option("--beam_path", default=PB_FILE, dest="beam_path", type="str", help="path to hdf5 file containing beams")
    parser.add_option("--overwrite", action="store_true", dest="overwrite", help="delete output files if they already exist")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.error("incorrect number of arguments")
    imstack_path = args[0]
    metafits = args[1]
    chan_str = args[2]

    if opts.verbose == 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
    elif opts.verbose > 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)

    # get metadata
    logging.debug("getting metadata")
    obsid, metafits_suffix = os.path.splitext(metafits)
    # FIXME Use central time of image_stack rather than centre point of observation?
    gridnum, t = get_meta(obsid, metafits_suffix)

    #open beam file
    logging.debug("generate spline from beam file")
    df = File(opts.beam_path, 'r')

    low_index, n_chan = coarse_range(df['chans'][...], chan_str)
    weights = trap(n_chan)
    logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+n_chan], weights)
    beams = get_avg_beam_spline(df, gridnum, low_index, n_chan, weights)

    with File(imstack_path, 'a') as imstack:
        group = imstack[chan_str]
        data_shape = list(group['image'].shape)
        logging.debug("data_shape %s", data_shape)
        beam_shape = data_shape[:-1] + [1] # just one beam for all timesteps for now
        logging.debug("beam_shape %s", beam_shape)
        if "beam" in group.keys():
            assert group['beam'].shape == tuple(beam_shape), "Error, beam already exists and is the wrong shape %s %s" % (group['beam'].shape, beam_shape)
            if opts.overwrite:
                logging.warn("Overwriting existing beam")
            else:
                raise RuntimeError("Beam already exists. User --overwrite to overwrite")

            beam = group['beam']
        else:
            beam = group.create_dataset("beam", beam_shape, dtype=np.float32, compression='lzf', shuffle=True)
        logging.debug("calculate pixel ra, dec")
        ra, dec = header_to_pixel_radec(group['header'].attrs)
        logging.debug("convert to az el")
        alt, az = radec_to_altaz(ra, dec, t)

        # store metadata in fits header
        beam.attrs['PBVER'] = df.attrs['VERSION']
        beam.attrs['PBPATH'] = opts.beam_path
        beam.attrs['PBTIME'] = t.isot
        beam.attrs['PBGRIDN'] = gridnum

        # get values for each image pix
        for p, pol in enumerate(POLS):
            logging.debug("interpolating beams for %s", pol)
            beam[p] = beams[pol](alt, az, beam_shape[1:])
        logging.debug("closing hdf5 file")
    logging.debug("finished")
