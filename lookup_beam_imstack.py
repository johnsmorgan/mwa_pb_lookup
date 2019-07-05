#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
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
from lookup_beam import trap, get_avg_beam_spline, coarse_range, mhz_to_index_weight, get_meta, tidy_spline, header_to_pixel_radec, ra_to_ha

from matplotlib import pyplot as plt


N_POL=2
POLS = ("XX", "YY")

def plot_beam(beam, filename):
    # get values for each fits image pix
    plt.clf()
    plt.imshow(beam)
    plt.colorbar()
    plt.savefig(filename)

def get_meta(metapath):
    meta_hdus = fits.open(metapath)
    gridnum = meta_hdus[0].header['GRIDNUM']
    start_lst = meta_hdus[0].header['LST']
    lst = start_lst + 360*meta_hdus[0].header['Exposure']/86164.1
    return gridnum, lst

if __name__ == '__main__':
    try:
        PB_FILE = os.environ['MWA_PB_LOOKUP']
    except KeyError:
        PB_FILE = ""

    parser = OptionParser(usage="usage: imstack_in metafits_in" +
                          """
                          read imstack and metafits and produce XX and YY beams with correct dimensions 

                          top level groups in hdf5 file must be coarse channel range (e.g. 121-132) 

                          A path to the lookup table must also be specified, either via --beam_path or via the global variable MWA_PB_LOOKUP.
                          """)
    parser.add_option("--beam_path", default=PB_FILE, dest="beam_path", type="str", help="path to hdf5 file containing beams")
    parser.add_option("-v", "--verbose", action="count", dest="verbose", help="-v info, -vv debug")
    parser.add_option("-n", "--dry-run", action="store_true", dest="dry_run", help="don't actually write beam, open imstack read only")
    parser.add_option("--no_overwrite", action="store_true", dest="no_overwrite", help="don't overwrite an existing beam (overwrites by default)")
    parser.add_option("--time_offset_seconds", type="int", dest="time_offset", help="offset in seconds from start of observations to generate beam for (default mid-point of observation)")
    parser.add_option("--plot", action="store_true", dest="plot", help="plot both beams to png file")

    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.error("incorrect number of arguments")
    imstack_in = args[0]
    metafits_in = args[1]

    if opts.verbose == 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.INFO)
    elif opts.verbose > 1:
        logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)


    # get metadata
    logging.debug("getting metadata")
    gridnum, lst = get_meta(metafits_in)
    if opts.dry_run:
        mode = 'r'
    else:
        mode = 'r+'

    #open beam file
    logging.debug("opening image stack")
    with File(imstack_in, mode) as imstack:
        logging.info("opening beam file")
        df = File(opts.beam_path)
        for groupname in imstack.keys():
            logging.debug("generating beams for freqs %s" % groupname)
            group = imstack[groupname]
            data_shape = list(group['image'].shape)
            beam_shape = data_shape[:-1] + [1] # just one beam for all timesteps for now
            if not 'beam' in group.keys():
                logging.info("creating beam dataset")
                if not opts.dry_run:
                    beam = group.create_dataset("beam", beam_shape, dtype=np.float32, compression='lzf', shuffle=True)
            else:
                if opts.no_overwrite:
                    raise RuntimeError, "beam dataset already exists and no_overwrite set!"
                logging.warn("beam dataset already exists")
                beam = group['beam']
                assert beam.shape == beam_shape, "beam shape is incorrect"
            logging.debug("hdf5 beam shape %s" % beam_shape)
            low_index, n_chan = coarse_range(df['chans'][...], groupname)
            weights = trap(n_chan)
            logging.info("averaging channels %s Hz with weights %s", df['chans'][low_index:low_index+n_chan], weights)
            beams = get_avg_beam_spline(df, gridnum, low_index, n_chan, weights)

            logging.debug("calculate pixel ra, dec")
            ra, dec = header_to_pixel_radec(group['header'].attrs)
            logging.debug("convert to ha")
            ha = ra_to_ha(ra, lst)

            # store metadata in fits header
            if not opts.dry_run:
                beam.attrs['PBVER'] = df.attrs['VERSION']
                beam.attrs['PBPATH'] = opts.beam_path
                beam.attrs['PBLST'] = lst
                beam.attrs['PBGRIDN'] = gridnum

            # get values for each fits image pix
            for pol in POLS:
                logging.debug("interpolating beams for %s" % pol)
                #FIXME need to figure out dimensions!!!!
                interp_beam = beams[pol](dec, ha, beam_shape[1:])
                if not opts.dry_run:
                    logging.debug("writing %s beam to disk" % pol)
                    beam[0] = interp_beam
                if opts.plot == True:
                    logging.debug("plotting %s" % pol)
                    plot_beam(interp_beam.reshape(beam_shape[1:3]), imstack_in.replace('.', '_')+'_beam_'+pol+'.png')
            logging.debug("finished")
