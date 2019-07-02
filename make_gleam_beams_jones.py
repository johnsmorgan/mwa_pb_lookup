"""
Make all-sky XX and YY power beams for edge of all GLEAM sub-bands and similar for coarse channels typically used for observations around 300MHz.

MWA embedded primary beam model (Sokolowski 2017) is only calculated for the centre of each coarse channel. In order to approximate the band edge,
the beams for the two neighbouring coarse channels are averaged together with equal weighting.

"""
import sys
import re
import json
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline

from glob import glob
from matplotlib import pyplot as plt
from h5py import File
from sweet_dict import delays, sweetspot_number

from primary_beam import MWA_Tile_full_EE
OUT_FILE="/data/other/pb_lookup/gleam_jones.hdf5"

LAT = -26.7
CHANS = ( 56,  57, # bottom edge of GLEAM 69
          62,  63,
          68,  69,
          74,  75,
          80,  81,
          86,  87,
          92,  93,
          98,  99,
         104, 105, # skipping orbcomm
         108, 109,
         114, 115,
         120, 121,
         126, 127,
         132, 133,
         138, 139,
         144, 145,
         150, 151,
         156, 157,
         162, 163,
         168, 169,
         174, 175,
         180, 181, # top edge of GLEAM 169
         222, 223, # bottom of 300MHz band
         228, 229,
         234, 235,
         240, 241,
         246, 247)

CHAN_FREQS = [c*1280000. for c in CHANS]
FREQS = []
for c in range(len(CHANS)/2):
    FREQS.append(640000*(CHANS[2*c]+CHANS[2*c+1]))

N_POL = 4
SWEETSPOTS = range(197)

sweet_dict = json.load(open("sweetspots.json"))
delays = {int(k): v for k, v in sweet_dict['delays'].iteritems()}
assert sorted(delays.keys()) == SWEETSPOTS
# Generate grid of all HA and Declination coordinates
ha_scale = np.linspace(-179.5, 179.5, 360) #this has the advantage of avoiding the singularities at 0 and 180 degrees
dec_scale = np.linspace(-90, 90, 181)
has, decs = np.meshgrid(ha_scale, dec_scale)

# figure out corresponding azimuth and elevation
# for simplicity, use basic trig rather than trying to construct a time when RA0h is precisely on the meridian.
alt = np.degrees(np.arcsin(np.sin(np.radians(decs))*np.sin(np.radians(LAT)) + np.cos(np.radians(decs))*np.cos(np.radians(LAT))*np.cos(np.radians(has))))
# NB This doesn't quite work for HA = 180deg 
az = np.degrees(np.arccos((np.sin(np.radians(decs)) - np.sin(np.radians(alt))*np.sin(np.radians(LAT)))/(np.cos(np.radians(alt))*np.cos(np.radians(LAT)))))
az = np.where(np.sin(np.radians(has)) < 0, az, 360-az)
print alt.shape
print az.shape

num_unique_beams = len(SWEETSPOTS)
#num_unique_beams = N
beam_shape = [num_unique_beams, len(FREQS), N_POL] + list(az.shape)
chunks = tuple([1, 1, N_POL] + list(az.shape))

# pick out just those pixels that are above the horizon
up = alt > 0

# theta phi (and rX, rY) are 1D arrays.
theta = (np.pi/2) - np.radians(alt[up])
phi = np.radians(az[up])
#theta = (np.pi/2) - np.radians(alt.ravel())
#phi = np.radians(alt.ravel())

with File(OUT_FILE) as df:
    # actual beam data
    data = df.create_dataset('beams', beam_shape, chunks=chunks, compression='lzf', shuffle=True, dtype=np.complex64)
    # various metadata
    df['beams'].dims[0].label = 'beam'
    df.create_dataset('sweetspot_number', data=SWEETSPOTS)
    df['beams'].dims.create_scale(df['sweetspot_number'])
    df['beams'].dims[0].attach_scale(df['sweetspot_number'])

    df['beams'].dims[0].label = 'chans'
    df.create_dataset('chans', data=FREQS)
    df['beams'].dims.create_scale(df['sweetspot_number'])
    df['beams'].dims[0].attach_scale(df['sweetspot_number'])

    df['beams'].dims[2].label = 'dec'
    df.create_dataset('dec_scale', data=dec_scale)
    df['beams'].dims.create_scale(df['dec_scale'])
    df['beams'].dims[2].attach_scale(df['dec_scale'])

    df['beams'].dims[3].label = 'ha'
    df.create_dataset('ha_scale', data=ha_scale)
    df['beams'].dims.create_scale(df['ha_scale'])
    df['beams'].dims[3].attach_scale(df['ha_scale'])

    df.create_dataset('delays', data=np.array([delays[i] for i in SWEETSPOTS], dtype=np.uint8))

    shape = beam_shape[2:]
    d1 = np.nan*np.ones(shape, dtype=np.complex64)
    d2 = np.nan*np.ones(shape, dtype=np.complex64)
    # generate beams
    for s in SWEETSPOTS:
    #for s in range(N):
        print "Sweetspot %d" % s
        for f, freq in enumerate(CHAN_FREQS):
            print freq
            #if azel[0] > 45:
            #    continue
            jones = MWA_Tile_full_EE([theta], [phi],
                                     freq=freq, delays=delays[s],
                                     zenithnorm=True, power=True,
                                     jones=True, interp=False).reshape(N_POL, len(theta))
            print jones.shape
            if f % 2:
                d1[:, up] = jones
                data[s, f//2, ...] = (d1 + d2)/2
            else:
                d2[:, up] = jones
