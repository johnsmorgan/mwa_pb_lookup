"""
Make all-sky jones matrices for edge of all GLEAM sub-bands and similar for coarse channels typically used for observations around 300MHz.

MWA embedded primary beam model (Sokolowski 2017) is only calculated for the centre of each coarse channel. In order to approximate the band edge,
the beams for the two neighbouring coarse channels are averaged together with equal weighting.

"""
import os, sys
import json
import numpy as np

from h5py import File
from sweet_dict import delays

from primary_beam import MWA_Tile_full_EE
OUT_FILE=os.path.join(sys.argv[1], "gleam_jones.hdf5")

ZENITHNORM = True
POWER = False
JONES = True
INTERP = False
AREA_NORM = False

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
print delays
assert sorted(delays.keys()) == SWEETSPOTS
# Generate grid of all Az and El coordinates
az_scale = np.linspace(0, 360, 360)
#NB we could make this linear in cos(alt) (\equiv sin(zenith angle), however with 100 points we jump from 0-8 degrees altitude, which seems a little too coarse!
#cosalt_scale = np.linspace(0, 1, 100) # cos of altitude (also sin of zenith angle)
#alt_scale = np.arccos(cosalt_scale)
alt_scale = np.linspace(0, 90, 90) # cos of altitude (also sin of zenith angle)
az, alt = np.meshgrid(az_scale, alt_scale)

num_unique_beams = len(SWEETSPOTS)
#num_unique_beams = N
beam_shape = [num_unique_beams, len(FREQS), N_POL] + list(az.shape)
chunks = tuple([1, 1, N_POL] + list(az.shape))

# theta phi (and rX, rY) are 1D arrays.
#theta = ((np.pi/2) - np.radians(alt)).ravel
#phi = np.radians(az)
theta = (np.pi/2) - np.radians(alt.ravel())
phi = np.radians(alt.ravel())

with File(OUT_FILE) as df:
    # actual beam data
    data = df.create_dataset('beams', beam_shape, chunks=chunks, compression='lzf', shuffle=True, dtype=np.complex64)
    # various metadata
    df.attrs['BIBCODE'] = '2017PASA...34...62S'
    df.attrs['VERSION'] = '02'
    df['beams'].attrs['zenithnorm'] = True
    df['beams'].attrs['power'] = True
    df['beams'].attrs['jones'] = True
    df['beams'].attrs['interp'] = False
    df['beams'].dims[0].label = 'beam'
    df.create_dataset('sweetspot_number', data=SWEETSPOTS)
    df['beams'].dims.create_scale(df['sweetspot_number'])
    df['beams'].dims[0].attach_scale(df['sweetspot_number'])

    df['beams'].dims[1].label = 'chans'
    df.create_dataset('chans', data=FREQS)
    df['beams'].dims.create_scale(df['chans'])
    df['beams'].dims[1].attach_scale(df['chans'])

    df['beams'].dims[2].label = 'alt'
    df.create_dataset('alt_scale', data=alt_scale)
    df['beams'].dims.create_scale(df['alt_scale'])
    df['beams'].dims[2].attach_scale(df['alt_scale'])

    df['beams'].dims[3].label = 'az'
    df.create_dataset('az_scale', data=az_scale)
    df['beams'].dims.create_scale(df['az_scale'])
    df['beams'].dims[3].attach_scale(df['az_scale'])
    df['beams'].attrs['zenithnorm'] = ZENITHNORM
    df['beams'].attrs['power'] = POWER
    df['beams'].attrs['jones'] = JONES
    df['beams'].attrs['interp'] = INTERP
    df['beams'].attrs['area_norm'] = AREA_NORM

    df.create_dataset('delays', data=np.array([delays[i] for i in SWEETSPOTS], dtype=np.uint8))

    shape = beam_shape[2:]
    d1 = np.nan*np.ones(shape, dtype=np.complex64)
    d2 = np.nan*np.ones(shape, dtype=np.complex64)
    # generate beams
    for s in SWEETSPOTS:
        print "Sweetspot %d" % s
        for f, freq in enumerate(CHAN_FREQS):
            print freq
            print delays[s]
            #if azel[0] > 45:
            #    continue

            # theta and phi *must* be 2d arrays, hence square brackets
            # With jones=False (default) MWA_Tile_full_EE returns 
            # a single array. slowest axes are [theta].shape
            # fastest axes are [2, 2] (the jones matrices)
            # We reshape it to have just 2 dimensions
            jones = MWA_Tile_full_EE([theta], [phi],
                                     freq=freq, delays=delays[s],
                                     zenithnorm=ZENITHNORM, power=POWER,
                                     jones=JONES, interp=INTERP).reshape(len(theta), N_POL)
            print jones.shape
            if f % 2:
                d1 = jones.swapaxes(0, 1).reshape(beam_shape[2:])
                data[s, f//2, ...] = (d1 + d2)/2
            else:
                d2 = jones.swapaxes(0, 1).reshape(beam_shape[2:])
