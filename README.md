## Overview
This package provides an alternative, far quicker way of generating MWA primary beams compared with the [standard mwa_pb code](https://github.com/MWATelescope/mwa_pb).

The speedup is enabled by using the standard code to pre-compute all of the beams of interest over the whole sky in coordinates of Hour Angle and Declination on a rectangular grid. Generating the primary beam for a particular observation is then simply a matter of deriving the HA and Dec. of each point of interest and interpolating.

Users should cite the [MWA primary beam paper](http://adsabs.harvard.edu/abs/2017PASA...34...62S) for the beam model the corrections are ultimately derived from. 

## Lookup files
The precomputed beams are stored in hdf5 format. Example scripts for doing the pre-computation are included in the package.  They may be either 'XX' and 'YY' power beams, or Jones matrices.

If you are using standard frequencies (such as the GLEAM frequencies) it is likely that this precomputation has already been done for you, and may even be available on the system you use for data reduction. Speak to your collaborators!

## Generating XX and YY (power) primary beams for an existing fits image
This can be done with `lookup_beam.py`. `lookup_beam.py -h` will provide useful documentation. In addition to a fits image file you will also need a lookup file (path provided on the command line or via a global variable), and a metafits file.

## Generating Jones for an existing fits image
This can be done with `lookup_jones.py`. `lookup_jones.py -h` will provide useful documentation. In addition to a fits image file you will also need a lookup file (path provided on the command line or via a global variable), and a metafits file.

8 separate fits files will be created, real and imaginary for each of the jones matrix elements.

## Versions of the primary beam correction.
### 2019-07-03
At the time of writing, the MWA Embedded Primary Beam model published by [Sokolowski et al. (2017)](http://adsabs.harvard.edu/abs/2017PASA...34...62S) is used almost universally for correction of primary beam effects in MWA data. There are various software packages and libraries which carry out the calculations described in the paper. All use the same set of underlying data derived from a set of simulations, which is stored in an hdf5 file which has a version number '02'.

Previous primary beam models have been used in the past, but these have now been superceded by the embedded model for several years.

#### Traceability
All fits files produced by this software currently carry the following metadata in the fits header:
```
PBVER   = '02      '                                                            
PBPATH  = '/PATH/TO/LOOKUP_FILE'
PBLST   =    Local sidereal time used to generate the primary beam
PBGRIDN =    'GRIDNUM' or 'sweetspot number' of the beamformer delay settings used for the beam. 
```
The lookup file contains much more detailed information on other parameters that may have some effect on the beam produced (such as the resolution of the interpolation grid). Additionally, all parameters which were used when `primary_beam.MWA_Tile_full_EE` was called are included in the metadata of the file: e.g.

```python
df = h5py.File("path/to/lookup/file")
df.attrs['BIBCODE'] = '2017PASA...34...62S'
df.attrs['VERSION'] = '02'
df['beams'].attrs['zenithnorm'] = True
df['beams'].attrs['power'] = True
df['beams'].attrs['jones'] = True
df['beams'].attrs['interp'] = False
```
