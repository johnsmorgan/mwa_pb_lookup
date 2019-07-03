## Overview
This package provides an alternative, far quicker way of generating MWA primary beams 

The speedup is enabled by pre-computing all of the beams of interest over the whole sky in coordinates of Hour Angle and Declination. Generating the primary beam for a particular observation is then simply a matter of deriving the HA and Dec. of each point of interest and interpolating.

Scripts for doing this pre-computation is included in the package. However if you are using standard frequencies (such as the GLEAM frequencies) it is likely that this precomputation has already been done for you, and may even be available on the system you use for data reduction. Speak to your collaborators!

## Generating a primary beam for an existing fits image
This can be done with `lookup_beam.py`. `lookup_beam.py -h` will provide useful documentation. In addition to a fits image file you will also need a lookup file (path provided on the command line or via a global variable), and a metafits file.

## Versions of the primary beam correction.
### 2019-07-03
At the time of writing, the MWA Embedded Primary Beam model published by [Sokolowski et al. (2017)](http://adsabs.harvard.edu/abs/2017PASA...34...62S) is used almost universally for correction of primary beam effects in MWA data. There are various software packages and libraries which carry out the calculations described in the paper. All use the same set of underlying data derived from a set of simulations, which is stored in an hdf5 file which has a version number '02'.

Previous primary beam models have been used in the past, but these have now been superceded by the embedded model for several years.

#### Traceability
All fits files produced by this software currently carry the following metadata in the fits header:
PBVER   = '02      '                                                            
PBPATH  = '/PATH/TO/LOOKUP_FILE'
PBLST   =    Local sidereal time used to generate the primary beam
PBGRIDN =    Gridnum or 'sweetspot number' of the beamformer delay settings used for the beam. 

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
