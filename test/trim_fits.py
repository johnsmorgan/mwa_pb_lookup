from numpy import zeros, float32
from astropy.io import fits
"""
set image data to zero and keep only first HDU from metafits file to save space
"""
IN_IMAGE="1147313992_121-132-YY-image.fits"
OUT_IMAGE="1147313992_121-132_proto.fits"

IN_META="1147313992_full.metafits"
OUT_META="1147313992.metafits"

hdus = fits.open(IN_IMAGE)
hdus[0].data = zeros(hdus[0].data.shape, dtype=float32)
hdus.writeto(OUT_IMAGE)

hdus = fits.open(IN_META)
hdul = fits.HDUList([hdus[0]])
hdul.writeto(OUT_META)
