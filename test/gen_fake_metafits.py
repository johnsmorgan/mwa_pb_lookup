from astropy.io import fits
hdu = fits.PrimaryHDU()
hdu.header['LST'] = 0.0
hdu.header['DELAYS'] = "0,"*16
hdu.header['GRIDNUM'] = 0
hdu.header['EXPOSURE'] = 0.0
hdul = fits.HDUList([hdu])
hdul.writeto("0000000000.metafits")
