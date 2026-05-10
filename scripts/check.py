from astropy.io import fits

# Open the FITS file
hdul = fits.open(r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\raw\main_sequence\spec-0266-51630-0038.fits")

# See what's inside (displays a summary of the HDUs)
hdul.info()