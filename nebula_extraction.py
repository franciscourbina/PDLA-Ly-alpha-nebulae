"""
This module loads a MUSE cube and will do a full routine of contiuum + PSF subtraction, to 
then subtract any extended emission at Ly-alpha, CIV and HeII wavelengths.
For those purposes, a connected-component algorithm is employed. 

Notice you need to provide 2 masks for every cube. 

1) Mask to reescale the variance cube: this mask has to be as close as possible to the QSO and needs to avoid
any nearby continuum source. With this mask, the RMS surface brightness value is calculated and compared directly
to the variance cube at each wavelength. 

2) Mask of any nearby continuum/emitting source: to avoid contamination from low redshift galaxies or poorly subtracted
continuum sources. 
"""

import continuum_subtraction as csub 
import PSF_subtraction as PSF 
import astropy.io.fits as F
import numpy as np

class extended_emission:
    def __init__(self, path_data_cube, data_ind=1, var_ind=2):
        self.hdu = F.open(path_data_cube)
        self.data = self.hdu[data_ind].data
        self.variance = self.hdu[var_ind].data
        
    def compute_wavearr(self):
        initial_lamb = self.hdu[1].header['CRVAL3']
        wavearr = np.arange(0.0, self.data.shape[0] ,1.0) * 1.25 + initial_lamb
        self.wavearr = wavearr

    def wave_trim(self, lamb0, lamb1):
        pass
    def reescale_variance(self, RMS_mask_path):
        pass
    