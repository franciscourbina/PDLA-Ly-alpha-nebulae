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
import pickle
from scipy.interpolate import splrep, splev
import Voigt as vg

class cube:
    def __init__(self, path_data_cube, z_em, data_ind=1, var_ind=2):
        self.hdu = F.open(path_data_cube)
        self.data = self.hdu[data_ind].data
        self.variance = self.hdu[var_ind].data
        self.z_em = z_em

        # Computing wave arr
        initial_lamb = self.hdu[1].header['CRVAL3']
        wavearr = np.arange(0.0, self.data.shape[0] ,1.0) * 1.25 + initial_lamb
        self.wavearr = wavearr

    def load_spline(self, path_spline):
        # Loading
        data_sp =  pickle.load(open(path_spline, 'rb'), encoding='latin1')
        continuum_dots = data_sp[1]
        continuum_wave = data_sp[0]
        self.spline =  splrep(continuum_wave, continuum_dots)

    def load_systems(self, N_arr, z_arr):
        # We assume b = 10 km/s , this parameter is not relevant as we are dealing with DLAs
        self.N_arr = N_arr
        self.z_arr = z_arr

    def continuum_eval(self, wave):
        exponentials = vg.arrayvoigt(wave, len(self.N_arr)*[10], self.N_arr, self.z_arr)
        continuum = splev(wave, self.spline)
        return exponentials * continuum

    def wave_trim(self, lamb0, lamb1, update=True):
        low_ind = (np.abs(self.wavearr - lamb0)).argmin()
        max_ind = (np.abs(self.wavearr - lamb1)).argmin()

        if update:
            self.data = self.data[low_ind:max_ind,:,:]
            self.wavearr = self.wavearr[low_ind:max_ind]
        else:
            return self.wavearr[low_ind:max_ind], self.data[low_ind:max_ind,:,:]

    def spatial_trim(self, side):
        pass

    def reescale_variance(self, RMS_mask_path):
        pass
    