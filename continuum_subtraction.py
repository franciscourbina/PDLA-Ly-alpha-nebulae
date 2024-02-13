"""
This following file implements two different methods for continuum subtraction:
1) Classical broad median filtering approach, for dealing with nearby non-qso sources. 
2) QSO continuum subtraction: this subtracts the QSO spectrum at each spaxel, in a simillar fashion
as done in North (2017). This approach should be good for inner regions. Before doing this, 
a list of DLA systems (redshift and column density) and continuum fitting is needed.
"""
from scipy.interpolate import splev, splev 
import numpy as np
import pickle


def qso_subtraction(cube, wave_masked, qso_mask=None):
    """
    Performs QSO subtraction at the position of a given mask. 
    wave_masked [np.array] = array of wavelength in A, where the scaling factor is calculated
    
    Notice wave_masked has to be in a flat region of the spectrum of the QSO and continuum
    model has to go cover those regions. Also, ideally, wave_masked needs to be using the same sampling
    as in the cube (i.e. 1.25 A). 
    """
    wave = cube.wavearr
    delta_wave = wave_masked[1] - wave_masked[0]
    continuum_eval = cube.continuum_eval(wave_masked)
    
    mask_1 = wave >= wave_masked[0]
    mask_2 = wave <= wave_masked[-1]
    mask = mask_1 * mask_2

    h = np.sum(cube[mask,:,:], axis=0)*1.25 / (np.sum(continuum_eval)* delta_wave)

    cube.data = cube.data - h[np.newaxis,:,:] *  cube.continuum_eval(wave)[:, np.newaxis, np.newaxis]

    