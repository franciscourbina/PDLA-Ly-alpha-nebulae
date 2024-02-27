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
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

def qso_model_subtraction(cube, wave_masked, qso_mask=None, update=True):
    """
    Performs QSO subtraction at the position of a given mask. 
    wave_masked [np.array] = array of wavelength in A, where the scaling factor is calculated
    
    Notice wave_masked has to be in a flat region of the spectrum of the QSO and continuum
    model has to go cover those regions. Also, ideally, wave_masked needs to be using the same sampling
    as in the cube (i.e. 1.25 A). 
    """
    wave = cube.wavearr
    delta_wave = wave_masked[1] - wave_masked[0]
    continuum_eval = cube.model(wave_masked)
    
    mask_1 = wave >= wave_masked[0]
    mask_2 = wave <= wave_masked[-1]
    mask = mask_1 * mask_2

    h = np.sum(cube.data[mask,:,:], axis=0)*1.25 / (np.sum(continuum_eval)* delta_wave)
    if update:
        cube.data = cube.data - h[np.newaxis,:,:] *  cube.model(wave)[:, np.newaxis, np.newaxis]
    else:
        return cube.data - h[np.newaxis,:,:] *  cube.model(wave)[:, np.newaxis, np.newaxis]
    


# Fast median filtering approach

def rebinning_cube(datacube, width=150, filter_width=2):
    nw, nx, ny = datacube.shape
    num_bins =int(np.ceil(nw/width))
    rebinned = np.zeros((num_bins, nx, ny))

    i = 0
    while i < (num_bins - 1):

        rebinned[i,:,:] = np.median(datacube[i*width: (i+1)*width, :, :], axis=0)
        i += 1
        if i % 5 == 0:
            print('Rebinning spectrum: {} % finnished'.format(int(((i+1)/num_bins)*100 ) ))

    rebinned[-1,:,:] = np.median(datacube[i*width:-1, :, :], axis=0)
    
    # Median filtering the result
    print('Finnished rebinning. \nApplying a median filter')

    result = median_filter(rebinned, size=(filter_width, 1, 1))

    return result

def cont_subtract_cube(cube, width=150, filter_width=2):

    nw, nx, ny = cube.data.shape
    fast_median_cube = rebinning_cube(cube.data)
    subtracted_cube = np.zeros_like(cube.data)
    
    for i in range(nw):
        pos = int(np.floor(i/width))
        
        # Avoid over subtraction :) 

        subtracted_cube[i, :, :] = cube[i,:,:] - fast_median_cube[pos, :,:]

    return subtracted_cube
