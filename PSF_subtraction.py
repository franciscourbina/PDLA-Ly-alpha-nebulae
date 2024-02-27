"""
This script implements classical empirical PSF subtraction methods as described in Borisova et al. (2016) including
a variation in the PSF construction that avoids Ly-alpha regions to construct the PSF. 
"""

import numpy as np
import astropy.io.fits as F

def create_empirical_psf(cube, index, wave_mask, width=150):
    """
    Creates NB band image to estimate an empirical PSF. The chosen index should be located
    in QSO continuum as close as possible to the wavelength where the extended emission is expected.
    i.e. Ly-alpha, CIV, HeII, etc. 

    wave_mask[np.array] = binary mask with the same shape in the wavelenght direction telling where
                        the construction of the NB images are going to be performed. This is done with
                        the purpose of avoid subtracting the extended emission.
    """
    nw, nx, ny = cube.shape
    
    minimum_index = (index - int(width/2) >= 0) * (index - int(width/2))
    maximum_index = (index + int(width/2) < nw) * (index + int(width/2)) + (index + int(width/2) >= nw) * (nw - 1) 

    ind_mask = (wave_mask <= wave_mask[maximum_index]) * (wave_mask >= wave_mask[minimum_index])
    total_mask = ind_mask * wave_mask
    total_mask = total_mask.astype(bool)

    return np.sum(cube[total_mask,:,:], axis=0)
    

def subtract_PSF(cube, qso_pos,  wave_mask, RMS_values=0, delta_x=0.2, seeing=0.6, width=150, lower_limit_sigma=3):
    
    subtracted_cube = np.zeros_like(cube)
    
    nw, nx, ny = cube.shape

    radius = 5 * seeing
    radius_px = int(radius/delta_x)

    for i in range(nw):
        
        PSF = create_empirical_psf(cube, i, wave_mask, width=width)
    
        # Calculating flux in 1'' x 1'' area
        norm = np.sum(cube[i, qso_pos[0] - 2: qso_pos[0] + 3, qso_pos[1] - 2: qso_pos[1] + 3])
        norm_PSF = np.sum(PSF[qso_pos[0] - 2: qso_pos[0] + 3, qso_pos[1] - 2: qso_pos[1] + 3])
        # Using sigma-clipped average

        #mean_data = sigma_clipped_stats(cube[i, qso_pos[0] - 2: qso_pos[0] + 3, qso_pos[1] - 2: qso_pos[1] + 3])[0]
        #mean_psf = sigma_clipped_stats(PSF[qso_pos[0] - 2: qso_pos[0] + 3, qso_pos[1] - 2: qso_pos[1] + 3])[0]

        # Create circular mask
        xx, yy = np.mgrid[:nx, :ny]
        circle = (xx - qso_pos[0]) ** 2 + (yy - qso_pos[1]) ** 2
        circle = circle < (radius_px)**2

        # Avoid oversubtraction
        #pre_sub = (cube[i,:,:] - (norm/norm_PSF) * PSF * circle ) * circle  
        
        #mask_over_sub = pre_sub < -lower_limit_sigma * RMS_values[i]


        subtracted_cube[i,:,:] = cube[i,:,:] - (norm/norm_PSF) * PSF * circle 

        if i % 500 ==0:
            print('Progress: {}'.format(int((i+1)*100/nw)))
    return subtracted_cube