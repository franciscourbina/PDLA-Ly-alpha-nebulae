"""
The following script can detect extended emission by using a connected labelling algorithm. The largest group is then identified to be the detected extended emission unless 
the voxel detections is too low, in that case, we also check for it to be relatively close to the central  QSO. 
This set of functions also applies gaussian convolutions before doing applying the detection procedure. 
"""
import cc3d
from scipy.signal import convolve2d
import numpy as np

# Convolution functions
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def convolve_data_SNR(cube):
    filter = gkern(l=5, sig=2.5) # this is sigma = 0.5" (same as in Borisova's 2016 work)
    
    data_to_conv = np.zeros_like(cube.data)
    var_to_conv = np.zeros_like(cube.variance)
    
    N = cube.data.shape[0]
    for i in range(N):
        data_to_conv[i,:,:] = convolve2d(cube.data[i,:,:], filter, mode='same')
        var_to_conv[i,:,:] = convolve2d(cube.variance[i,:,:], filter**2, mode='same')
    
    cube.data = data_to_conv
    cube.variance = var_to_conv

def extract_k_largest(labels, k):
    N = np.max(labels)
    
    if k > N:
        return 0
    
    numbers = np.zeros(N)
    for i in range(N):
        numbers[i] = np.sum(labels == i+1)
    
    return np.argsort(numbers)[-k:] + 1 , numbers[np.argsort(numbers)[-k:]]

def segmentation_mask(SNR_cube, SNR, vox_thr):
    binary = SNR_cube >= SNR

    labels_dust = cc3d.dust(
            binary, threshold=vox_thr, 
            connectivity=26, in_place=False) # connectivity = 26 means we connect voxels in both spectral and spatial direction
        
    labels_out = cc3d.connected_components(binary*labels_dust, connectivity=26)
    
    max_label = extract_k_largest(labels_out, 1)

    if max_label != 0:
        max_label = extract_k_largest(labels_out, 1)[0]
        return ( (labels_out == max_label).astype(int), labels_out)
    else: 
        return 0


def iterative_detection(SNR_cube, max_snr, starting_thr, min_snr=1.3, delta_snr=0.05, delta_thr=50, test_overlap=True):
    
    SNR = max_snr
    threshold = starting_thr
    
    SNR_list = []
    voxels_list = []
    
    detected_voxels = 0

    while detected_voxels  < 1:
        
        SNR_list.append(SNR)
        print('Using a SNR of {0:.2f}.'.format(SNR))
        
        labels_out = segmentation_mask(SNR_cube, SNR, threshold)

        if labels_out != 0:
            provitional_mask = labels_out[0]
            detected_voxels = np.sum(provitional_mask)
            voxels_list.append(detected_voxels)
            print('Largest number of connected voxels: {}'.format(detected_voxels))

        else:
            voxels_list.append(0)
            print('No significant detection obtained at this SNR.')
        
        SNR -= delta_snr

    while SNR >= min_snr:
        
        SNR_list.append(SNR)
        print('Using a SNR of {}.'.format(SNR))
        
        mask = segmentation_mask(SNR_cube, SNR, threshold)[0]
        detected_voxels = np.sum(mask)
        # Checking intersection! 
        intersection = mask * provitional_mask
        cardinal = np.sum(intersection) 
        overlapping = cardinal/np.sum(provitional_mask)

        print('Intersection %: {}'.format(overlapping*100))
        
        k = 2
        if test_overlap:
            while overlapping < 0.95:
                print('Warning: new mask does not contain the old mask!\nChanging group.')
                labels = segmentation_mask(SNR_cube, SNR, threshold)[1]
                numbers = extract_k_largest(labels, k)[0]
                mask = (labels == numbers[-k]).astype(int)
                    
                intersection = mask * provitional_mask
                cardinal = np.sum(intersection) 
                overlapping = cardinal/np.sum(provitional_mask)

                print('Intersection %: {}'.format(overlapping*100))
                k += 1
            
        print('Largest number of connected voxels: {}'.format(detected_voxels))
        voxels_list.append(detected_voxels)

        SNR -= delta_snr
        threshold += delta_thr
        provitional_mask = mask

    return SNR_list, voxels_list, mask

c = 299792458/1000 # light speed in km/s 

def velocity(w0, wavearr):
    vel_arr = c *( (wavearr - w0)/w0 ) 
    return vel_arr

def first_second_moment(mask, data_cut, wavearr_cut, w0):
    
    masked_cut = (mask*data_cut)
    
    nebula_no_scale=  np.sum(masked_cut, axis=0)

    masked_cut_square = (mask*data_cut)

    for j in range(masked_cut.shape[0]):
        masked_cut[j,:,:] *= wavearr_cut[j]
        masked_cut_square[j,:,:] *= wavearr_cut[j]**2

    first_moment = np.sum(masked_cut, axis=0)

    for i in range(first_moment.shape[0]):
        for j in range(first_moment.shape[1]):
            if nebula_no_scale[i,j] != 0:
                first_moment[i,j] = first_moment[i,j]/nebula_no_scale[i,j]
            
    first_moment_vel = velocity(w0, first_moment)

    for i in range(first_moment.shape[0]):
        for j in range(first_moment.shape[1]):
            if np.abs(first_moment_vel[i,j]) > 0.9*c:
                first_moment_vel[i,j] = np.nan

    second_moment = np.sqrt(np.sum(masked_cut_square, axis=0)/nebula_no_scale - first_moment**2)
    second_moment_vel = (second_moment/w0)*c
    
    return first_moment, first_moment_vel, second_moment_vel