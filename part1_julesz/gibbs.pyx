# gibbs_sampler.pyx

import numpy as np
cimport numpy as np
import random

from astropy.units.quantity_helper.function_helpers import histogram
from libc.math cimport exp, fabs
from pyarrow import float64
from spyder_kernels.utils.iofuncs import DTYPES

# Type declarations for performance
ctypedef np.float64_t DTYPE_f
ctypedef np.int32_t DTYPE_i

# Define the convolution function in Cython
def conv(np.ndarray[DTYPE_f, ndim=2] image, np.ndarray[DTYPE_f, ndim=2] filter):
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int fH = filter.shape[0]
    cdef int fW = filter.shape[1]
    cdef int pad_h = fH // 2
    cdef int pad_w = fW // 2
    cdef np.ndarray[DTYPE_f, ndim=2] filtered_image = np.zeros_like(image, dtype=np.float64)

    cdef int i, j
    cdef np.ndarray[DTYPE_f, ndim=2] padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    #cdef np.ndarray[DTYPE_f, ndim=2] padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
    cdef np.ndarray[DTYPE_f, ndim=2] neigh
    for i in range(H):
        for j in range(W):
            neigh = padded_image[i:i + fH, j:j + fW]
            filtered_image[i, j] = np.sum(neigh * filter)

    return filtered_image

# Define the convolution for position in Cython
def conv_pos(np.ndarray[DTYPE_f, ndim=2] padded_image, int H, int W,
             np.ndarray[int, ndim=1] pos, np.ndarray[DTYPE_f, ndim=2] filter,
             DTYPE_f max_response, DTYPE_f min_response, int bins_num):
    cdef int fH = filter.shape[0]
    cdef int fW = filter.shape[1]
    cdef np.ndarray[DTYPE_f, ndim=1] bin_edges = np.linspace(min_response, max_response, bins_num + 1)
    cdef np.ndarray[DTYPE_i, ndim=2] histogram_change = np.zeros((8, bins_num), dtype=np.int32)
    cdef np.ndarray[DTYPE_f, ndim=2] zero_filter
    cdef np.ndarray[DTYPE_f, ndim=2] neigh
    cdef DTYPE_f response_base
    cdef DTYPE_f response
    cdef int index
    cdef int di, dj, i, j
    cdef int num_influenced = 0
    for di in range(-(fH-1), 1):
        i = pos[0] + di
        if i < 0 or i >= H:
            continue
        for dj in range(-(fW-1), 1):
            j = pos[1] + dj
            if j < 0 or j >= W:
                continue
            num_influenced +=1
            zero_filter = filter.copy()
            zero_filter[-di, -dj] = 0
            #print('zero_filter',zero_filter)
            neigh = padded_image[i:i + fH, j:j + fW]
            #print('the topleft of filtered',i, j)
            #print('neigh', neigh)
            response_base = np.sum(zero_filter * neigh, dtype=np.float64)
            for k in range(8):
                response = response_base + k * filter[-di, -dj]

                index = np.digitize(response, bin_edges, right=False) - 1
                #print(f'{response} in {index} index')
                if index < 0:
                    index = 0
                if index >= bins_num:
                    index = bins_num - 1
                #histogram_change[k, index] += 1 / (H * W)
                histogram_change[k, index] +=1
    #print('num_influenced:', num_influenced)
    return histogram_change



def pos_gibbs_sample_update(np.ndarray[DTYPE_f, ndim= 2] img_syn,
                            np.ndarray[DTYPE_i, ndim=2] hists_syn,
                            np.ndarray[DTYPE_f, ndim=2] img_ori,
                            np.ndarray[DTYPE_i, ndim=2] hists_ori,
                            object filter_list,
                            np.ndarray[DTYPE_f, ndim = 2] bounds,
                            np.ndarray[DTYPE_f, ndim = 1] weight,
                            np.ndarray[int, ndim = 1] pos,
                            int num_bins, DTYPE_f T):
    cdef int H = img_syn.shape[0]
    cdef int W = img_syn.shape[1]
    cdef int pos_h = pos[0]
    cdef int pos_w = pos[1]
    cdef int num_chosen_filters = len(filter_list)
    cdef list hists_new = []
    for _ in range(8):
        hists_new.append(np.zeros_like(hists_syn))

    cdef int orig_value = int(img_syn[pos_h, pos_w])
    cdef np.ndarray[DTYPE_f, ndim=1] energy = np.zeros(8)
    cdef int fH, fW
    cdef int pad_h
    cdef int pad_w
    cdef np.ndarray[DTYPE_f, ndim=2] padded_image
    cdef np.ndarray[int, ndim=1] padded_pos
    #cdef np.ndarray[DTYPE_f, ndim=2] histogram_change
    cdef np.ndarray[DTYPE_i, ndim=2] histogram_change
    cdef np.ndarray[DTYPE_f, ndim=2] new_img = img_syn.copy()
    for j in range(num_chosen_filters):
        fH, fW = filter_list[j].shape
        pad_h = fH // 2
        pad_w = fW // 2
        padded_image = np.pad(img_syn, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        #padded_image = np.pad(img_syn, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
        padded_pos = np.array([pos_h + pad_h, pos_w + pad_w])
        #print(img_syn)
        #print(padded_image)
        #print(padded_pos)

        histogram_change = conv_pos(padded_image, H, W, padded_pos, filter_list[j].astype(np.float64), bounds[j][0],
                                                                 bounds[j][1], num_bins)
        #print('change:', histogram_change)
        for i in range(8):
            if i == orig_value:
                hists_new[i][j] = hists_syn[j]

            else:
                hists_new[i][j] = hists_syn[j] - histogram_change[orig_value] + histogram_change[i]
            #print(f'this time change for {i} {histogram_change[i]-histogram_change[orig_value]}')
            #new_img[pos_h, pos_w] = float(i)
            #print(new_img)
            #print(hists_new[i][j])
            #print(get_histogram(conv(new_img, filter_list[j].astype(np.float64)), num_bins, bounds[j, 0], bounds[j, 1], H,W))
            #print('pass check')
            #assert np.array_equal(hists_new[i][j], get_histogram(conv(new_img, filter_list[j].astype(np.float64)), num_bins, bounds[j, 0], bounds[j, 1], H, W))




    for i in range(8):
        energy[i] = (np.abs(hists_new[i] - hists_ori) @ weight).sum()

    if energy.min()!=energy.max():
        energy = (energy- energy.min()) / (energy.max() - energy.min())
    else:
        energy = np.ones(8)



    cdef np.ndarray[DTYPE_f, ndim=1] probs = np.exp(-energy / T)
    cdef np.ndarray[DTYPE_f, ndim=1] probs_cdf = np.cumsum(probs)

    cdef DTYPE_f eps = 1e-10
    probs += eps
    probs /= probs.sum()

    cdef DTYPE_f x = random.random()
    cdef int new_value
    for i in range(8):
        if x < probs_cdf[i]:
            new_value = i
            break
    cdef np.ndarray[DTYPE_f, ndim=2] return_filtered
    img_syn[pos_h, pos_w] = float(new_value)
    #assert (hists_new[new_value]>=0).all()


    '''for j in range(num_chosen_filters):
        if hists_new[new_value][j].min() < 0 :
            print('accum_error')
            #return_filtered = conv(img_syn, filter_list[j].astype(np.float64))
            #hists_new[new_value][j] = get_histogram(return_filtered, num_bins, bounds[j][0], bounds[j][1], H, W)'''


    return img_syn, hists_new[new_value]



# Cython version of get_histogram
def get_histogram(np.ndarray[DTYPE_f, ndim=2] filtered_image,
                  int bins_num,
                  DTYPE_f max_response,
                  DTYPE_f min_response,
                  int img_H,
                  int img_W):


    # Create bin edges for the histogram
    cdef np.ndarray[DTYPE_f, ndim=1] bin_edges = np.linspace(min_response, max_response, bins_num + 1, dtype=np.float64)

    # Flatten the image to a 1D array
    cdef np.ndarray[DTYPE_f, ndim=1] flattened_image = filtered_image.flatten()

    # Initialize a histogram array with zeros
    cdef np.ndarray[DTYPE_i, ndim=1] histogram = np.zeros(bins_num, dtype=np.int32)

    # Count the occurrences of each bin
    cdef int i, bin_index
    for i in range(flattened_image.shape[0]):
        bin_index = np.digitize(flattened_image[i], bin_edges, right=False) - 1
        #print(f'true {flattened_image[i]} in {bin_index} index')
        if bin_index == bins_num:  # Handle the case where the value falls in the last bin
            bin_index = bins_num - 1
        elif bin_index == -1:  # Handle the case where the value is below the first bin edge
            bin_index = 0

        histogram[bin_index] += 1

    # Normalize the histogram
    normalized_histogram = histogram.astype(np.float64) / (img_H * img_W)
    return normalized_histogram
    #return histogram



