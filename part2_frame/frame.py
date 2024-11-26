'''
This is the main file of Part 1: Julesz Ensemble
'''

import numpy as np
from filters_frame import get_filters
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from filters_frame import plt_filter
import pyximport
pyximport.install()
from gibbs_frame import (conv_pos, conv, get_histogram, pos_gibbs_sample_update)

mean_error_list = np.array([])
max_error_list = np.array([])
total_iter = 0

def gibbs_sample(
        coeffs, learning_rate, img_syn, hists_syn,
                 img_ori, hists_ori,
                 filter_list, epoch, sweep, bounds,
                 T, num_bins):
    '''
    The gibbs sampler for synthesizing a texture image using annealing scheme
    Parameters:
        0. coeffs: the lambda, shape: [num_chosen_filters, num_bins]
        1. img_syn: the synthesized image, numpy array, shape: [H,W]
        2. hists_syn: the histograms of the synthesized image, numpy array, shape: [num_chosen_filters,num_bins]
        3. img_ori: the original image, numpy array, shape: [H,W]
        4. hists_ori: the histograms of the original image, numpy arrays, shape: [num_chosen_filters,num_bins]
        5. filter_list: the list of selected filters
        6. sweep: the number of sweeps
        7. bounds: the bounds of the responses of img_ori, a array of numpy arrays in shape [num_chosen_filters,2], bounds[x][0] max response, bounds[x][1] min response
        8. T: the initial temperature
        9. weight: the weight of the error, a numpy array in the shape of [num_bins]
        10. num_bins: the number of bins of histogram, a scalar
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
    '''
    global mean_error_list, max_error_list, total_iter
    H, W = (img_syn.shape[0], img_syn.shape[1])
    print(" ---- GIBBS SAMPLING ---- ")
    last_mean = 0
    last_max = 0
    for t in tqdm(range(epoch)):
        grad = (hists_syn - hists_ori) / float(H * W)
        coeffs += learning_rate * grad
        #last_mean_sweep = 0
        #last_max_sweep = 0
        init_T = T
        for s in tqdm(range(sweep)):
            for pos_h in range(H):
                for pos_w in range(W):
                    pos = np.array([pos_h, pos_w])

                    img_syn, hists_syn = pos_gibbs_sample_update(coeffs.copy(), img_syn.copy(), hists_syn.copy(), filter_list,
                                                             bounds, pos, num_bins, init_T)
            error = np.sum(np.abs(hists_syn - hists_ori), axis=1)
            max_error = np.max(error)
            mean_error = np.mean(error)

            init_T *= 0.5
        error = np.sum(np.abs(hists_syn - hists_ori), axis=1)
        max_error = np.max(error)
        mean_error = np.mean(error)
        print(
            f'Gibbs epoch {t + 1}: mean_error = {mean_error} max_error: {max_error}')
        mean_error_list = np.append(mean_error_list, mean_error)
        max_error_list = np.append(max_error_list, max_error)
        total_iter += 1
        learning_rate *=0.9 # 0,96   # annealing?
        if t!=0:
            if np.abs(mean_error - last_mean)/last_mean <0.01 or np.abs(max_error - last_max)/last_max <0.01:
                print(f"Gibbs iteration {t + 1}: max_error: {max_error} convergence, stop!")
                break
        last_mean = mean_error
        last_max = max_error

        if (max_error/(H*W)) < 0.1:
            print(f"Gibbs iteration {t + 1}: max_error: {(max_error/(H*W))} < 0.1, stop!")
            break
    return img_syn, hists_syn, coeffs



def frame(img_size = 64, img_name = "fur_obs.jpg", save_img = True):
    '''
    The main method
    Parameters:
        1. img_size: int, the size of the image
        2. img_name: str, the name of the image
        3. save_img: bool, whether to save intermediate results, for autograder
    '''
    global mean_error_list, max_error_list, total_iter
    mean_error_list = np.array([])
    max_error_list = np.array([])
    total_iter = 0
    max_intensity = 255

    # get filters   step1
    F_list = get_filters()
    F_list = np.array(F_list, dtype = object)
    #F_list = [filter for filter in F_list]



    # selected filter list, initially set as empty   step2
    #filter_list = []


    # size of image
    img_H  = img_W = img_size


    # read image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float64)
    img_ori = img_ori * 7 // max_intensity

    # store the original image
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))

    # synthesize image from random noise  step3
    img_syn = np.random.randint(0,8,img_ori.shape).astype(np.float64)

    # TODO
    #max_error = 0 # TODO
    threshold = 0.3 # TODO

    round = 0
    print("---- Julesz Ensemble Synthesis ----")


    #weight = np.array([8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8])
    #weight = np.array([2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    #weight = np.ones(15)
    epoch = 2
    num_bins = 15
    sweep = 2
    T = 1
    learning_rate =1
    num_filters_total = len(F_list)
    coeffs = np.zeros((num_filters_total, num_bins), dtype=np.float64)
    coeffs_history = np.array([coeffs])
    #coeffs_history +=coeffs
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
              'orange', 'purple', 'brown', 'pink', 'lime',
              'olive', 'teal', 'navy']

    hists_ori = np.zeros((num_filters_total, num_bins),  dtype = np.int32)
    hists_syn = np.zeros((num_filters_total, num_bins),  dtype = np.int32)


    bounds = np.zeros((num_filters_total, 2))
    index_fiters_unselect = np.zeros(num_filters_total)  #0为未选择
    max_error_newfilter = []
    #计算origi图像的histogram
    for i in range(num_filters_total):
        filtered_img_ori = conv(img_ori, F_list[i].astype(np.float64))
        #print(F_list[i])
        #print(filtered_img_ori)
        bounds[i] = np.array([filtered_img_ori.max(), filtered_img_ori.min()])
        #print(bounds[i])
        hists_ori[i] = get_histogram(filtered_img_ori, num_bins, bounds[i, 0], bounds[i, 1], img_H, img_W)
        #print(hists_ori[i])

    while 0 in index_fiters_unselect: # Not empty
    #for _ in range(1):
        # TODO
        error = np.zeros(num_filters_total)
        for i in range(num_filters_total):
            if index_fiters_unselect[i] == 0:
                filtered_img_syn = conv(img_syn, F_list[i].astype(np.float64))
                hists_syn[i] = get_histogram(filtered_img_syn, num_bins, bounds[i, 0], bounds[i, 1], img_H, img_W)
                error[i] = np.sum(np.abs(hists_syn[i] - hists_ori[i]))
        select_index = np.argmax(error)
        max_error = error[select_index]
        max_error_newfilter.append(max_error)
        if (max_error / (img_W * img_H)) < threshold:
            break
        else:
            index_fiters_unselect[select_index] =1
            #plt_filter(F_list[select_index], round, select_index, img_name)
            indices = np.where(index_fiters_unselect == 1)[0]
            #print(F_list[indices])

            img_syn, hists_syn[indices], coeffs[indices] =gibbs_sample(coeffs=coeffs[indices], learning_rate=learning_rate, img_syn=img_syn, hists_syn=hists_syn[indices], img_ori=img_ori, hists_ori = hists_ori[indices].copy(), filter_list=F_list[indices], epoch=epoch, sweep=sweep, bounds = bounds[indices], T = T, num_bins=num_bins)
        # save the synthesized image
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)


        round += 1
    return img_syn  # return for testing

if __name__ == '__main__':
    ##print('starting horse')
    #frame(img_name='horse_obs.jpg')
    print('starting fur')
    frame(img_name='fur_obs.jpg')
    #print('starting mud')
    #julesz(img_name='mud_obs.jpg')
    print('ending julesz')