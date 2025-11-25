''' 
This is file for part 1 
It defines the Gibbs sampler and we use cython for acceleration
'''
from tqdm import tqdm
import numpy as np
import random


def gibbs_sample(img_syn, hists_syn,
                 img_ori, hists_ori,
                 filter_list, sweep, bounds,
                 T, weight, num_bins):
    '''
    The gibbs sampler for synthesizing a texture image using annealing scheme
    Parameters:
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

    H,W = (img_syn.shape[0],img_syn.shape[1])
    num_chosen_filters = len(filter_list)
    print(" ---- GIBBS SAMPLING ---- ")
    for s in tqdm(range(sweep)):
        for pos_h in range(H):
            for pos_w in range(W):
                pos = [pos_h,pos_w]
                img_syn,hists_syn = pos_gibbs_sample_update(img_syn,hists_syn,img_ori,hists_ori,filter_list,bounds,weight,pos,num_bins,T)
        max_error = (np.abs(hists_syn-hists_ori) @ weight).max()
        print(f'Gibbs iteration {s+1}: error = {(np.abs(hists_syn-hists_ori) @ weight).mean()} max_error: {max_error}')
        T = T * 0.96
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error} < 0.1, stop!")
            break
    return img_syn, hists_syn


def pos_gibbs_sample_update(img_syn, hists_syn,
                            img_ori, hists_ori,
                            filter_list, bounds,
                            weight, pos, 
                            num_bins, T):
    '''
    The gibbs sampler for synthesizing a value of single pixel
    Parameters:
        1. img_syn: the synthesized image, a numpy array in shape [H,W]
        2. hists_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        3. img_ori: the original image, a numpy array in shape [H,W]
        4. hists_ori: the histograms of the original image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        5. filter_list: the list of filters, a list of numpy arrays 
        6. bounds: the bounds of the responses of img_ori, a list of numpy arrays in shape [num_chosen_filters,2], in the form of (max_response, min_response)
        7. weight: the weight of the error, a numpy array in the shape of [num_bins]
        8. pos: the position of the pixel, a list of two scalars
        9. num_bins: the number of bins of histogram, a scalar
        10. T: current temperture of the annealing scheme
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
        hist_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
    '''

    H = img_syn.shape[0]
    W = img_syn.shape[1]
    pos_h = pos[0]
    pos_w = pos[1]
    energy = 0

    # calculate the conditional probability: p(I(i,j) = intensity | I(x,y),\forall (x,y) \neq (i,j))
    # perturb (i,j) pixel's intensity

    # TODO
    

    # calculate the energy
    # TODO
    
    probs = np.exp(-energy/T)
    eps = 1e-10
    probs = probs + eps
    # normalize the probs
    probs = probs / probs.sum()
    # sample the intensity change the synthesized image
    try:
        # inverse_cdf
        # TODO
    except:
        raise ValueError(f'probs = {probs}')

    # update the histograms
    # TODO

    return img_syn,hists_syn


