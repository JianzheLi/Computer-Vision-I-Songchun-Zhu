''' 
This is the main file of Part 1: Julesz Ensemble
'''

from numpy.ctypeslib import ndpointer
import numpy as np
from filters import get_filters
import cv2
from torch.nn.functional import conv2d, pad
import torch
from gibbs import gibbs_sample


def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Notice: Choose your padding method!
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape (x, x)
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    # TODO

    return filtered_image 

def get_histogram(filtered_image,bins_num, max_response, min_response, img_H, img_W):
    ''' 
    Computes the normalized filter response histogram on an image.
    Parameters:
        1. filtered_image: numpy array of shape (H, W)
        2. bins_num: int, the number of bins
        3. max_response: int, the maximum response of the filter
        4. min_response: int, the minimum response of the filter
    Return:
        1. histogram: histogram (numpy array)
    '''

    # TODO

    return histogram

def julesz(img_size = 64, img_name = "fur_obs.jpg", save_img = True):
    ''' 
    The main method
    Parameters:
        1. img_size: int, the size of the image
        2. img_name: str, the name of the image
        3. save_img: bool, whether to save intermediate results, for autograder
    '''
    max_intensity = 255

    # get filters
    F_list = get_filters()
    F_list = [filter.astype(np.float32) for filter in F_list]

    # selected filter list, initially set as empty
    filter_list = []


    # size of image
    img_H  = img_W = img_size


    # read image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity 

    # store the original image
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))

    # synthesize image from random noise
    img_syn = np.random.randint(0,8,img_ori.shape).astype(np.float32)

    # TODO
    max_error = 0 # TODO
    threshold = 0 # TODO

    round = 0
    print("---- Julesz Ensemble Synthesis ----")
    while max_error > threshold: # Not empty

        # TODO


        # save the synthesized image
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)
        round += 1
    return img_syn  # return for testing
    
if __name__ == '__main__':
    julesz()
