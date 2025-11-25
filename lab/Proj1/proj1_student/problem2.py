'''
This is the code for project 1 question 2
Question 2: Verify the 1/f power law observation in natural images in Set A
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
path = "./image_set/setA/"
colorlist = ['red', 'blue', 'black', 'green']
linetype = ['-', '-', '-', '-']
labellist = ["natural_scene_1.jpg", "natural_scene_2.jpg",
                 "natural_scene_3.jpg", "natural_scene_4.jpg"]

img_list = [cv2.imread(os.path.join(path,labellist[i]), cv2.IMREAD_GRAYSCALE) for i in range(4)]
def fft(img):
    ''' 
    Conduct FFT to the image and move the dc component to the center of the spectrum
    Tips: dc component is the one without frequency. Google it!
    Parameters:
        1. img: the original image
    Return:
        1. fshift: image after fft and dc shift
    '''
    f = np.fft.fft2(img)
    
    fshift = np.fft.fftshift(f)

    return fshift

def amplitude(fshift):
    '''
    Parameters:
        1. fshift: image after fft and dc shift
    Return:
        1. A: the amplitude of each complex number
    '''

    A = np.abs(fshift)
    
    # TODO: Add your code here

    return A

def xy2r(x, y, centerx, centery):
    ''' 
    change the x,y coordinate to r coordinate
    '''
    rho = math.sqrt((x - centerx)**2 + (y - centery)**2)
    return rho

def cart2porl(A,img):
    ''' 
    Finish question 1, calculate the A(f) 
    Parameters: 
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. f: the frequency list 
        2. A_f: the amplitude of each frequency
    Tips: 
        1. Use the function xy2r to get the r coordinate!
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    basic_f = 1
    max_r = min(centerx, centery)
    f = np.arange(0, max_r + 1, basic_f)  # 频率范围
    
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij') 
    
    
    r = np.sqrt((xx - centerx)**2 + (yy - centery)** 2)
    f_idx = np.round(r).astype(int)  
    
    
    valid_mask = (f_idx >= 0) & (f_idx < len(f))
    valid_f_idx = f_idx[valid_mask]
    valid_A = A[valid_mask]
    
   
    sum_A = np.bincount(valid_f_idx, weights=valid_A, minlength=len(f))
    count = np.bincount(valid_f_idx, minlength=len(f))
    
    
    A_f = np.divide(sum_A, count, out=np.zeros_like(sum_A), where=count != 0)
    
    return f, A_f

def get_S_f0(A,img):
    ''' 
    Parameters:
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. S_f0: the S(f0) list
        2. f0: frequency list
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    
    power = A **2
    power_flat = power.flatten()
      
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    r = np.sqrt((xx - centerx)** 2 + (yy - centery)**2).flatten()
    

    max_r = min(centerx, centery)
    max_f0 = int(max_r / 2)
    f0 = np.arange(1, max_f0, 1)
    S_f0 = np.zeros_like(f0, dtype=np.float64)
    
    
    #for i, f0_val in enumerate(tqdm(f0, desc="计算S(f0)")):
    for i, f0_val in enumerate(f0):
        index = (r >= f0_val) & (r < 2 * f0_val)
        S_f0[i] = power_flat[index].sum()
        
    return S_f0, f0
    
def main():
    """arr = np.array([1 + 2j, 3 + 4j])
    print(np.abs(arr))"""
    plt.figure(1)
    # q1
    #print("Start fig 1")
    for i in range(4):
        
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        #print("Finish amplitude")
        f, A_f = cart2porl(A,img_list[i])
        plt.plot(np.log(f[1:190]),np.log(A_f[1:190]), color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("1/f law")
    plt.savefig("./pro2_result/f1_law.jpg", bbox_inches='tight', pad_inches=0.0)

    # q2
    #print("Start fig 2")
    plt.figure(2)
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        #print(  "Finish amplitude")
        S_f0, f0 = get_S_f0(A,img_list[i])
        #print("Finish S")
        plt.plot(f0[10:],S_f0[10:], color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("S(f0)")
    plt.savefig("./pro2_result/S_f0.jpg", bbox_inches='tight', pad_inches=0.0)
if __name__ == '__main__':
    main()
