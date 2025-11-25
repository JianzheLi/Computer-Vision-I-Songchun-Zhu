'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt


def plot_gibbs_steps(img_dir, prefix='gibbs_', ext='.bmp', steps=25, title='Gibbs Sampling Results'):
    #rows, cols = 5, 5
    if steps == 50:
        rows = 10
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(18, 32))  # 减小高度
        
    else:
        rows = 5
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(18, 16))  # 减小高度

    for i in range(steps):
        img_path = os.path.join(img_dir, f"{prefix}{i}{ext}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.set_title(f"Sweep {i+1}", fontsize=14)  # 
        ax.axis('off')
    # 
    for i in range(steps, rows * cols):
        axes[i // cols, i % cols].axis('off')
    plt.suptitle(title, fontsize=20)  #
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.96)  # 
    plt.savefig(os.path.join(img_dir, 'gibbs_sampling_steps.png'))
    print(f"figures have saved in {img_dir}")
    plt.show()





def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale
        4. beta: annealing temperature
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    
    img_height, img_width = img.shape
    i, j = loc

    
    energy_list = np.zeros((256,1))  


    original_pixel = img[i, j]

    # 周期性边界
    up    = (i-1)%img_height
    down  = (i+1)%img_height
    left  = (j-1)%img_width
    right = (j+1)%img_width

    for val in range(256):
        img[i, j] = val

        # 这里只计算4个方向与该点相关的梯度项
        gx = img[i, right] - img[i, j]
        
        gy = img[down, j] - img[i, j]
        
        gx_inv = img[i, j] - img[i, left]

        gy_inv = img[i, j] - img[up, j]

        energy_here = cal_pot(gx, norm) + cal_pot(gy, norm) + cal_pot(gx_inv, norm) + cal_pot(gy_inv, norm)
        energy_list[val] = energy_here

    
    img[i, j] = original_pixel

    # normalize the energy
    energy_list = energy_list - energy_list.min()
    

    probs = np.exp(-energy_list * beta)
    probs = probs / probs.sum()

    try:
        rand = np.random.rand()
        cumsum = np.cumsum(probs)
        new_val = np.searchsorted(cumsum, rand)
        img[i, j] = new_val
    except:
        raise ValueError(f'probs = {probs}')
    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    filtered_image = image
    # TODO
    filter_h, filter_w = filter.shape
    H, W = image.shape

    if filter_h == 1: # x方向卷积
        k = filter_w
        pad_size = k // 2
    
        padded = np.pad(image, ((0,0), (pad_size, pad_size)), mode='wrap')
        filtered_image = np.zeros_like(image)
        for c in range(k):
            filtered_image += padded[:, c:c+W] * filter[0, c]
    else:            # y方向卷积
        k = filter_h
        pad_size = k // 2
        
        padded = np.pad(image, ((pad_size, pad_size), (0,0)), mode='wrap')
        filtered_image = np.zeros_like(image)
        for r in range(k):
            filtered_image += padded[r:r+H, :] * filter[r, 0]

    return filtered_image


def main():
    # read the distorted image and mask image
    name = "library"
    size = "big"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    # calculate initial energy
    red_channel = distort[:,:,2]
    energy = 0

    #calculate nabla_x
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))



    norm = "L2"
    
    beta_init=0.1
    beta =  beta_init  
    max_beta = 1.0
    img_height, img_width, _ = distort.shape

    sweep = 25
    print(f"./result/{name}/{size}/{norm}")
    for s in tqdm(range(sweep)):
        beta += (max_beta - beta_init) * (s / (sweep - 1))

        for i in range(img_height):
            for j in range(img_width):
                if mask[i,j,2] == 255:
                    #continue
                    distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
        # TODO
        if s == 0: 
            errors = []
        mask_region = (mask[:, :, 2] == 255)
        X = distort[:, :, 2]
        O = ori[:, :, 2]
        mse = np.mean((X[mask_region] - O[mask_region]) ** 2)
        errors.append(mse)
        
        if s == sweep - 1:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(errors, marker='o')
            plt.xlabel('Sweep')
            plt.ylabel('Per-pixel Error (MSE, masked region)')
            plt.title(f'Gibbs Sampler Error Curve: {name} {size}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_path}/gibbs_sampler_error_curve.png")
            plt.close()
            np.save(f"{save_path}/gibbs_sampler_errors.npy", np.array(errors))

        save_path = f"./result/{name}/{size}/{norm}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
    #img_dir = './result/library/big/L2'  # 修改为你的图片存放目录
    plot_gibbs_steps(save_path,steps=sweep)



if __name__ == "__main__":
    #name = "gate"
    #size = "small"
    #norm="L1"
    #save_path = f"./result/{name}/{size}/{norm}"
    #plot_gibbs_steps(save_path, steps=25)
    main()

    # room + big +l2    
    # room + big +l1    
    # room + small +l2    
    # room + small +L1
    # gate big l1
    # gate big l2
    # gate small l2
    # gate small l1 
    # ibrary small l1
    # library big l1
    # library small l2
    #library big l2

    

