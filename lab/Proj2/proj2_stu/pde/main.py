'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os
import matplotlib.pyplot as plt



def plot_steps(img_dir, prefix='pde_', ext='.bmp', steps=101, title='PDE Results'):
    #rows, cols = 5, 5
    
    cols=5
    rows=4

    fig, axes = plt.subplots(rows, cols, figsize=(18, 14))  # 减小高度

    num=-1
    
    for i in range(steps):
        if i%10 == 0 or i <=10:
            img_path = os.path.join(img_dir, f"{prefix}{i}{ext}")
            num+=1
        else:
            continue
     
        img = cv2.imread(img_path)
       
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes[num  // cols,  num % cols]
        ax.imshow(img)
        ax.set_title(f"Sweep {i}", fontsize=16)  # 
        ax.axis('off')
    # 
    print(num)

    for i in range(100, rows * cols):
        axes[i // cols, i % cols].axis('off')
    plt.suptitle(title, fontsize=18)  #
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.9)  # 
    plt.savefig(os.path.join(img_dir, 'gibbs_sampling_steps.png'))
    plt.show()


def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''

    # TODO
    img_height, img_width = img.shape
    i, j = loc
    up    = (i-1)%img_height
    down  = (i+1)%img_height
    left  = (j-1)%img_width
    right = (j+1)%img_width

    
    lap = (img[up,j] + img[down,j] + img[i,left] + img[i,right]) / 4.0 - img[i,j]
    img[i,j] = img[i,j] + beta * lap

    
    img[i,j] = np.clip(img[i,j], 0, 255)
    return img



def main():
    # read the distorted image and mask image
    name = "room"
    size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 1
    img_height, img_width, _ = distort.shape

    sweep = 101
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                # TODO]
                if mask[i, j, 2] == 255:
                    distort[:,:,2] = pde(distort[:,:,2], [i,j], beta)

        # TODO
        if s == 0:
            errors = []
        mask_region = (mask[:,:,2] == 255)
        X = distort[:,:,2]
        O = ori[:,:,2]
        mse = np.mean((X[mask_region] - O[mask_region])**2)
        errors.append(mse)
        if s == sweep - 1:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(errors, marker='o')
            plt.xlabel('Sweep')
            plt.ylabel('Per-pixel Error (MSE, masked region)')
            plt.title(f'PDE Error Curve: {name} {size}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./result/{name}/{size}/pde_error_curve.png")
            plt.close()
            np.save(f"./result/{name}/{size}/pde_errors.npy", np.array(errors))

        if s % 1 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)
    
    plot_steps(save_path,steps=sweep)


if __name__ == "__main__":
    main()
    #name = "room"
    #size = "small"
    #save_path = f"./result/{name}/{size}"
    #plot_steps(save_path)

    #libary small
    #libary big
    #room small
    







        

