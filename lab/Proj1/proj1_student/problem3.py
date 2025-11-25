'''
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    y = 1.0 / np.sqrt(1 - x)
    
    return np.maximum(y, r_min)
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')
    for i in range(N):
        #
        if length[i] < r_min:
            continue
            
        x1, y1 = points[i]
        # final point
        x2 = x1 + length[i] * np.cos(rad[i])
        y2 = y1 + length[i] * np.sin(rad[i])
        
        #int
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        
        # 
        x1 = max(0, min(x1, pixel-1))
        y1 = max(0, min(y1, pixel-1))
        x2 = max(0, min(x2, pixel-1))
        y2 = max(0, min(y2, pixel-1))
        
        
        cv2.line(bg, (x1, y1), (x2, y2), 0, 1)
    # TODO: Add your code here

    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''

    length = GenLength(N)

    points = np.random.rand(N, 2) * pixel
    # TODO: Add your code here

    rad = np.random.rand(N) * 2 * np.pi
    # TODO: Add your code here

    image = DrawLine(points,rad,length,pixel,N)
    return image,points,rad,length

def DownSampling(img,points,rad,length,pixel,N,rate):
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''
    image = img # Need to be changed    
    # TODO: Add your code here
    new_pixel = pixel // rate
    new_points = points / rate
    
    new_length = length / rate
    
    # 
    image = DrawLine(new_points, rad, new_length, new_pixel, N)
    return image

def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    
    # TODO: Add your code here
    crop_size = 128
    num_crops = 2  
    
 
    fig, axes = plt.subplots(3, num_crops, figsize=(10, 15))
    fig.suptitle('Cropped Patches from Different Scales', fontsize=16)
    
    # I1 (1024x1024)
    for i in range(num_crops):
        max_coord = image1.shape[0] - crop_size
        x = np.random.randint(0, max_coord)
        y = np.random.randint(0, max_coord)
        crop_img = image1[y:y+crop_size, x:x+crop_size]
        axes[0, i].imshow(crop_img, cmap='gray')
        axes[0, i].set_title(f'I1 Crop {i+1}')
        axes[0, i].axis('off')
        cv2.imwrite(f'./pro3_result/crop/crop_i1_{i}.png', crop_img)
    
    # I2 (512x512)
    for i in range(num_crops):
        max_coord = image2.shape[0] - crop_size
        x = np.random.randint(0, max_coord)
        y = np.random.randint(0, max_coord)
        crop_img = image2[y:y+crop_size, x:x+crop_size]
        axes[1, i].imshow(crop_img, cmap='gray')
        axes[1, i].set_title(f'I2 Crop {i+1}')
        axes[1, i].axis('off')
        cv2.imwrite(f'./pro3_result/crop/crop_i2_{i}.png', crop_img)
    
    # I3 (256x256)
    for i in range(num_crops):
        max_coord = image3.shape[0] - crop_size
        x = np.random.randint(0, max_coord)
        y = np.random.randint(0, max_coord)
        crop_img = image3[y:y+crop_size, x:x+crop_size]
        axes[2, i].imshow(crop_img, cmap='gray')
        axes[2, i].set_title(f'I3 Crop {i+1}')
        axes[2, i].axis('off')
        cv2.imwrite(f'./pro3_result/crop/crop_i3_{i}.png', crop_img)
    
    plt.tight_layout()
    plt.savefig('./pro3_result/crop/all_crops.png')
    #plt.show()
    return


def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
if __name__ == '__main__':
    main()
