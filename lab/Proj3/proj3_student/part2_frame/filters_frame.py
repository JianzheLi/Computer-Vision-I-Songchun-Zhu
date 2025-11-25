#filteres.py

''' 
This file is part of the code for Part 1:
    It contains a function get_filters(), which generates a set of filters in the
format of matrices. (Hint: You add more filters, like the Dirac delta function, whose response is the intensity of the pixel itself.)
'''
import numpy as np
import math

def gaussian_filter(size, sigma):
    
    assert size % 2 == 1, "Size must be odd."
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    print("gaussion kernel:", kernel)
    return kernel


#
# Laplacian of Gaussian 
def laplacian_of_gaussian(size, sigma):
    """Return LoG filter."""
    assert size % 2 == 1
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    
    LoG = (r2 - 2*sigma**2) * np.exp(-r2 / (2*sigma**2))
    # 归一化但保持零和特性
    LoG = LoG - np.mean(LoG)
    LoG = LoG / (np.sqrt(np.sum(LoG**2)) + 1e-10)
    #print(f"LoG sum (should be ~0): {np.sum(LoG):.6f}")
    #print(f"LoG: {LoG}")
    return LoG



def get_filters():
    '''
    define set of filters which are in the form of matrices
    Return
          F: a list of filters

    '''
    """
    # nabla_x, and nabla_y
    F = [np.array([-1, 1]).reshape((1, 2)), np.array([-1, 1]).reshape((2, 1))]
    # gabor filter
    F += [gabor for size in [3,5] for theta in range(0, 150, 30)  for gabor in gaborFilter(size, theta)]

    """
    
    F = []
    # Dirac delta filter (3x3)
    dirac = np.zeros((3, 3))
    dirac[1, 1] = 1.0
    F.append(dirac)

   

    # 4. Gaussian filters (3x3, 5x5)

    for size in [3, 5]:
        F.append(gaussian_filter(size, sigma=size/6))


    # 5. Laplacian of Gaussian filters
    for size in [3, 5]:
        F.append(laplacian_of_gaussian(size, sigma=size/6))

    # 6. Gabor filters
    for size in [3,5]:
        for theta in range(0, 180, 30):
            cos_gabor, sin_gabor = gaborFilter(size, theta)
            F.append(cos_gabor)
            F.append(sin_gabor)
   
    return F




def gaborFilter(size, orientation):
    """
      [Cosine, Sine] = gaborfilter(scale, orientation)

      Defintion of "scale": the sigma of short-gaussian-kernel used in gabor.
      Each pixel corresponds to one unit of length.
      The size of the filter is a square of size n by n.
      where n is an odd number that is larger than scale * 6 * 2.
    """
    # TODO (gabor filter is quite useful, you can try to use it)
    assert size % 2 == 1
    theta = np.deg2rad(orientation)

    sigma = size / 3
    wavelength = size / 1.5

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

   
    gaussian_envelope = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))

    Cosine = gaussian_envelope * np.cos(2 * np.pi * x_theta / wavelength)
    Sine   = gaussian_envelope * np.sin(2 * np.pi * x_theta / wavelength)

    Cosine = Cosine / np.sqrt(np.sum(Cosine**2))
    Sine   = Sine / np.sqrt(np.sum(Sine**2))

    #print("gabor cosine:",Cosine.sum())
    #print("gabor sine:",Sine.sum())
    return Cosine, Sine

if __name__ == '__main__':
    get_filters()



