''' 
This file is part of the code for Part 1:
    It contains a function get_filters(), which generates a set of filters in the
format of matrices. (Hint: You add more filters, like the Dirac delta function, whose response is the intensity of the pixel itself.)
'''
import numpy as np
import math


def get_filters():
    '''
    define set of filters which are in the form of matrices
    Return
          F: a list of filters

    '''

    # nabla_x, and nabla_y
    F = [np.array([-1, 1]).reshape((1, 2)), np.array([-1, 1]).reshape((2, 1))]
    # gabor filter
    F += [gabor for size in [3,5] for theta in range(0, 150, 30)  for gabor in gaborFilter(size, theta)]

    # TODO

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

    return Cosine, Sine

if __name__ == '__main__':
    get_filters()



