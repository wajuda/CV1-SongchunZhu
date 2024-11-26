''' 
This file is part of the code for Part 1:
    It contains a function get_filters(), which generates a set of filters in the
format of matrices. (Hint: You add more filters, like the Dirac delta function, whose response is the intensity of the pixel itself.)
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def get_filters():
    '''
    define set of filters which are in the form of matrices
    Return
          F: a list of filters

    '''

    # nabla_x, and nabla_y
    F = [np.array([-1, 1]).reshape((1, 2)), np.array([-1, 1]).reshape((2, 1))]
    #F = [np.array([-1,1]).reshape((1, 2))]
    # gabor filter
    F += [gabor for scale in [1] for theta in range(0, 150, 30)  for gabor in gaborFilter_cv2(scale, theta)] #150
    #F +=[gaborFilter(3,0)]
    # TODO
    # Dirac delta function
    F +=[np.array([1]).reshape((1, 1))]
    #F +=[np.ones((3, 3))/9] #均值滤波器 low pass
    # low_pass
    # high_pass?
    #laplace
    F +=[np.array([[ 0, -1, 0],
                [ -1,  4, -1],
               [ 0, -1, 0]])]


    return F


def gaborFilter(scale, orientation):
    """
      [Cosine, Sine] = gaborfilter(scale, orientation)

      Defintion of "scale": the sigma of short-gaussian-kernel used in gabor.
      Each pixel corresponds to one unit of length.
      The size of the filter is a square of size n by n.
      where n is an odd number that is larger than scale * 6 * 2.
    """
    # TODO (gabor filter is quite useful, you can try to use it)
    orientation = orientation / 180 * math.pi
    size = scale * 6 * 2 + 1
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    X, Y = np.meshgrid(x, y)
    X_prime = X * np.cos(orientation) + Y * np.sin(orientation)
    Y_prime = -X * np.sin(orientation) + Y * np.cos(orientation)

    f = 1/(2*np.pi*scale)
    gaussian = np.exp(-(X_prime ** 2 + Y_prime ** 2)/(2*scale**2))
    sinusoidal_sin = np.sin(2*np.pi*f*X_prime)
    sinusoidal_cos = np.cos(2*np.pi*f*X_prime)

    Cosine = gaussian * sinusoidal_cos
    Sine = gaussian * sinusoidal_sin

    return [Cosine, Sine]

def gaborFilter_cv2(scale, orientation):
    orientation = orientation / 180 * math.pi
    size = scale * 6 * 2 + 1
    cosine = cv2.getGaborKernel((size, size), scale,orientation, 10, 0.5, np.pi/2, ktype=cv2.CV_64F)
    sine = cv2.getGaborKernel((size, size), scale,orientation, 10, 0.5, 0, ktype=cv2.CV_64F)
    return cosine, sine


def plt_filter(filter, round, index,img_name):
    """
    生成并绘制Gabor滤波器图像。


    """
    plt.figure(figsize=(6, 6))
    plt.imshow(filter, cmap='gray')
    plt.title(f'Add filter{index} at round {round}')
    plt.colorbar()
    plt.axis('off')

    plt.savefig(f"results/{img_name.split('.')[0]}/filter_{round}.jpg")
if __name__ == '__main__':
    #print([(scale, ori) for scale in [3,5] for ori in range(0, 150, 30)])
    F = get_filters()
    print(F[2])
    print(F[2].dtype)
    #plt_filter(F[4], 221, 5, 'fur_obs')
    #plt_filter(F[5], 222, 5, 'fur_obs')
    #plt_filter(F[6], 223, 5, 'fur_obs')




