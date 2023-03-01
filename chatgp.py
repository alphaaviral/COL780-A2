import numpy as np
from scipy.ndimage import *
import cv2
import time
def hessian_corner_detector(image, sigma=1):
    """
    Detects corners in an image using the Hessian corner detector algorithm.

    Args:
        image (ndarray): The input image as a numpy array.
        sigma (float): The standard deviation of the Gaussian filter applied to the image.

    Returns:
        corners (ndarray): A binary image with the same shape as the input image, where
                           the detected corners are marked as 1 and all other pixels are 0.
    """
    # Calculate the second-order derivatives of the image using the Hessian matrix.
    Ixx = convolve(image, np.array([[1, -2, 1]]), mode='constant')
    Iyy = convolve(image, np.array([[1], [-2], [1]]), mode='constant')
    Ixy = convolve(image, np.array([[-1, 1], [1, -1]]), mode='constant')

    # Apply Gaussian smoothing to the second-order derivatives.
    gxx = gaussian_filter(Ixx, sigma)
    gyy = gaussian_filter(Iyy, sigma)
    gxy = gaussian_filter(Ixy, sigma)

    # Calculate the determinant and trace of the Hessian matrix at each pixel.
    det = gxx * gyy - gxy**2
    trace = gxx + gyy

    # Calculate the corner response function R = det - k * trace^2, where k is an empirical constant.
    k = 0.04
    R = det - k * trace**2

    # Threshold the corner response function to obtain a binary image of detected corners.
    threshold = 0.99999 * np.max(R)
    corners = np.zeros_like(image)
    corners[R > threshold] = 1

    return corners

img = cv2.imread('Dataset\\CV_assignment_2_dataset\\1\\1.jpg',cv2.IMREAD_GRAYSCALE)
start = time.time()
corners = hessian_corner_detector(img)
print(time.time()-start)
color_img = cv2.imread('Dataset\\CV_assignment_2_dataset\\1\\1.jpg',cv2.IMREAD_COLOR)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if corners[i][j]==1:
            color_img = cv2.circle(color_img, (j,i), 15, (0,0,255), -1)
cv2.imwrite('./color_img.png', color_img)