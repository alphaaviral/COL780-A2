import cv2
import numpy as np
import multiprocessing
import time
import csv
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import convolve as cnv

def calculateDerivatives(image, xaxis, yaxis):
    xDer = np.array([[-1, 0, 1]], np.float64)
    yDer = np.array([[1], [0], [-1]], np.float64)
    if xaxis:
        Ix = cnv(image, xDer, mode='constant')
    if yaxis:
        Iy = cnv(image, yDer, mode='constant')
    if xaxis and yaxis:
        return Ix,Iy
    elif xaxis:
        return Ix
    elif yaxis:
        return Iy

def calculateHessian(image, patch):
    Ix, Iy = calculateDerivatives(image, True, True)
    Ixx, Ixy = calculateDerivatives(Ix, True, True)
    Iyy = calculateDerivatives(Iy, False, True)
    
    Ixx = cnv(Ixx, np.ones((patch,patch)), mode='constant')
    Iyy = cnv(Iyy, np.ones((patch,patch)), mode='constant')
    Ixy = cnv(Ixy, np.ones((patch,patch)), mode='constant')
    return (Ixx*Iyy - Ixy**2)

if __name__ == '__main__':
    patchSize = 7
    imgpath = 'Dataset\\CV_assignment_2_dataset\\1\\2.jpg'
    img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    img = img.astype('int64')
    hessianDet = calculateHessian(img,patchSize)
    start_time = time.time()
    maximas = maxf2D(hessianDet, size=(27,27),mode='constant', cval=0.0)
    threshold = 0.2*np.max(maximas)
    color_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if hessianDet[i][j] == maximas[i][j] and hessianDet[i][j]>threshold:
                color_img = cv2.circle(color_img, (j,i), 15, (0,0,255), -1)
    cv2.imwrite('./color_img.png', color_img)