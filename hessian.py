import cv2
import numpy as np
import multiprocessing
import scipy.signal as sc
import time
import csv
from scipy.ndimage import maximum_filter as maxf2D

def calculateIntegral(image):
    width = image.shape[1]
    height = image.shape[0]
    res = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            if j>0 and i>0:
                res[i][j]  = res[i][j-1] + res[i-1][j] - res[i-1][j-1] + image[i][j]
            elif j==0 and i==0:
                res[i][j] = image[i][j]
            elif j==0:
                res[i][j] = res[i-1][j] + image[i][j]
            elif i==0:
                res[i][j] = res[i][j-1] + image[i][j]
    return res

def convolve(image, kernel):
    res = np.zeros(image.shape)
    xoffset = int(kernel.shape[0]/2)
    yoffset = int(kernel.shape[1]/2)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sum=0
            for m in range(0,kernel.shape[0]):
                xindex = i+m-xoffset
                if xindex<0:
                    continue
                for n in range(0,kernel.shape[1]):
                    yindex = j+n-yoffset
                    if yindex<0:
                        continue
                    try:
                        sum += image[xindex][yindex]*kernel[m][n]
                    except IndexError:
                        continue
            res[i][j] = int(sum)
    return res

def calculateDerivatives(image, xaxis, yaxis):
    xDer = np.array([[-1, 0, 1]], np.float64)
    yDer = np.array([[1], [0], [-1]], np.float64)
    if xaxis:
        Ix = sc.convolve2d(image, xDer,mode='same')
    if yaxis:
        Iy = sc.convolve2d(image, yDer,mode='same')
    if xaxis and yaxis:
        return Ix,Iy
    elif xaxis:
        return Ix
    elif yaxis:
        return Iy

def calculateHessian(image, patch):
    print("Calculating derivatives...")
    start_time = time.time()
    Ix, Iy = calculateDerivatives(image, True, True)
    Ixx, Ixy = calculateDerivatives(Ix, True, True)
    Iyy = calculateDerivatives(Iy, False, True)
    print(time.time() - start_time)
    print("Calculating integral...")
    start_time = time.time()
    Ixxint = calculateIntegral(Ixx)
    Ixyint = calculateIntegral(Ixy)
    Iyyint = calculateIntegral(Iyy)
    print(time.time() - start_time)
    hessian = np.zeros((image.shape[0], image.shape[1], 2, 2))
    print("Calculating hessian...")
    start_time = time.time()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            xmax = i+int(patch/2)
            ymax = j+int(patch/2)
            xmin = i-int(patch/2)-1
            ymin = j-int(patch/2)-1
            if xmax>=image.shape[0]:
                xmax = image.shape[0]-1
            if ymax>=image.shape[1]:
                ymax = image.shape[1]-1
            
            sigmaIxx = Ixxint[xmax][ymax]
            sigmaIyy = Iyyint[xmax][ymax]
            sigmaIxy = Ixyint[xmax][ymax]
            if xmin>=0 and ymin>=0:
                sigmaIxx = sigmaIxx - Ixxint[xmax][ymin] - Ixxint[xmin][ymax] + Ixxint[xmin][ymin]
                sigmaIyy = sigmaIyy - Iyyint[xmax][ymin] - Iyyint[xmin][ymax] + Iyyint[xmin][ymin]
                sigmaIxy = sigmaIxy - Ixyint[xmax][ymin] - Ixyint[xmin][ymax] + Ixyint[xmin][ymin]
            elif xmin>=0:
                sigmaIxx = sigmaIxx - Ixxint[xmin][ymax]
                sigmaIyy = sigmaIyy - Iyyint[xmin][ymax]
                sigmaIxy = sigmaIxy - Ixyint[xmin][ymax]
            elif ymin>=0:
                sigmaIxx = sigmaIxx - Ixxint[xmax][ymin]
                sigmaIyy = sigmaIyy - Iyyint[xmax][ymin]
                sigmaIxy = sigmaIxy - Ixyint[xmax][ymin]

            hessian[i][j][0][0] = sigmaIxx
            hessian[i][j][1][1] = sigmaIyy
            hessian[i][j][1][0] = sigmaIxy
            hessian[i][j][0][1] = sigmaIxy
    print(time.time() - start_time)
    return hessian

def diagonalizeHessian(hess,resArray):
    print("Diagonalizing hessian...")
    start_time = time.time()
    # hessianDet = np.zeros((hess.shape[0],hess.shape[1]))
    # hessianDiagonal = np.zeros((img.shape[0], img.shape[1], 2))
    for i in range(hess.shape[0]):
        for j in range(hess.shape[1]):
            P,D,Q = np.linalg.svd(hess[i][j], full_matrices=False, hermitian=True)
            # hessianDiagonal[i][j] = D
            # hessianDet[i][j] = D[0]*D[1]
            resArray[i*hess.shape[1]+j] = D[0]*D[1]
    print(time.time() - start_time)
    # return hessianDet


if __name__ == '__main__':
    patchSize = 7
    img = cv2.imread('Chess_Board.png',cv2.IMREAD_GRAYSCALE)
    img = img.astype('int64')
    # start_time = time.time()
    hessian = calculateHessian(img,patchSize)



    # hessianDet = diagonalizeHessian(hessian)

    retArrays = []
    for i in range(4):
        retArrays.append(multiprocessing.Array('d', int(img.shape[0]*img.shape[1]/4)))

    processes = []
    for i in range(4):
        processes.append(multiprocessing.Process(target = diagonalizeHessian, args=(hessian[int(i*hessian.shape[0]/4):int((i+1)*hessian.shape[0]/4)][:][:][:],retArrays[i])))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    hessianDet = np.zeros(img.shape)
    for i in range(4):
        subsize  = int(img.shape[0]/4)
        for j in range(int(img.shape[0]*img.shape[1]/4)):
            hessianDet[(i*subsize)+(j//img.shape[1])] [j%img.shape[1]]= retArrays[i][j]

    # print(time.time() - start_time)


    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         hessianDet[i][j] = hessian[i][j][0]*hessian[i][j][1]

    # with open("hessianDet.csv","w+") as my_csv:
    #     csvWriter = csv.writer(my_csv,delimiter=',')
    #     csvWriter.writerows(hessianDet)
    print("Finding maxima...")
    start_time = time.time()
    maximas = maxf2D(hessianDet, size=(27,27),mode='constant', cval=0.0)
    print(time.time() - start_time)
    # maximas = np.zeros(img.shape)
    # maximaPatchSize = 27
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         xmax = i+int(maximaPatchSize/2)
    #         ymax = j+int(maximaPatchSize/2)
    #         xmin = i-int(maximaPatchSize/2)
    #         ymin = j-int(maximaPatchSize/2)
    #         if xmax>=img.shape[0]:
    #             xmax = img.shape[0]-1
    #         if ymax>=img.shape[1]:
    #             ymax = img.shape[1]-1
    #         if xmin<0:
    #             xmin = 0
    #         if ymin<0:
    #             ymin = 0
    #         maxValue = 0
    #         for m in range(xmin, xmax+1):
    #             for n in range(ymin, ymax+1):
    #                 if hessianDet[m][n]>maxValue:
    #                     maxValue = hessianDet[m][n]
    #         maximas[i][j] = maxValue

    color_img = cv2.imread("Chess_Board.png", cv2.IMREAD_COLOR)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if hessianDet[i][j] == maximas[i][j] and hessianDet[i][j]>1500000:
                color_img = cv2.circle(color_img, (j,i), 15, (0,0,255), -1)
    cv2.imwrite('./color_img.png', color_img)
    # doubleDerivative = calculateDerivatives(derivative, True, True)
    # cv2.imshow('img',xder)
    # cv2.imshow('iwmg',yder)
    # cv2.waitKey(5000)

