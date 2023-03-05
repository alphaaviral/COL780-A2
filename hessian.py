import cv2
import numpy as np
import multiprocessing
import time
import csv
import sys
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
    image = image.astype('int64')
    Ix, Iy = calculateDerivatives(image, True, True)
    Ixx, Ixy = calculateDerivatives(Ix, True, True)
    Iyy = calculateDerivatives(Iy, False, True)
    
    Ixx = cnv(Ixx, np.ones((patch,patch)), mode='constant')
    Iyy = cnv(Iyy, np.ones((patch,patch)), mode='constant')
    Ixy = cnv(Ixy, np.ones((patch,patch)), mode='constant')

    determinant = Ixx*Iyy - Ixy**2
    maximas = maxf2D(determinant, size=(27,27),mode='constant', cval=0.0)

    difference = determinant-maximas
    isMaxima = np.zeros_like(image)
    isMaxima[difference==0] = 1

    res = np.zeros_like(image)
    threshold = 0.35*np.max(maximas)
    res[maximas>threshold] = 1

    res = res*isMaxima
    res[0:10,:] = 0
    res[:,0:10] = 0
    res[image.shape[0]-10:image.shape[0],:] = 0
    res[:,image.shape[1]-10:image.shape[1]] = 0
    return res

def getSSD(image1, image2, pos1, pos2, patch):
    x1min = pos1[0]-int(patch/2)
    y1min = pos1[1]-int(patch/2)
    x2min = pos2[0]-int(patch/2)
    y2min = pos2[1]-int(patch/2)
    
    ssd=0
    for i in range(patch):
        for j in range(patch):
            # for color in range(3):
                ssd += (image1[x1min+i][y1min+j]-image2[x2min+i][y2min+j])**2
    return ssd

def mapHessianPoints(image1, image2, hess1, hess2, patch):
    image1 = image1.astype('int64')
    image2 = image2.astype('int64')
    padded_image1 = np.pad(image1, int(patch/2), 'constant', constant_values=0)
    padded_image2 = np.pad(image2, int(patch/2), 'constant', constant_values=0)
    cnt = np.bincount(np.reshape(hess1, hess1.size))
    mapping = np.zeros((cnt[1],5), dtype=np.int64)
    
    pointCount = 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if hess1[i][j] == 1:
                xmin = i-int(image1.shape[0]/8)
                xmax = i+int(image1.shape[0]/8)+1
                ymin = j-int(image1.shape[1]/8)
                ymax = j+int(image1.shape[1]/8)+1
                if xmin<0:
                    xmin = 0
                if ymin<0:
                    ymin = 0
                if xmax>=image2.shape[0]:
                    xmax = image2.shape[0]
                if ymax>=image2.shape[1]:  
                    ymax = image2.shape[1]
                
                minSSD = 9223372036854775806
                minPos = (0,0)
                for m in range(xmin,xmax):
                    for n in range(ymin,ymax):
                        if hess2[m][n] == 1:
                            ssd = getSSD(padded_image1,padded_image2,(i,j),(m,n),patch)
                            if ssd<minSSD:
                                minSSD = ssd
                                minPos = (m,n)

                # if minSSD == float('inf'):
                #     minSSD = 
                mapping[pointCount] = [minSSD, i, j, minPos[0], minPos[1]]
                pointCount += 1

    sortedMapping = mapping[mapping[:, 0].argsort()]
    return sortedMapping

def checkCollinearity(points):
    mat = np.array([[points[0][0], points[1][0], points[2][0]], [points[0][1], points[1][1], points[2][1]], [1, 1, 1]])
    det = np.linalg.det(mat)
    if det == 0:
        return False
    mat = np.array([[points[0][2], points[1][2], points[2][2]], [points[0][3], points[1][3], points[2][3]], [1, 1, 1]])
    det = np.linalg.det(mat)
    if det == 0:
        return False
    return True

def getAffinePoints(map):
    affinePoints = np.zeros((3,4), dtype=np.float32)
    index = 0
    for i in range(3):
        affinePoints[i] = map[i,1:5]
        index += 1

    while(checkCollinearity(affinePoints) == False):
        affinePoints[2] = map[index,1:5]
        index += 1

    return affinePoints

def getAffineMatrix(src_points, dst_points):
    src_points[:, 0], src_points[:, 1] = src_points[:, 1], src_points[:, 0].copy()
    dst_points[:, 0], dst_points[:, 1] = dst_points[:, 1], dst_points[:, 0].copy()
    affMatrix = np.zeros((2,3))
    Y = np.array([[src_points[0][0]], [src_points[1][0]], [src_points[2][0]]])
    A = np.array([[dst_points[0][0], dst_points[0][1], 1], [dst_points[1][0], dst_points[1][1], 1], [dst_points[2][0], dst_points[2][1], 1]])
    affMatrix[0] = np.transpose(np.linalg.solve(A, Y))
    Y = np.array([[src_points[0][1]], [src_points[1][1]], [src_points[2][1]]])
    affMatrix[1] = np.transpose(np.linalg.solve(A, Y))
    return affMatrix

def getCumulativeAffine(cumAffine, newAffine):
    cumAffine = np.append(cumAffine, [[0, 0, 1]], axis=0)
    newAffine = np.append(newAffine, [[0, 0, 1]], axis=0)
    finalAffine = np.dot(newAffine, cumAffine)
    finalAffine = finalAffine[0:2]
    return finalAffine

if __name__ == '__main__':
    start_time = time.time()
    patchSize = 7
    imgpath = 'Dataset\\New Dataset\\2\\image '
    imageNo = -1
    oldHessian = None
    oldImg = None
    stitched = None
    oldTransform = None
    finalSize = (0,0)
    while(1):
        imageNo += 1
        path = imgpath + str(imageNo) + '.jpg'
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hessian = calculateHessian(img_gray,patchSize)
        if imageNo == 0:
            finalSize = (img.shape[0], img.shape[1])
            oldHessian = hessian
            oldImg = img_gray
            stitched = img
            # stitched = np.pad(stitched, ((1000, 1000), (1000, 1000), (0,0)), 'constant', constant_values=0)
            # stitchAffine = np.array([[1,0,int(cols/2)],[0,1,int(rows/2)]]).astype('float32')
            # transformed_stitched = cv2.warpAffine(stitched, stitchAffine, (int(2*cols),int(2*rows)))
            # stitched = transformed_stitched
            # cv2.imshow('Stitched Image', stitched)
            # cv2.waitKey(10000)
            continue
        map = mapHessianPoints(oldImg,img_gray,oldHessian,hessian,patchSize)
        aff = getAffinePoints(map)
        affineTransform = getAffineMatrix(aff[:,0:2], aff[:,2:4])
        if oldTransform is not None:
            affineTransform = getCumulativeAffine(oldTransform, affineTransform)
        
        # img = np.pad(img, ((1000, 1000), (1000, 1000), (0,0)), 'constant', constant_values=0)
        rows, cols, ch = img.shape
        affineTransform[0][2] += int(cols/2)
        affineTransform[1][2] += int(rows/2)
        transformed_img = cv2.warpAffine(np.float64(img), affineTransform, (int(2*cols),int(2*rows)))
        affineTransform[0][2] -= int(cols/2)
        affineTransform[1][2] -= int(rows/2)
        # rows, cols, ch = stitched.shape
        # stitched_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        stitchedTransform = np.array([[1,0,int(cols/2)],[0,1,int(rows/2)]]).astype('float32')
        stitched = cv2.warpAffine(np.float64(stitched), stitchedTransform, (int(2*cols),int(2*rows)))
        stitched = np.where(stitched!=0, stitched, transformed_img)
        dummy = np.argwhere(stitched != 0) # assume blackground is zero
        max_y = dummy[:, 0].max()
        min_y = dummy[:, 0].min()
        min_x = dummy[:, 1].min()
        max_x = dummy[:, 1].max()
        stitched = stitched[min_y:max_y, min_x:max_x]

        cv2.imwrite('./stitched.png', stitched)
        oldTransform = affineTransform
        oldHessian = hessian
        oldImg = img_gray

    # img1 = cv2.imread(imgpath,cv2.IMREAD_COLOR)
    # img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # hessian1 = calculateHessian(img1_gray,patchSize)
    # imgpath = 'Dataset\\CV_assignment_2_dataset\\2\\2.jpg'
    # img2 = cv2.imread(imgpath,cv2.IMREAD_COLOR)
    # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = img2.astype('int64')
    # hessian2 = calculateHessian(img2_gray,patchSize)
    # map = mapHessianPoints(img1_gray,img2_gray,hessian1,hessian2,patchSize)
    # aff = getAffinePoints(map)
    # affineTransform = getAffineMatrix(aff[:,0:2], aff[:,2:4])
    # rows, cols, ch = img2.shape
    # affineTransform[0][2] += int(cols/2)
    # affineTransform[1][2] += int(rows/2)
    # transformed_img2 = cv2.warpAffine(np.float64(img2), affineTransform, (int(2*cols),int(2*rows)))
    # affineTransform = np.array([[1,0,int(cols/2)],[0,1,int(rows/2)]]).astype('float32')
    # transformed_img1 = cv2.warpAffine(np.float64(img1), affineTransform, (int(2*cols),int(2*rows)))
    # stitched = np.where(transformed_img1!=0, transformed_img1, transformed_img2)

    # print(time.time()-start_time)
    # cv2.imwrite('./affine1.png', transformed_img1)
    # cv2.imwrite('./affine2.png', transformed_img2)
    # cv2.imwrite('./stitched.png', stitched)