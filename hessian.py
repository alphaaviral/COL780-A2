import cv2
import numpy as np
import time
from scipy.ndimage import maximum_filter as maxf2D
from scipy.ndimage import convolve as cnv

def calculateDerivatives(image, xaxis, yaxis): # calculate derivatives of image along given axes
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

def calculateHessian(image, patch):  #Determine whether each pixel is a hessian point or not. Return a matrix containing 1s and 0s
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

def getSSD(image1, image2, pos1, pos2, patch):  #Calculate sum of squared differences for two points in consecutive frames
    x1min = pos1[0]-int(patch/2)
    y1min = pos1[1]-int(patch/2)
    x2min = pos2[0]-int(patch/2)
    y2min = pos2[1]-int(patch/2)
    
    ssd=0
    for i in range(patch):
        for j in range(patch):
            ssd += (image1[x1min+i][y1min+j]-image2[x2min+i][y2min+j])**2
    return ssd

def mapHessianPoints(image1, image2, hess1, hess2, patch): #Map hessian point in a frame to those in the next frame by minimising SSD.
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

                mapping[pointCount] = [minSSD, i, j, minPos[0], minPos[1]]
                pointCount += 1

    sortedMapping = mapping[mapping[:, 0].argsort()]
    return sortedMapping

def checkCollinearity(points): #Check if the points are collinear
    mat = np.array([[points[0][0], points[1][0], points[2][0]], [points[0][1], points[1][1], points[2][1]], [1, 1, 1]])
    det = np.linalg.det(mat)
    if det == 0:
        return False
    mat = np.array([[points[0][2], points[1][2], points[2][2]], [points[0][3], points[1][3], points[2][3]], [1, 1, 1]])
    det = np.linalg.det(mat)
    if det == 0:
        return False
    return True

def getAffinePoints(map): #Get 3 non collinear points with min SSD from the mapping of points.
    affinePoints = np.zeros((3,4), dtype=np.float32)
    index = 0
    for i in range(3):
        affinePoints[i] = map[i,1:5]
        index += 1

    while(checkCollinearity(affinePoints) == False):
        if(np.array_equal(affinePoints[0, 2:4], affinePoints[1, 2:4])):
            affinePoints[1] = map[index,1:5]
        else:
            affinePoints[2] = map[index,1:5]
        index += 1

    return affinePoints

def getAffineMatrix(src_points, dst_points): #Get affine parameters from 3 points in consecutive frames.
    src_points[:, 0], src_points[:, 1] = src_points[:, 1], src_points[:, 0].copy()
    dst_points[:, 0], dst_points[:, 1] = dst_points[:, 1], dst_points[:, 0].copy()
    affMatrix = np.zeros((2,3))
    Y = np.array([[src_points[0][0]], [src_points[1][0]], [src_points[2][0]]])
    A = np.array([[dst_points[0][0], dst_points[0][1], 1], [dst_points[1][0], dst_points[1][1], 1], [dst_points[2][0], dst_points[2][1], 1]])
    affMatrix[0] = np.transpose(np.linalg.solve(A, Y))
    Y = np.array([[src_points[0][1]], [src_points[1][1]], [src_points[2][1]]])
    affMatrix[1] = np.transpose(np.linalg.solve(A, Y))
    return affMatrix

def getCumulativeAffine(cumAffine, newAffine): #Get cumulative affine matrix to find the transformation wrt first frame.
    cumAffine = np.append(cumAffine, [[0, 0, 1]], axis=0)
    newAffine = np.append(newAffine, [[0, 0, 1]], axis=0)
    finalAffine = np.dot(newAffine, cumAffine)
    finalAffine = finalAffine[0:2]
    return finalAffine

def trimImage(image): #Trim the stitched image to remove black borders.
    temp = np.argwhere(image != 0)
    max_y = temp[:, 0].max()
    min_y = temp[:, 0].min()
    min_x = temp[:, 1].min()
    max_x = temp[:, 1].max()
    ret = image[min_y:max_y, min_x:max_x]
    return ret

if __name__ == '__main__':
    patchSize = 7                               #Patch size for calculating SSD, for summation in hessian matrix
    imgpath = 'Dataset\\New Dataset\\1\\image '
    imageNo = -1
    oldHessian = None
    oldImg = None
    stitched = None
    oldTransform = None

    while(1):
        imageNo += 1
        path = imgpath + str(imageNo) + '.jpg'
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hessian = calculateHessian(img_gray,patchSize)    #Calculate hessian points for the image

        if imageNo == 0:
            oldHessian = hessian
            oldImg = img_gray
            stitched = img
            continue

        map = mapHessianPoints(oldImg,img_gray,oldHessian,hessian,patchSize) #Find map of hessian points using the hessian of previous image and current image
        aff = getAffinePoints(map)                                           #Get affine points from the map
        affineTransform = getAffineMatrix(aff[:,0:2], aff[:,2:4])            #Find the transformation from this image to previous image

        if oldTransform is not None:
            affineTransform = getCumulativeAffine(oldTransform, affineTransform) #Find the transformation from this image to first image
        
        rows, cols, ch = img.shape
        affineTransform[0][2] += int(cols/2)
        affineTransform[1][2] += int(rows/2)
        transformed_img = cv2.warpAffine(np.float64(img), affineTransform, (int(2*cols),int(2*rows))) #Transform the image to align with the first image
        affineTransform[0][2] -= int(cols/2)
        affineTransform[1][2] -= int(rows/2)

        stitchedTransform = np.array([[1,0,int(cols/2)],[0,1,int(rows/2)]]).astype('float32')
        stitched = cv2.warpAffine(np.float64(stitched), stitchedTransform, (int(2*cols),int(2*rows)))
        stitched = np.where(stitched!=0, stitched, transformed_img)                                   #Add the current image to the already stitched image
        stitched = trimImage(stitched)                                                                #Trim the image to remove black borders

        cv2.imwrite('./stitched.png', stitched)
        oldTransform = affineTransform
        oldHessian = hessian
        oldImg = img_gray