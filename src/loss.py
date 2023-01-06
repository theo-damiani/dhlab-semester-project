###############################################################################
###############################################################################
###############################################################################

import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import variation_of_information as voi
from skimage.metrics import normalized_root_mse as NRMSE
from scipy.spatial import distance

from skimage.feature import hog

from scipy.ndimage import distance_transform_edt
import cv2
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
# Loss used with distance transformed target and inverted patch:
def mask_mean(X, Y):
    mask = Y==0
    return np.mean(X[mask])

def IntersectionOverUnion(X, Y):
    i = np.sum(X&Y)
    u = np.sum(X|Y)
    return 1-(i/u)

def mask_mean_symmetric(X, Y):
    maskY = Y==0
    maskX = X==0
    
    yc = Y[maskX]
    if not np.any(yc):
        return (np.mean(X[maskY])+1000)/2
    else:
        return (np.mean(X[maskY])+np.mean(yc))/2
    
def mask_mean_symmetric_edt(X, Y):
    X_edt = distance_transform_edt(X)
    X_edt = cv2.normalize(X_edt, None, alpha=0, beta=1,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 

    l1 = mask_mean(X_edt, Y)
    
    Y_edt = distance_transform_edt(Y)
    Y_edt = cv2.normalize(Y_edt, None, alpha=0, beta=1,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 

    l2 = mask_mean(Y_edt, X)    
    return l1+l2

def mask_mean_symmetric_edt_threshold(X, Y):
    X_edt = distance_transform_edt(X)
    _ ,X_edt = cv2.threshold(X_edt,10,255,cv2.THRESH_TRUNC)
    
    l1 = mask_mean(X_edt, Y)
    
    Y_edt = distance_transform_edt(Y)
    _ ,Y_edt = cv2.threshold(Y_edt,10,255,cv2.THRESH_BINARY)
    
    l2 = mask_mean(Y_edt, X)    
    return l1+l2
    
def dsim(X, Y):
    return 1-ssim(X, Y, data_range=(Y.max()-Y.min()))

def nmi_neg(X, Y):
    return (1-nmi(X, Y))

def voi_mean(X, Y):
    return np.mean(voi(X, Y))

# Loss used with distance transformed target and patch:
def RMSE(X, Y):
    sub = np.subtract(X,Y, dtype=np.float64)
    return np.sqrt(np.mean(sub**2))

###############################################################################

def mse(X, Y):
    loss = np.mean(np.square(np.subtract(X,Y, dtype=np.float64)))

    return loss

def zncc(X, Y, eps = 1e-5):

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    meanX = np.mean(X)
    meanY = np.mean(Y)
    
    stdX = np.std(X)
    stdY = np.std(Y)
    
    X1 = X-meanX
    Y1 = Y-meanY
    
    num = X1*Y1
    
    dem = stdX*stdY
    
    return -np.mean(num/(dem+eps))

def root_mean_squared_error(act, pred):

    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    rmse_val = np.sqrt(mean_diff)
    return rmse_val

def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    return d

def hog_based(X, Y):
    
    hog_X = hog(X, orientations=9)
    hog_Y = hog(Y, orientations=9)
    
    #l=mse(hog_X, hog_Y)
    #l=distance.euclidean(hog_X, hog_Y)
    l = np.linalg.norm(hog_X-hog_Y)
    #l=root_mean_squared_error(hog_X, hog_Y)
    #l=chi2_distance(hog_X, hog_Y)
    
    return l