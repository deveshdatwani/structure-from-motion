import cv2 
from matplotlib import pyplot as plt
from numpy.linalg import svd
import numpy as np
import os

from config import *

imgs = [cv2.imread(os.path.join(DATA_BASE_PATH, i), 0) for i in IMG_NAMES]
K = np.array([531.122155322710, 0, 407.192550839899,
			0, 531.541737503901, 313.308715048366,
			0, 0, 1]).reshape((3,3))
img1 = imgs[0]
img2 = imgs[1]

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT) 
U, S, Vt = np.linalg.svd(F)
S[2] = 0  
F = U @ np.diag(S) @ Vt

E = K.T @ F @ K
U, S, Vt = np.linalg.svd(F)
S = np.diag((1,1,0))
E = U @ S @ Vt  

print(E)