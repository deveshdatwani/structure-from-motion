import os
import cv2 
import numpy as np
from config import *
from ExtractImages import *
from matplotlib import pyplot as plt


def estimate_essential_matrix(K, pts1, pts2):
    F = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=1.0, confidence=0.999, maxIters=20000)[0]
    print("Fundamental Matrix:\n", F)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=20000)
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    error = np.diag(np.dot(pts1, np.dot(E, pts2.T))).ravel()
    mask = np.abs(error) < 500000 
    return E, mask