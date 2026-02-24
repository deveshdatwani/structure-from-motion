import numpy as np
import cv2
import os


def extract_keypoints(DATA_BASE_PATH, IMG_NAMES):
    imgs = [cv2.imread(os.path.join(DATA_BASE_PATH, i), 0) for i in IMG_NAMES]
    img1 = imgs[0]
    img2 = imgs[1]
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.45*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good.append(m)
    good = sorted(good, key = lambda x:x.distance)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return good, pts1, pts2