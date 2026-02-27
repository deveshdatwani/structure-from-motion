import cv2
import numpy as np


def linear_triangulation(pts1, pts2, K, R, t):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)
    pts_3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d /= pts_3d[3]
    return pts_3d