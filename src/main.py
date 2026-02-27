import os
import cv2 
import numpy as np
from config import *
from numpy.linalg import svd
from matplotlib import pyplot as plt
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from SavePLY import *
from NonLinearTriangulation import *


if __name__ == "__main__":
    good, pts1, pts2 = extract_keypoints(DATA_BASE_PATH, IMG_NAMES)
    E, mask = estimate_essential_matrix(K, pts1, pts2)
    print(f"Essential matrix: \n {E}")
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    R, t = extract_camera_pose(E, pts1, pts2, K)
    print(f"Rotation matrix: \n {R}")
    print(f"Translation vector: \n {t}")
    points3d = linear_triangulation(pts1, pts2, K, R, t)
    points3d_non_linear = []
    for i in range(len(pts1)):      
        p3d = non_linear_triangulation(pts1[i], pts2[i], K, R, t)
        points3d_non_linear.append(p3d)
    plt.scatter(points3d[0], points3d[2], s=10, c='b', label='Linear Triangulation')
    plt.scatter([p[0] for p in points3d_non_linear], [p[2] for p in points3d_non_linear], s=3, c='r', label='Non-Linear Triangulation')
    plt.legend()
    plt.show()