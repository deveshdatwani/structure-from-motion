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
    save_ply(points3d.T, "output.ply")
    plt.scatter(points3d[0], points3d[2], color='blue', marker='.')
    plt.scatter(t[0], t[2], color='red', marker='x')
    plt.scatter(0, 0, color='green', marker='o')
    plt.show()