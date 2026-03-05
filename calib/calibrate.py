import cv2
import numpy as np
import os, glob


def remove_distortion(img, camera_matrix, dis_coeff):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dis_coeff, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dis_coeff, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]
    return undistorted_img

def calibrate(show_pics=False):
    root = os.getcwd()
    calibration_dir = os.path.join(root, 'calib', 'calibration_images')
    image_path_list = glob.glob(os.path.join(calibration_dir, '*.jpg'))
    n_rows = 9
    n_cols = 6
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    world_points = np.zeros((n_rows * n_cols, 3), np.float32)
    world_points[:, :2] = np.mgrid[0:n_rows, 0:n_cols].T.reshape(-1, 2)
    world_points_list = []
    image_points_list = []
    for image_path in image_path_list:
        print(f"processing {image_path}")
        img = cv2.imread(image_path)
        print(img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found_corners, img_corners = cv2.findChessboardCorners(img_gray, (n_rows, n_cols), None)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if found_corners:
            world_points_list.append(world_points)
            refined_corners = cv2.cornerSubPix(img_gray, img_corners, (11, 11), (-1, -1), termination_criteria)
            image_points_list.append(refined_corners)
            if show_pics:
                cv2.drawChessboardCorners(img, (n_rows, n_cols), refined_corners, found_corners)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    rep_error, camera_matrix, dis_coeff, rvecs, tvecs = cv2.calibrateCamera(world_points_list, image_points_list, img_gray.shape[::-1], None, None)
    print(f"reprojection error: \n {rep_error}")
    print(f"camera matrix: \n {camera_matrix}")
    print(f"distortion coefficients: {dis_coeff}")
    return camera_matrix, dis_coeff

if __name__ == "__main__":
    camera_matrix, dis_coeff = calibrate(show_pics=True)