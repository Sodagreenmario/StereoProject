#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
import os
from zhang_utils import homography, intrinsics, extrinsics, refinement, distortion, util
from scipy.optimize import curve_fit

class Calibration(object):
    def __init__(self):
        super(Calibration, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--InputDir', type=str, default='./left', help='the path of the input dirent')
        self.parser.add_argument('--image_file', type=str, default=None, help='the path of the undistorted image')
        self.parser.add_argument('--Undistort', type=bool, default=False, help='whether undistort the image')
        self.parser.add_argument('--Zhang', type=bool, default=False, help='Add Zhang\'s medthod as a comparison')

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*7, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.calibrated_results = None

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('######################### Load Options ########################')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

    def calibrate_single(self, img):
        # Input parameter:
        #   img: np.array

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)

            corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)

    def getPoints(self, inputdir):
        # Input parameters:
        #   inputdir: str of path
        for fimg in os.listdir(inputdir):
            img = cv2.imread(os.path.join(inputdir, fimg), 0)
            self.calibrate_single(img)

    def calibrate_dozens(self, img_shape=(480, 640)):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_shape[::-1],None,None)
            
        self.calibrated_results = dict(camera_matrix=mtx, distortion_coeff=dist, 
                                   rotation_vector=rvecs, translation_vector=tvecs)

        print('-------------------- Calibration Succeeded! --------------------')
        print('-- Camera Matrix: \n', mtx)
        print('-- Distortion Coefficients: \n', dist)
        print('----------------------------------------------------------------')
        print('################################################################')

    def undistort(self, imgfile):
        if imgfile is None:
            return
        img = cv2.imread(imgfile)
        h, w = img.shape[:2]
        mtx, dist = self.calibrated_results['camera_matrix'], self.calibrated_results['distortion_coeff']
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('./result.jpg', dst)
        print('Output: ' + './result.jpg')

    def calibrate_implemented_zhang_method(self):
        objp = self.objpoints[0][:, :2]
        imgp = []
        # Compute homographies for each image
        homographies = []
        for point in self.imgpoints:
            curp = point.reshape(-1, 2)
            H = homography.calculate_homography(objp, curp)
            H = homography.refine_homography(H, objp, curp)
            imgp.append(curp)
            homographies.append(H)

        # Compute intrinsics
        K = intrinsics.recover_intrinsics(homographies)

        obj_homo_3d = util.to_homogeneous_3d(objp)

        extrinsics_matrices = []
        for h, H in enumerate(homographies):
            E = extrinsics.recover_extrinsics(H, K)
            extrinsics_matrices.append(E)

            # projection matrix
            P = np.dot(K, E)

            predicted = np.dot(obj_homo_3d, P.T)
            predicted = util.to_inhomogeneous(predicted)
            points = self.imgpoints[h]
            nonlinear_sse_decomp = np.sum((predicted - points) ** 2)

        k = distortion.calculate_lens_distortion(objp, imgp, K, extrinsics_matrices)

        K_opt, k_opt, extrinsics_opt = refinement.refine_all_parameters(objp, imgp, K, k, extrinsics_matrices)

        print('-------------------- Calibration Succeeded! --------------------')
        print('-- Camera Matrix: \n', K_opt)
        print('-- Distortion Coefficients: \n', k_opt)
        print('----------------------------------------------------------------')
        print('################################################################')



if __name__ == '__main__':
    calibration = Calibration()
    opts = calibration.parse()
    inputdir = opts.InputDir
    img = opts.image_file
    undistort = opts.Undistort
    zhang = opts.Zhang

    calibration.getPoints(inputdir)
    calibration.calibrate_dozens()
    if undistort:
        calibration.calibrate_implemented_zhang_method()
    if zhang:
        calibration.undistort(img)

'''
Reference
[1] OpenCV API Document (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
[2] OpenCV Python Tutorial (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
'''