#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
import os

class Calibration(object):
    def __init__(self):
        super(Calibration, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--InputDir', type=str, default='./left', help='the path of the input dirent')

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*7, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

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

    def calibrate_dozens(self, inputdir):
        # Input parameters:
        #   inputdir: str of path
        for fimg in os.listdir(inputdir):
            img = cv2.imread(os.path.join(inputdir, fimg), 0)
            self.calibrate_single(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[::-1],None,None)
            
        print('-------------------- Calibration Succeeded! --------------------')
        print('-- Camera Matrix: \n', mtx)
        print('-- Distortion Coefficients: \n', dist)
        print('-- Rotation Vectors: \n', rvecs)
        print('-- Translation Vectors: \n', tvecs)
        print('----------------------------------------------------------------')
        print('################################################################')


if __name__ == '__main__':
    calibration = Calibration()
    opts = calibration.parse()
    inputdir = opts.InputDir

    calibration.calibrate_dozens(inputdir)

'''
Reference
[1] OpenCV API Document (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
[2] OpenCV Python Tutorial (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
'''