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
        self.parser.add_argument('--InputDirL', type=str, default='./left', help='the path of the input left camera\'s dirent')
        self.parser.add_argument('--InputDirR', type=str, default='./right', help='the path of the input right camera\'s image')

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*9, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        self.objpointsL = [] # 3d point in real world space
        self.objpointsR = []
        self.imgpointsL = [] # 2d points in image plane.
        self.imgpointsR = []  # 2d points in image plane.
        self.calibrated_resultsL = None
        self.calibrated_resultsR = None

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('######################### Load Options ########################')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

    def calibrate_single(self, img, left=None):
        # Input parameter:
        #   img: np.array

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (9,6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), self.criteria)
            if left:
                self.objpointsL.append(self.objp)
                self.imgpointsL.append(corners2)
            else:
                self.objpointsR.append(self.objp)
                self.imgpointsR.append(corners2)

    def getPoints(self, inputdir, left=None):
        # Input parameters:
        #   inputdir: str of path
        for fimg in os.listdir(inputdir):
            img = cv2.imread(os.path.join(inputdir, fimg), 0)
            self.calibrate_single(img, left)

    def calibrate_dozens(self, img_shape=(480, 640)):
        ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(self.objpointsL, self.imgpointsL, img_shape[::-1], None, None)
        ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(self.objpointsR, self.imgpointsR, img_shape[::-1], None, None)
        self.calibrated_resultsL = dict(camera_matrix=mtxL, distortion_coeff=distL)
        self.calibrated_resultsR = dict(camera_matrix=mtxR, distortion_coeff=distR)
        print(len(self.objpointsL), len(self.objpointsR), len(self.imgpointsL), len(self.imgpointsR))
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(self.objpointsR, self.imgpointsL,
                                                        self.imgpointsR, mtxL, distL, mtxR, distR, img_shape[::-1])

        print('-------------------- Calibration Succeeded! --------------------')
        print('-- Left  - Camera Matrix: \n', mtxL)
        print('-- Left  - Distortion Coefficients: \n', distL)
        print('-- Right - Camera Matrix: \n', mtxR)
        print('-- Right - Distortion Coefficients: \n', distR)
        print('-- Transformation from left to right: \n')
        print('-- R = ', R)
        print('-- T = ', T)
        print('-- E = ', E)
        print('-- F = ', F)
        print('----------------------------------------------------------------')

if __name__ == '__main__':
    calibration = Calibration()
    opts = calibration.parse()
    inputdirL = opts.InputDirL
    inputdirR = opts.InputDirR
    img = opts.image_file

    calibration.getPoints(inputdirL, left=True)
    calibration.getPoints(inputdirR, left=False)
    calibration.calibrate_dozens()
    print('################################################################')

'''
Reference
[1] https://blog.csdn.net/xuelabizp/article/details/50417914
[2] https://github.com/ChihaoZhang/Stereo
'''