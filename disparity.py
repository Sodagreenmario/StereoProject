#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import argparse

class Disparity(object):
    def __init__(self):
        super(Disparity, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--Left', type=str, default='left.jpg',
                                 help='the path of the input left image')
        self.parser.add_argument('--Right', type=str, default='right.jpg',
                                 help='the path of the input right image')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('######################### Load Options ########################')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

    def compute_disparity(self, imgL, imgR):
        imgL = cv2.imread(imgL, 0)
        imgR = cv2.imread(imgR, 0)

        window_size = 3
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16,
            blockSize=3,
            P1=8 * 1 * window_size ** 2,
            P2=32 * 1 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # use WLS_Filter to do filtering
        # FILTER Parameters
        lmbda = 8000
        sigma = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        # compute the disparities and convert the resulting images to int16 format
        print('computing disparity...')
        displ = left_matcher.compute(imgL, imgR).astype('int16')  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL).astype('int16')  # .astype(np.float32)/16
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)

        # normalize the depth map and show it
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)

        print('disparity computation succeeded!')
        cv2.imwrite('disparity.jpg', filteredImg)

if __name__ == '__main__':
    disparity = Disparity()
    opts = disparity.parse()
    left = opts.Left
    right = opts.Right

    disparity.compute_disparity(left, right)
    print('################################################################')

'''
Reference
[1] https://github.com/ChihaoZhang/Stereo/blob/master/Binocular%20Camera%20Calibration/stereo_matching.py
'''