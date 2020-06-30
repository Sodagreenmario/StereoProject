# Stereo project

The source code of the stereo project.

# Index

### Problem 6 & 7 & 8

#### - Dependency

* numpy==1.18.4

* cv2==3.4.2.16
* scicy==1.4.1

#### - Preparation 

Put images into the dirent `./left` , and run `python3 calibrate.py --InputDir ./left `

To get a undistort image of a given image and write it in the current folder:

`python3 calibrate.py --InputDir ./left --image_file ./left/left12.jpg`

To compare the result with implemented Zhang's method:

`python3 calibrate.py --Input ./left --Zhang True  `

### Problem 12 & 14

#### - Dependency

* numpy==1.18.4

* cv2==3.4.2.16

#### - Preparation

Put images from the left camera into the dirent `./left`, and put images for the right camera into the dirent `./right`.

Then run:

`python3 binocular_calibration --InputDirL ./left --OutputDirR ./right`

To rectify the left and the right images, and get the result of a pair of samples(say, `left13.jpg` and `right13.jpg`), run:

`python3 binocular_calibration --InputDirL ./left --OutputDirR ./right --Rectify True --img_index 13 `