import os
import cv2 as cv

# NAME = ['00000000.jpg', '00000001.jpg', '00000002.jpg', '00000003.jpg', '00000004.jpg', '00000005.jpg', '00000006.jpg', '00000007.jpg', '00000008.jpg', '00000009.jpg', '00000010.jpg', '00000011.jpg', '00000012.jpg', '00000013.jpg', '00000014.jpg', '00000015.jpg', '00000016.jpg', '00000017.jpg', '00000018.jpg', '00000019.jpg', '00000020.jpg', '00000021.jpg', '00000022.jpg'];

# for n in NAME:

# 	img = cv.imread('images/'+n);
# 	res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC);

# 	cv.imwrite('images_downsampled/'+n, res);

img = cv.imread('aakash.jpg');
res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC);