# -*- coding: utf-8 -*-

import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np
from cvplot import plot_cvimg # cv에서 불러온 BGR이미지를 pyplot을 사용해 출력

img = cv.imread('mistyroad.jpg')
if img is None:
    sys.exit('File not found.')

ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(ycrcb)

fig, ax = plt.subplots(2,2, figsize=(15,8))
h = cv.calcHist([ycrcb],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plot_cvimg(ax[0][0],img)
fig.set_label('Histogram equalization')
ax[0][1].plot(h,color='r',linewidth=1)


equal = cv.equalizeHist(y)
h_equal = cv.calcHist([equal],channels=[0],mask=None,histSize=[256],ranges=[0,256])
equal_img = cv.cvtColor(cv.merge([equal, cr, cb]), cv.COLOR_YCrCb2BGR)
plot_cvimg(ax[1][0],equal_img)
ax[1][1].plot(h_equal,color='r',linewidth=1)

img2 = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)
b, g, r, alpha = cv.split(img2)

thres, bin_img2 = cv.threshold(alpha, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
fig, ax = plt.subplots()
plot_cvimg(ax, bin_img2)

h, w = bin_img2.shape
img2_small = bin_img2[h//2:h,0:h//2+1]
fig, ax = plt.subplots(1,4,figsize=(15,5))
plot_cvimg(ax[0], img2_small)

se=np.uint8([[0,0,1,0,0],
             [0,1,1,1,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
              [0,0,1,0,0]])
small_dilation = cv.dilate(img2_small,se,iterations=1)
plot_cvimg(ax[1], small_dilation)
small_erode = cv.erode(img2_small,se,iterations=1)
plot_cvimg(ax[2], small_erode)
small_closing = cv.erode(cv.dilate(img2_small,se,iterations=1),se,iterations=1)
plot_cvimg(ax[3], small_closing)

img3 = cv.imread('soccer.jpg')
img3 = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
img3 = cv.putText(img3,"soccer",(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
smooth_img3 = cv.GaussianBlur(img3,(5,5),0.0)
emboss_filter = np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]])
emboss_img3 = np.uint8(np.clip(cv.filter2D(img3,-1,emboss_filter)+128,0,255))

fig, ax = plt.subplots(1,3,figsize=(15,5))
plot_cvimg(ax[0],img3)
plot_cvimg(ax[1],smooth_img3)
plot_cvimg(ax[2],emboss_img3)