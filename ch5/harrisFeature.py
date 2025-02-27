# -*- coding: utf-8 -*-
import cv2 as cv
import sys
import numpy as np

img = cv.imread('mot_color70.jpg')
if img is None:
    sys.exit('File not found.')

img_copy = cv.cvtColor(img,cv.COLOR_BGR2GRAY).astype(np.float32)/255.
cv.imshow('Gray',img_copy)
ux, uy = np.array([-1,0,1]), np.array([-1,0,1]).transpose()
kernel = cv.getGaussianKernel(3,1)
g = np.outer(kernel,kernel.transpose())

dx, dy = cv.filter2D(img_copy,cv.CV_32F,ux), cv.filter2D(img_copy,cv.CV_32F,uy)
dyy = dy*dy
dyx = dy*dx
dxx = dx*dx
gdyy = cv.filter2D(dyy,cv.CV_32F,g)
gdyx = cv.filter2D(dyx,cv.CV_32F,g)
gdxx = cv.filter2D(dxx,cv.CV_32F,g)

C = (gdyy*gdxx - gdyx*gdyx) - 0.04*(gdyy + gdxx)*(gdyy + gdxx)
C = (C-C.min()) / (C.max()-C.min())
threshold = C.max()*0.95

for i in range(178,338):
    for j in range(443,576):
        c_center = C[i-1:i+2,j-1:j+2]
        if C[i,j] > threshold and sum(sum(C[i,j]>c_center))==8:
            cv.circle(img,(j,i),3,(0,0,255),1)
cv.imshow('Harris feature',img)
while True:
    if cv.waitKey(1)==ord('q'): break
    if cv.getWindowProperty('Harris feature', cv.WND_PROP_VISIBLE ) < 1: break
cv.destroyAllWindows()