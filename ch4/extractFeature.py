# -*- coding: utf-8 -*-
import skimage
import cv2 as cv
import numpy as np

original = skimage.data.horse()
img = 255 - np.uint8(original*255)

contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

img2 = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
cv.drawContours(img2,contours,-1,(255,0,255),2,hierarchy=hierarchy)

hull = cv.convexHull(contours[0])
img3 = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
hull = hull.reshape(1,hull.shape[0],hull.shape[2])
cv.drawContours(img3,hull,-1,(0,0,255),2)

img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
disp = np.hstack([img,img2,img3])
cv.imshow('Horse',disp)

while True:
    key = cv.waitKey(1)
    if key & 0xFF == 27 : # enter ESC
        break
    if cv.getWindowProperty('Horse', cv.WND_PROP_VISIBLE ) < 1:
        break
        
cv.destroyAllWindows()