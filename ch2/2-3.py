# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:16:33 2025

@author: disre
"""

import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

cv.imshow('Original image', img)
cv.imshow('Gray image', gray)
cv.imshow('Gray small image', gray_small)

cv.waitKey()
cv.destroyAllWindows()