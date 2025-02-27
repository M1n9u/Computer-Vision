# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:45:37 2025

@author: disre
"""

import cv2 as cv
import sys

img = cv.imread('girl_laughing.jpg')
if img is None:
    sys.exit('File not found.')

def draw(event,x,y,flag,param):
    global ix, iy
    if event==cv.EVENT_LBUTTONDOWN or event==cv.EVENT_RBUTTONDOWN:
        ix, iy = x, y
    elif event==cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)
    elif event==cv.EVENT_RBUTTONUP:
        cv.rectangle(img, (ix,iy), (x,y), (255,0,0), 2)
    
    cv.imshow('Drawing',img)

cv.namedWindow('Drawing')
cv.imshow('Drawing',img)
cv.setMouseCallback('Drawing',draw)

while True:
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break