# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:31:10 2025

@author: disre
"""

import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

frames = []
while True:
    ret, frame = cap.read()
    
    if not ret:
        print('프레임 획득에 실패했습니다.')
        break
    
    cv.imshow('Video display', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        frames.append(frame)
        

cap.release()
cv.destroyAllWindows()

if len(frames) > 0:
    imgs = frames[0]
    if len(frames) > 1:
        for frame in frames[1:min(3,len(frames))]:
            imgs = np.hstack((imgs,frame))
    cv.imshow('Captured images', imgs)
    cv.waitKey()
    cv.destroyAllWindows()