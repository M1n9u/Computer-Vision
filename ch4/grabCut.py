# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('File not found.')

disp = np.copy(img)
h,w,c = img.shape
mask = np.zeros((h,w), dtype=np.uint8)
mask[:,:] = cv.GC_PR_BGD

brushSiz = 9
LColor, RColor = (255,0,0),(0,0,255)

def draw(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(disp,(x,y),brushSiz,LColor,-1)
        cv.circle(mask,(x,y),brushSiz,cv.GC_FGD,-1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(disp,(x,y),brushSiz,RColor,-1)
        cv.circle(mask,(x,y),brushSiz,cv.GC_BGD,-1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(disp,(x,y),brushSiz,LColor,-1)
        cv.circle(mask,(x,y),brushSiz,cv.GC_FGD,-1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(disp,(x,y),brushSiz,RColor,-1)
        cv.circle(mask,(x,y),brushSiz,cv.GC_BGD,-1)
    
    cv.imshow('Cut',disp)

cv.namedWindow('Cut')
cv.imshow('Cut',disp)
cv.setMouseCallback('Cut',draw)

while True:
    if cv.waitKey(1) == ord('q'):
        break

background = np.zeros((1,65),dtype=np.float64)
foreground = np.zeros((1,65),dtype=np.float64)

cv.grabCut(img,mask,None,background,foreground,5,cv.GC_INIT_WITH_MASK)
fore_mask = np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD),0,1).astype('uint8')
masked_img = img*fore_mask[:,:,np.newaxis]

cv.imshow('GrabCut image',masked_img)
cv.waitKey()
cv.destroyAllWindows()