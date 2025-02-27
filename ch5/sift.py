# -*- coding: utf-8 -*-
import cv2 as cv
import sys
import numpy as np

img1=cv.imread('mot_color70.jpg')[190:350,440:560]
if img1 is None: sys.exit('File not found.')
img2=cv.imread('mot_color83.jpg')
if img2 is None: sys.exit('File not found.')
    
sift=cv.SIFT_create()
gray1,gray2=cv.cvtColor(img1,cv.COLOR_BGR2GRAY),cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)

flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1,des2,2)

T = 0.7
good_match = []
for k1, k2 in knn_match:
    if (k1.distance/k2.distance) < T:
        good_match.append(k1)

points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.queryIdx].pt for gm in good_match])

H,_=cv.findHomography(points1,points2,cv.RANSAC)

h1,w1=img1.shape[0],img1.shape[1]
h2,w2=img2.shape[0],img2.shape[1]

box1=np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)
box2=cv.perspectiveTransform(box1,H)

img2 = cv.polylines(img2,[np.int32(box2)],True,(0,255,0),8)

img_match = np.empty((max(img.shape[0],img2.shape[0]),img.shape[1]+img2.shape[1],3),dtype=np.uint8)
cv.drawMatches(img,key,img2,key2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matches and Homography',img_match)

while True:
    if cv.waitKey(1) == ord('q'): break
    if cv.getWindowProperty('Matches and Homography',cv.WND_PROP_VISIBLE) < 1: break

cv.destroyAllWindows()
sys.exit()
