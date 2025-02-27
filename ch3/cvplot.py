# -*- coding: utf-8 -*-
import cv2 as cv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def plot_cvimg(ax, img):
    if len(img.shape) == 3: gray = False
    elif len(img.shape) == 2: gray = True
    else: return
    
    rgbimg = deepcopy(img)
    if gray:
        rgbimg = np.uint8(np.clip(rgbimg,0,255))
        ax.imshow(rgbimg, cmap='gray'), ax.set_xticks([]), ax.set_yticks([])
    else:
        rgbimg = cv.cvtColor(rgbimg, cv.COLOR_BGR2RGB)
        b, g, r = cv.split(rgbimg)
        b = np.uint8(np.clip(b,0,255))
        g = np.uint8(np.clip(g,0,255))
        r = np.uint8(np.clip(r,0,255))
        
        rgbimg = cv.merge([b, g, r])
        ax.imshow(rgbimg), ax.set_xticks([]), ax.set_yticks([])
        