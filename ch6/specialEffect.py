# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
import numpy as np
import cv2 as cv
import sys

class specialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('특수 효과')
        self.setGeometry(10,50,800,200)
        
        self.readImgButton = QPushButton('사진 읽기',self)
        self.embossButton = QPushButton('엠보싱',self)
        self.cartoonButton = QPushButton('카툰',self)
        self.sketchButton = QPushButton('연필 스케치',self)
        self.oilBUtton = QPushButton('유화',self)
        self.saveButton = QPushButton('저장',self)
        self.quitButton = QPushButton('나가기',self)
        
        self.readImgButton.setGeometry(10,10,100,30)
        self.embossButton.setGeometry(110,10,100,30)
        self.cartoonButton.setGeometry(210,10,100,30)
        self.sketchButton.setGeometry(310,10,100,30)
        self.oilBUtton.setGeometry(410,10,100,30)
        self.saveButton.setGeometry(510,10,100,30)
        self.quitButton.setGeometry(610,10,100,30)
        
        self.readImgButton.clicked.connect(self.readImgFunction)
        self.embossButton.clicked.connect(self.embossFunction)
        self.cartoonButton.clicked.connect(self.cartoonFunction)
        self.sketchButton.clicked.connect(self.sketchFunction)
        self.oilBUtton.clicked.connect(self.oilFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)
    
    def readImgFunction(self):
        fname = QFileDialog.getOpenFileName(self, '이미지 열기','./',self.tr("Images (*.jpg);; All Files(*.*)"))
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('File not found.')
        self.img_save = np.copy(self.img)
        cv.imshow('Original Image',self.img)
        
    def embossFunction(self):
        femboss = np.array([[-1,0,0],[0,0,0],[0,0,1]],dtype=np.float32)
        gray = cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        gray16 = np.int16(gray)
        self.img_save = np.uint8(np.clip(cv.filter2D(gray16,-1,femboss)+128,0,255))
        cv.imshow('Special Effect Image',self.img_save)
        
    def cartoonFunction(self):
        self.img_save = cv.stylization(self.img,sigma_s=60,sigma_r=0.45)
        cv.imshow('Special Effect Image',self.img_save)
        
    def sketchFunction(self):
        _,self.img_save = cv.pencilSketch(self.img,sigma_s=60,sigma_r=0.45,shade_factor=0.02)
        cv.imshow('Special Effect Image',self.img_save)
        
    def oilFunction(self):
        self.img_save = cv.xphoto.oilPainting(self.img, 10, 1, cv.COLOR_BGR2Lab)
        cv.imshow('Special Effect Image',self.img_save)
        
    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '이미지 저장','./',self.tr("Images (*.jpg);; All Files(*.*)"))
        cv.imwrite(fname[0],self.img_save)
        
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = specialEffect()
win.show()
app.exec_()