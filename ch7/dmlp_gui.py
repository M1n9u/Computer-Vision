# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import cv2 as cv
import matplotlib.pyplot as plt

class mlp(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1,10)
        x = self.softmax(x)
        return x
    
layers = nn.Sequential(nn.Flatten(2,-1),
                    nn.Linear(in_features=784,out_features=512),
                    nn.Tanh(),
                    nn.Linear(in_features=512,out_features=10))

model = mlp(layers)
model.load_state_dict(torch.load('./dmlp_trained.h5'))
model.eval()

def reset():
    global img
    img = np.ones((200,520,3),dtype=np.uint8)*255
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(110+i*100,150),(0,0,255))
    cv.putText(img,'e:erase, s:show, r:recognize, q:quit',(10,40),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)

def grab_numerals():
    numerals = []
    for i in range(5):
        num = img[51:149,11+i*100:109+i*100,0]
        num = 255-cv.resize(num,(28,28),interpolation=cv.INTER_CUBIC)
        numerals.append(num)
    numerals = np.array(numerals)
    return numerals

def show():
    numerals = grab_numerals()
    fig,ax = plt.subplots(1,5,figsize=(25,5))
    for i in range(5):
        ax[i].imshow(numerals[i],cmap='gray')
        ax[i].set_xticks([]); ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()

def recognize():
    numerals = grab_numerals()
    numerals = numerals.astype(np.float32)/255.0
    num_torch = torch.from_numpy(numerals).unsqueeze(dim=1)
    out = model(num_torch).detach().numpy()
    pred = np.argmax(out,axis=1)
    for i in range(5):
        cv.putText(img,str(pred[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    
def write(event,x,y,flags,param):
    brushSize = 4
    LColor=(0,0,0)
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),brushSize,LColor,-1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),brushSize,LColor,-1)

reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing',write)
while True:
    cv.imshow('Writing',img)
    key = cv.waitKey(1)
    if key == ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognize()
    elif key == ord('q'):
        break

cv.destroyAllWindows()