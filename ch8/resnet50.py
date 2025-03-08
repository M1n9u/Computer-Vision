# -*- coding: utf-8 -*-
import torch
from torchvision import transforms, models
import cv2 as cv
import numpy as np
import requests

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_names = requests.get(url).text.splitlines()

img = cv.imread('rabbit.jpg')
img = np.array(img,dtype=np.float32) / 255.0
mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
x = trans(cv.resize(img,(224,224),interpolation=cv.INTER_CUBIC)).unsqueeze(0)
pred = resnet50(x)
top5 = pred.topk(5,1,True,True)
print(top5)

for i in range(5):
    prob,indices = top5.values.detach().numpy(), top5.indices.numpy()
    cv.putText(img,class_names[indices[0][i]]+':'+str(prob[0][i]),(10,20+i*20),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

cv.imshow('Recognition result',img)
while True:
    key = cv.waitKey(1)
    if key & 0xFF == 27 : # enter ESC
        break
    if cv.getWindowProperty('Recognition result', cv.WND_PROP_VISIBLE ) < 1:
        break
cv.destroyAllWindows()