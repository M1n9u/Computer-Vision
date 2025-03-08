# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
cifar_train = datasets.CIFAR10(root='../',train=True,
                               download=True,transform=ToTensor())
cifar_test = datasets.CIFAR10(root='../',train=False,
                              download=False,transform=ToTensor())

train_loader = DataLoader(cifar_train,batch_size=128,shuffle=True)
test_loader = DataLoader(cifar_test,batch_size=128,shuffle=False)

# 모델 생성
class lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25))
        self.fc = nn.Sequential(
            nn.Linear(in_features=1600,out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512,out_features=10))
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,1600)
        x = self.fc(x)
        return x
    
# 학습
model = lenet5()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(),lr=0.001)
n_epoch = 30
train_loss = []; valid_loss = []
train_acc = []; valid_acc = []

for epoch in range(n_epoch):
    model.train()
    loss_train = 0.0; acc_train = 0.0; count = 0.0
    for data, target in train_loader:
        data = data.to(device); target = target.to(device)
        data = data.float()
        target_onehot = nn.functional.one_hot(target,num_classes=10).float()
        output = model(data)
        loss = criterion(output,target_onehot)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_train += loss.cpu().item()
        count += len(data)
        pred = output.argmax(dim=1)
        acc_train += torch.eq(pred,target).sum().item()
    
    loss_train /= count; acc_train = acc_train/count*100.0
    train_loss.append(loss_train); train_acc.append(acc_train)
    loss_valid = 0.0; acc_valid = 0.0; count = 0.0
    model.eval()
    for data, target in test_loader:
        data = data.to(device).float()
        target_onehot = nn.functional.one_hot(target,num_classes=10).float()
        output = model(data)
        loss = criterion(output,target_onehot)
        loss_valid += loss.cpu().item()
        count += len(data)
            
        pred = output.argmax(dim=1)
        acc_valid += torch.eq(pred,target).sum().item()
    loss_valid /= count; acc_valid = acc_valid/count*100.0
    valid_loss.append(loss_valid); valid_acc.append(acc_valid)
    print(f"Epoch={epoch+1}/{n_epoch}, Train Loss={loss_train:.2f}, Train Accuracy={acc_train:.2f} %,\nValidation loss={loss_valid:.2f}, Validation Accuracy={acc_valid:.2f} %")
     
torch.save(model.state_dict(),'./lenet5_trained.h5')

fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss (MSE)',color='tab:red')
ax[0].plot(range(len(train_loss)),train_loss,'r')
ax[0].plot(range(len(valid_loss)),valid_loss,'b')
plt.title('Loss')
ax[0].set_legend(['Train Loss','Validation Loss'])

ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)',color='tab:red')
ax[1].plot(range(len(train_acc)),train_acc,'r')
ax[1].plot(range(len(valid_acc)),valid_acc,'b')
plt.subtitle('Accuracy')
ax[0].set_legend(['Train Accuracy','Validation Accuracy'])

plt.show()