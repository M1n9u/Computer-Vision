# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121
from torch.optim import Adam
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()
import gc
gc.collect()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
densenet = densenet121(weights='DEFAULT', num_classes=1000)
densenet.classifier = nn.Linear(in_features=1024,out_features=120)

trans = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
dataset = ImageFolder('Stanford_dogs/Images',transform=trans)
train_dataset, test_dataset = train_test_split(dataset,train_size=0.8,random_state=42)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

densenet.to(device)
n_epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=densenet.parameters(),lr=0.000001)

train_loss = []; train_acc = []
test_loss = []; test_acc = []
for epoch in range(n_epochs):
    densenet.train()
    loss_train, acc_train = 0.0,0.0
    count = 0
    for (data,target) in train_loader:
        data, target = data.to(device), target.to(device)
        output = densenet(data)
        loss = criterion(output,target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_train += loss.cpu().item()
        pred = output.argmax(dim=1)
        acc_train += (pred == target).sum().cpu().item()
        count += len(data)
    train_loss.append(loss_train/len(train_loader))
    train_acc.append(acc_train/float(count)*100.0)
    with torch.no_grad():
        loss_test, acc_test = 0.0,0.0
        count = 0
        densenet.eval()
        for (data,target) in test_loader:
            data, target = data.to(device), target.to(device)
            output = densenet(data)
            loss = criterion(output,target)
            loss_test += loss.cpu().item()
            pred = output.argmax(dim=1)
            acc_test += (pred == target).sum().cpu().item()
            count += len(data)
        test_loss.append(loss_test/len(test_loader))
        test_acc.append(acc_test/float(count)*100.0)
    print(f'Epoch: {epoch+1}, Loss: {train_loss[-1]}, Accuracy: {train_acc[-1]}')

torch.save(densenet.state_dict(),'./densenet_trained.h5')
fig, ax = plt.subplots(1,2,figsize=(12,8))
plt.suptitle('Loss & Accuracy')
ax[0].plot(range(len(train_loss)),train_loss,color='b',label='train')
ax[0].plot(range(len(test_loss)),test_loss,'r',label='test')
ax[0].legend(loc='upper right')
ax[0].set_xlabel('Epoch'), ax[0].set_ylabel('Loss')
plt.title('Loss')

ax[1].plot(range(len(train_acc)),train_acc,color='b',label='train')
ax[1].plot(range(len(test_acc)),test_acc,'r',label='test')
ax[1].legend(loc='upper right')
ax[1].set_xlabel('Epoch'), ax[1].set_ylabel('Accuracy (%)')
plt.title('Accuracy')

