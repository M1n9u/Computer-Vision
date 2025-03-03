# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
mnist_train = datasets.MNIST(root='../',train=True,
                             download=False,transform=ToTensor())
mnist_test = datasets.MNIST(root='../',train=False,
                             download=False,transform=ToTensor())

train_loader = DataLoader(mnist_train,batch_size=128,shuffle=True)
test_loader = DataLoader(mnist_test,batch_size=1,shuffle=False)

# 모델 생성
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

# 학습
model = mlp(layers)
criterion = nn.MSELoss()
optim = optim.Adam(model.parameters(),lr=0.001)
n_epoch = 50

def evaluate():
    acc = 0.0
    model.eval()
    test_data, test_target = mnist_test.data.float(), mnist_test.targets.detach().numpy()
    test_data = test_data.unsqueeze(1)
    out = model(test_data).detach().numpy()
    pred = np.argmax(out,axis=1)
    acc = float(np.sum(pred==test_target))/float(len(test_data))*100.0
    return acc

model.train()
total_loss = []
total_acc = []
for epoch in range(n_epoch):
    epoch_loss = 0.0
    count = 0
    for data, target in train_loader:
        out = model(data)
        target = nn.functional.one_hot(target, num_classes=10).float()
        loss = criterion(out,target)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        epoch_loss += loss.data
        count += 1
    print(f"Epoch= {epoch+1}/{n_epoch}, Loss= {epoch_loss/count}")
    val_acc = evaluate()
    total_acc.append(val_acc)
    total_loss.append(epoch_loss/count)
    
fig,ax = plt.subplots()
ax.set_xlabel('Epochs')
ax.set_ylabel('Train Loss (MSE)',color='tab:red')
ax.plot(range(len(total_loss)),total_loss,'r')
ax2 = ax.twinx()
ax2.set_ylabel('Validation Accuracy (%)',color='tab:blue')
ax2.plot(range(len(total_acc)),total_acc,'b')
plt.title('Loss and Accuracy')
fig.legend(['Train Loss','Validation Accuracy'])
plt.show()