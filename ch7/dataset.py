# -*- coding: utf-8 -*-
from torchvision import datasets
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(root='../',train=True,download=True)
x_train, y_train = mnist_train.data.numpy(), mnist_train.targets.numpy()

fig = plt.figure(figsize=(12,10))
gs = plt.GridSpec(5,5,height_ratios=[2,2,1,2,2])
plt.suptitle('MNIST, CIFAR10',fontsize=30)
plt.tight_layout()
for i in range(10):
    ax = fig.add_subplot(gs[i])
    ax.imshow(x_train[i],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(y_train[i]),fontsize=20)

cifar10_dataset = datasets.CIFAR10(root='../',train=True,download=True)
x_train, y_train = cifar10_dataset.data, cifar10_dataset.targets
class_names = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(5):
    ax = fig.add_subplot(gs[i+10])
    ax.axis("off")
    
for i in range(10):
    ax = fig.add_subplot(gs[i+15])
    ax.imshow(x_train[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(class_names[y_train[i]],fontsize=20)
    
fig.show()