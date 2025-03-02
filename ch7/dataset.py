# -*- coding: utf-8 -*-
from torchvision import datasets
import matplotlib.pyplot as plt

mnist_dataset = datasets.MNIST(root='../',train=True,download=True)
mn_data,mn_target = mnist_dataset.data.numpy(), mnist_dataset.targets.numpy()
fig = plt.figure(figsize=(12,12))
plt.tight_layout()
plt.suptitle('MNIST / CIFAR10', fontsize=20)
gs = plt.GridSpec(5,5,fig,height_ratios=[1, 1, 0.3, 1, 1])
for i in range(10):
    ax = fig.add_subplot(gs[i])
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(mn_data[i],cmap='gray')
    ax.set_title(str(mn_target[i]), fontsize=12)

for i in range(10,15):
    ax = fig.add_subplot(gs[i])
    ax.axis('off')

cifar10_dataset = datasets.CIFAR10(root='../',train=True,download=True)
cifar_data,cifar_target = cifar10_dataset.data, cifar10_dataset.targets
class_names = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(15,25):
    ax = fig.add_subplot(gs[i])
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(cifar_data[i],cmap='gray')
    ax.set_title(class_names[cifar_target[i]], fontsize=12)

fig.show()