# -*- coding: utf-8 -*-
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cifar_dataset = datasets.CIFAR10(root='../',train=True,download=False)
cifar_img,cifar_target = cifar_dataset.data, np.array(cifar_dataset.targets)
imgs, targets = cifar_img[:15],cifar_target[:15]
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

trans = transforms.Compose([
    transforms.RandomAffine(degrees=0.0,translate=(0.2,0.2),interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomRotation((-20,20), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(0.5)
    ])

plt.figure(figsize=(20,2))
plt.suptitle('First 15 images in the train set')
plt.tight_layout()
for i in range(15):
    plt.subplot(1,15,i+1)
    plt.imshow(imgs[i])
    plt.title(class_names[targets[i]])
    plt.xticks([]), plt.yticks([])
plt.show()

n_trials = 3; batch_size = 4
for trial in range(n_trials):
    indices = np.random.choice(15,batch_size,replace=False)
    x,y = imgs[indices],targets[indices]
    x_augmented = [trans(Image.fromarray(img)) for img in x]
    plt.figure(figsize=(8,2.4))
    plt.suptitle(f'Augment {trial+1}')
    plt.tight_layout()
    for i in range(batch_size):
        plt.subplot(1,4,i+1)
        plt.imshow(np.array(x_augmented[i]))
        plt.xticks([]),plt.yticks([])
        plt.title(class_names[y[i]])
    plt.show()
    