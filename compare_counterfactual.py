from numpy.core.fromnumeric import transpose
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import DataParallel
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_normalization(image):
    tmp_values = image.reshape(image.shape[0], -1)
    min_values = np.min(tmp_values, axis=1)
    max_values = np.max(tmp_values, axis=1)
    min_values = np.expand_dims(min_values, axis=[i for i in range(1, (len(image.shape)))])
    max_values = np.expand_dims(max_values, axis=[i for i in range(1, (len(image.shape)))])
    normed_image = (image - min_values) / (max_values - min_values)
    # normed_image = (normed_image - 0.5)*2
    return np.expand_dims(normed_image, axis=0)


file_path = './GC_brain/result/brain_test/'
dir_list = os.listdir(file_path)

cnt = 0
for dir in dir_list:
    img_list = os.listdir(file_path+dir+'/')
    img_list.sort()
    first_number = re.sub(r'[^0-9]', '', img_list[0])
    second_number = re.sub(r'[^0-9]', '', img_list[1])
    if first_number=='0' and second_number=='1':
        cnt += 1
        
        f, ax = plt.subplots(1, 4, figsize=(13,13))
        first_image = plt.imread(file_path+dir+'/'+img_list[0])
        second_image = plt.imread(file_path+dir+'/'+img_list[1])
        ax[0].imshow(first_image, cmap='gray')
        ax[0].set_title('number '+dir+' '+img_list[0])
        
        ax[1].imshow(second_image, cmap='gray')
        ax[1].set_title('number '+dir+' '+img_list[1])
        
        
        diff = (second_image-first_image).reshape(1,96, 96).squeeze()
        diff[np.where(diff>0.3)] = 1
        diff[np.where(diff<=0.3)] = 0
        ax[2].set_title('number '+dir+' '+'difference')
        ax[2].imshow(diff, cmap='gray')

        hightlighted = first_image.copy()
        hightlighted[np.where(diff>0.5)] = 1
        rgb = np.stack((hightlighted, first_image, first_image), axis=2)
        ax[3].set_title('number '+dir+' ' +'highlight')
        ax[3].imshow(rgb, cmap='gray')
        

print(f'cnt : {cnt}')
print(f'all : {len(dir_list)}')



cnt = 0
for dir in dir_list:
    img_list = os.listdir(file_path+dir+'/')
    img_list.sort()
    first_number = re.sub(r'[^0-9]', '', img_list[0])
    second_number = re.sub(r'[^0-9]', '', img_list[1])
    if first_number=='1' and second_number=='0':
        cnt += 1
        
        f, ax = plt.subplots(1, 4, figsize=(13,13))
        first_image = plt.imread(file_path+dir+'/'+img_list[0])
        second_image = plt.imread(file_path+dir+'/'+img_list[1])
        ax[0].imshow(first_image, cmap='gray')
        ax[0].set_title('number '+dir+' '+img_list[0])
        
        ax[1].imshow(second_image, cmap='gray')
        ax[1].set_title('number '+dir+' '+img_list[1])
        
        
        diff = (second_image-first_image).reshape(1,96, 96).squeeze()
        diff[np.where(diff>0.3)] = 1
        diff[np.where(diff<=0.3)] = 0
        ax[2].set_title('number '+dir+' '+'difference')
        ax[2].imshow(diff, cmap='gray')

        hightlighted = first_image.copy()
        hightlighted[np.where(diff>0.5)] = 1
        rgb = np.stack((hightlighted, first_image, first_image), axis=2)
        ax[3].set_title('number '+dir+' ' +'highlight')
        ax[3].imshow(rgb, cmap='gray')
        

print(f'cnt : {cnt}')
print(f'all : {len(dir_list)}')