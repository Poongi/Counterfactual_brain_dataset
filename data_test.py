from matplotlib import colors
import torch
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 256

# data load
data_path = './data/brain_image'
vld_ad = np.load(data_path+'/test/ad//vld_ad.npy')
vld_nc = np.load(data_path+'/test/nc/vld_nc.npy')

def image_normalization(image):
    for batch in range(len(image)):
        image[batch] = (image[batch] - np.min(image[batch])) / (np.max(image[batch]) - np.min(image[batch]))
        image[batch] = (image[batch] - 0.5) * 2
    return np.expand_dims(image, axis=-1)

vld_ad_normed = image_normalization(vld_ad)
vld_nc_normed = image_normalization(vld_nc)

vld_ad_tensor = torch.from_numpy(vld_ad_normed)
vld_nc_tensor = torch.from_numpy(vld_nc_normed)

vld_ad_arange = vld_ad_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()
vld_nc_arange = vld_nc_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()

# without_black code
idx_list = []
for i, data in enumerate(vld_ad_arange):
    if torch.all(data == -1) == False:
        idx_list.append(i)
ad_without_black = vld_ad_arange[idx_list]

idx_list = []
for i, data in enumerate(vld_nc_arange):
    if torch.all(data == -1) == False:
        idx_list.append(i)
nc_without_black = vld_nc_arange[idx_list]

ad_normalized = cv2.normalize(np.array(ad_without_black.permute(0,2,3,1)), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
nc_normalized = cv2.normalize(np.array(nc_without_black.permute(0,2,3,1)), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# save the dataset
for i in range(300):
    cv2.imwrite('./GC_brain/ref_data/brain_ref/class0/'+str(i)+'.png', ad_normalized[i])

for i in range(300):
    cv2.imwrite('./GC_brain/ref_data/brain_ref/class1/'+str(i)+'.png', nc_normalized[i])

for i in range(300, nc_normalized.shape[0]):
    cv2.imwrite('./GC_brain/example/brain/'+str(i)+'.png', nc_normalized[i])


