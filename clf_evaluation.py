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

vld_ad_label = torch.ones(vld_ad_arange.shape[0])
vld_nc_label = torch.zeros(vld_nc_arange.shape[0])


test_data = torch.cat([vld_ad_arange, vld_nc_arange])
test_label = torch.cat([vld_ad_label, vld_nc_label]).long()

# without_black code
idx_list = []
for i, data in enumerate(test_data):
    if torch.all(data == -1) == False:
        idx_list.append(i)
test_data_without_black = test_data[idx_list]
test_label_without_black = test_label[idx_list]

test_dataset = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

test_dataset_without_black = TensorDataset(test_data_without_black, test_label_without_black)
test_loader_without_black = DataLoader(test_dataset_without_black, batch_size=batch_size, shuffle=True)

# evaluation function

def evaluation(model, data_loader):
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


# model load
model = torchvision.models.resnet18(pretrained=False).to(device)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).to(device)
model.fc = torch.nn.Linear(512,2).to(device)
model.load_state_dict(torch.load('./models/resnet_18_clf.pt'))
model = model.to(device)
acc = evaluation(model, test_loader)
print(f'model_with_black acc : {acc:.4f}')

model_without_black = torchvision.models.resnet18(pretrained=False).to(device)
model_without_black.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).to(device)
model_without_black.fc = torch.nn.Linear(512,2).to(device)
model_without_black.load_state_dict(torch.load('./models/resnet_18_clf_without_black.pt'))
model_without_black = model_without_black.to(device)
acc = evaluation(model_without_black, test_loader)
print(f'model_without_black acc : {acc:.4f}')

