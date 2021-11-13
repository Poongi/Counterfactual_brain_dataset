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
num_epochs = 500
batch_size = 256
weight_decay_lambda = 1e-4
learning_rate = 0.001
mementum = 0.9


# data load
data_path = './data/brain_image'
trn_ad = np.load(data_path+'/train/ad/trn_ad.npy')
trn_nc = np.load(data_path+'/train/nc/trn_nc.npy')
vld_ad = np.load(data_path+'/test/ad//vld_ad.npy')
vld_nc = np.load(data_path+'/test/nc/vld_nc.npy')

trn_ad.shape
trn_nc.shape

def image_normalization(image):
    for batch in range(len(image)):
        image[batch] = (image[batch] - np.min(image[batch])) / (np.max(image[batch]) - np.min(image[batch]))
        image[batch] = (image[batch] - 0.5) * 2
    return np.expand_dims(image, axis=-1)

trn_ad_normed = image_normalization(trn_ad)
trn_nc_normed = image_normalization(trn_nc)
vld_ad_normed = image_normalization(vld_ad)
vld_nc_normed = image_normalization(vld_nc)

trn_ad_tensor = torch.from_numpy(trn_ad_normed)
trn_nc_tensor = torch.from_numpy(trn_nc_normed)
vld_ad_tensor = torch.from_numpy(vld_ad_normed)
vld_nc_tensor = torch.from_numpy(vld_nc_normed)

trn_ad_arange = trn_ad_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()
trn_nc_arange = trn_nc_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()
vld_ad_arange = vld_ad_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()
vld_nc_arange = vld_nc_tensor.transpose(2,1).reshape(-1, 1, 96, 96).contiguous().clone()

trn_ad_label = torch.ones(trn_ad_arange.shape[0])
trn_nc_label = torch.zeros(trn_ad_arange.shape[0])
vld_ad_label = torch.ones(vld_ad_arange.shape[0])
vld_nc_label = torch.zeros(vld_nc_arange.shape[0])

train_data = torch.cat([trn_ad_arange, trn_nc_arange])
train_label = torch.cat([trn_ad_label, trn_nc_label]).long()

test_data = torch.cat([vld_ad_arange, vld_nc_arange])
test_label = torch.cat([vld_ad_label, vld_nc_label]).long()


# without_black code
idx_list = []
for i, data in enumerate(train_data):
    if torch.all(data == -1) == False:
        idx_list.append(i)

train_data = train_data[idx_list]
train_label = train_label[idx_list]

idx_list = []
for i, data in enumerate(test_data):
    if torch.all(data == -1) == False:
        idx_list.append(i)
test_data = test_data[idx_list]
test_label = test_label[idx_list]

train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# model load

model = torchvision.models.resnet18(pretrained=False).to(device)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).to(device)
model.fc = torch.nn.Linear(512,2).to(device)
model = DataParallel(model, device_ids=[0,1])

# train

loss_arr = []
total_step = len(train_loader)
max = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

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


for epoch in range(num_epochs) : 
    for i, (images, labels) in enumerate(train_loader) :
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            loss_arr.append(loss)
            
            with torch.no_grad() :
                model.eval()
                acc = evaluation(model, test_loader)
                print('Epoch [{}/{}], Step [{}/{}], Loss :{:.4f}, Val_acc : {:.4f}'
                .format(epoch, num_epochs, i+1, total_step, loss.item(), acc))

    if max < acc :
        max = acc
        print(f'max dev accuracy : {max:.4f}')
        torch.save(model.module.state_dict(), './models/resnet_18_clf_without_black.pt')
            

    with torch.no_grad():
        last_model = torchvision.models.resnet18(pretrained=False).to(device)
        last_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).to(device)
        last_model.fc = torch.nn.Linear(512,2).to(device)
        last_model.load_state_dict(torch.load('./models/resnet_18_clf_without_black.pt'))
        last_model = last_model.to(device)
        last_acc = evaluation(last_model, test_loader)
        print('='*40)
        print('Epoch : [{}/{}] , Accuracy : {:.2f}, Loss :{:.4f}'
        .format(epoch, num_epochs, last_acc*100, loss.item()))
        print('Accuracy of the last Network on the test images : {:.2f} %'
        .format(last_acc * 100))
        print('='*40)

with torch.no_grad():
    last_acc = evaluation(model, test_loader)
    print('*'*40)
    print('Result')
    print('Epoch : [{}/{}] , Accuracy : {:.2f}, Loss :{:.4f}'
    .format(epoch, num_epochs, last_acc*100, loss.item()))
    print('Accuracy of the last Network : {:.2f} %'
    .format(last_acc * 100))
    print('='*40)



