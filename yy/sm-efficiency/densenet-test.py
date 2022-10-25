import torch.optim as optim

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.utils.data

import torch.utils.data.distributed

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from torch.autograd import Variable

from torchvision.models import densenet121
modellr = 1e-4

BATCH_SIZE = 32

EPOCHS = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([

transforms.Resize((224, 224)),

transforms.ToTensor(),

transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])

transform_test = transforms.Compose([

transforms.Resize((224, 224)),

transforms.ToTensor(),

transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
dataset_train = datasets.ImageFolder('Dataset/Train', transform)

dataset_test = datasets.ImageFolder('Dataset/Validation',transform_test)