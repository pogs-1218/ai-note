'''
Data sets and data loaders
Dataset stores the samples and their corresponding labels
DataLoader wraps and iterable around the Dataset to enable easy access to the samples.

official link: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
'''
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.

# to use pre-defined dataset
from torchvision import datasets


training_data = datasets.FashionMNIST(
  root='data',
  train=True,
  download=True,
  transform=ToTensor()
)

test_data = datasets.FashionMNIST(
  root='data',
  train=False,
  download=True,
  transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]

