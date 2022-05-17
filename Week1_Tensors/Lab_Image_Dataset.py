"""
Lab 5) Images Dataset Lab
Loads the MNIST Fashion Dataset
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

torch.manual_seed(0)

# The following functions will be used as components of the dataset object. These are known as auxiliary functions.
# Load CSV data
directory = ''
csv_file = 'index.csv'
csv_path = os.path.join(directory, csv_file)

# Convert into Pandas dataframe
data_name = pd.read_csv(csv_path)
print(data_name.head())

# Filename #1: data_name.iloc[0, 1]
# Item #1: data_name.iloc[0, 0]
# Length of the data frame: data_name.shape[0]

# Load image
image_name = data_name.iloc[0, 1]
image_path = os.path.join(directory, image_name)

# Plot image
image = Image.open(image_path)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[0, 0])
# plt.show()


# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, directory, csv_file, transform=None):
        self.directory = directory
        self.csv_file = csv_file
        self.csv_path = os.path.join(self.directory, self.csv_file)
        self.image_data = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.image_data.iloc[index, 1])
        image = Image.open(image_path)
        image_name = self.image_data.iloc[index, 0]

        if self.transform:
            image = self.transform(image)

        return image, image_name

    def __len__(self):
        return self.image_data.shape[0]

def show_data(dataset, index):
    plt.imshow(dataset[index][0], cmap='gray', vmin=0, vmax=255)
    plt.title(image_dataset[index][1])
    plt.show()

# Create dataset
image_dataset = ImageDataset(directory='', csv_file='index.csv', transform=transforms.CenterCrop(10))
show_data(image_dataset, 41)