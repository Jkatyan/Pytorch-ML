"""
Lab 4) Simple Dataset
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(1)  # Prevents randomization

# Create a dataset class
class MyDataset(Dataset):

    # Constructor
    def __init__(self, length, transform=None):
        self.len = length
        self.transform = transform
        self.x = 2 * torch.ones(length, 2, dtype=torch.int64)
        self.y = torch.ones(length, 1, dtype=torch.int64)

    # Getter
    def __getitem__(self, index):
        return_val = self.x[index], self.y[index]
        if self.transform:
            return_val = self.transform(return_val)
        return return_val

    # Length
    def __len__(self):
        return self.len


# Instantiate dataset, find value on index 1, find length:
dataset = MyDataset(length=10)
print("Value at index 1: " + str(dataset[1]))  # Get directly using []
print("Dataset length: " + str(len(dataset)))


# Create a transform (Change the dataset)
class MultiplyTransform(object):

    # Constructor
    def __init__(self, multiplier=2):
        self.multiplier = multiplier

    # Executer
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.multiplier
        y = y * self.multiplier
        sample = x, y
        return sample


# Create transform object and apply the transform
multi_transform = MultiplyTransform(2)
new_dataset = MyDataset(5, transform=multi_transform)

print("Original: ", dataset[0])
print("New: ", new_dataset[0])

# Composing Transforms
# multiple_transforms = transforms.compose([transform_a(), transform_b()])
# new dataset = MyDataset(5, transform=multiple_transforms)
