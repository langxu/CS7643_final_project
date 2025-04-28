# src/data_loading.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # '..' goes one directory up from src/

# Add the root directory to sys.path if it's not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)  # Insert at the beginning

# Custom Gaussian Blur transform
class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        from torchvision.transforms import functional as F
        img = F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigma)
        return img

# Helper class to generate two augmented views of the same image
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return (self.base_transform(x), self.base_transform(x))

DATA_DIR = './data/cifar10'

def download_cifar10(root_dir=DATA_DIR):
    """Downloads the full CIFAR-10 dataset to the specified root directory."""
    os.makedirs(root_dir, exist_ok=True)
    datasets.CIFAR10(root=root_dir, train=True, download=True)
    datasets.CIFAR10(root=root_dir, train=False, download=True)
    print(f"CIFAR-10 dataset downloaded to: {root_dir}")

def get_trimmed_cifar10_loaders(root_dir=DATA_DIR, batch_size=64, train=True, transform=None, num_workers=4):
    """Loads the CIFAR-10 dataset, trimmed to include only 'dog', 'cat', and 'horse' classes.

    Args:
        root_dir (str): The root directory where the CIFAR-10 dataset is stored.
        batch_size (int): The batch size for the DataLoader.
        train (bool): If True, loads the training set; otherwise, loads the test set.
        transform (callable, optional): A function/transform to apply to the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the trimmed CIFAR-10 dataset.
    """
    full_dataset = datasets.CIFAR10(root=root_dir, train=train, download=False, transform=transform)

    # Define the classes we want to keep
    target_classes = ['dog', 'cat', 'ship', 'truck']
    class_to_idx = full_dataset.class_to_idx
    target_indices = [class_to_idx[cls] for cls in target_classes]

    # Get indices of samples belonging to the target classes
    indices = np.where(np.isin(full_dataset.targets, target_indices))[0]

    # Create a subset of the dataset with only the desired classes
    subset = Subset(full_dataset, indices)

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {'training' if train else 'testing'} data for classes: {target_classes} from {root_dir}")
    return dataloader

def get_simclr_data_loaders(root_dir=DATA_DIR, batch_size=64, num_workers=4):
    """Creates data loaders for SimCLR pre-training with augmentations,
    restricted to 'dog', 'cat', and 'horse' classes.
    """

    # Define the set of augmentations for SimCLR
    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # Load the full CIFAR-10 training dataset
    full_train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=False, transform=TwoCropsTransform(simclr_transform))
    # Define the classes we want to keep
    target_classes = ['dog', 'cat', 'ship', 'truck']
    class_to_idx = full_train_dataset.class_to_idx
    target_indices = [class_to_idx[cls] for cls in target_classes]

    # Get indices of samples belonging to the target classes
    train_indices = np.where(np.isin(full_train_dataset.targets, target_indices))[0]

    # Create a subset of the training data with only the desired classes
    trimmed_train_dataset = Subset(full_train_dataset, train_indices)

    # Create the DataLoader that applies the TwoCropsTransform with the SimCLR augmentations
    train_loader = DataLoader(
        dataset=trimmed_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"SimCLR pre-training data loader created for classes: {target_classes} from {root_dir}")
    return train_loader

if __name__ == '__main__':
    # Example usage:
    download_cifar10(DATA_DIR)

    # Basic transform for trimmed loader demonstration
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader_trimmed = get_trimmed_cifar10_loaders(train=True, transform=basic_transform, batch_size=32)
    test_loader_trimmed = get_trimmed_cifar10_loaders(train=False, transform=basic_transform, batch_size=32)

    print(f"Number of training batches (trimmed): {len(train_loader_trimmed)}")
    print(f"Number of testing batches (trimmed): {len(test_loader_trimmed)}")

    train_loader_simclr = get_simclr_data_loaders(batch_size=32)
    print(f"Number of training batches (SimCLR): {len(train_loader_simclr)}")

    # You can inspect a batch from the SimCLR loader
    for (augmented_images, _), _ in train_loader_simclr:
        print("Shape of augmented images (batch, 2 views, C, H, W):", augmented_images.shape)
        break
