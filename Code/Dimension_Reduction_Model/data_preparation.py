import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from collections import defaultdict
import random

# CIFAR-10 statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Define the original classes we want to keep and their new mappings
TARGET_CLASSES_ORIGINAL = [3, 5, 8, 9] # Original CIFAR-10 labels for cat, dog, ship, truck
NEW_CLASS_MAPPING = {
    3: 0, # cat -> 0
    5: 1, # dog -> 1
    8: 2, # ship -> 2
    9: 3  # truck -> 3
}

# Update the global class names list to reflect the new mapping
CIFAR10_CLASSES = ['cat', 'dog', 'ship', 'truck']
NUM_CLASSES = len(CIFAR10_CLASSES) # The new number of classes

# print(f"Using only classes: {CIFAR10_CLASSES}")
# print(f"New number of classes: {NUM_CLASSES}")


def filter_and_remap_dataset(dataset):
    """
    Filters the dataset to keep only specified original classes and
    remaps their labels according to NEW_CLASS_MAPPING.
    """
    original_data = dataset.data # numpy array
    original_targets = dataset.targets # list

    # Find indices corresponding to the target original classes
    indices_to_keep = [
        i for i, target in enumerate(original_targets)
        if target in TARGET_CLASSES_ORIGINAL
    ]

    # Filter data and targets
    new_data = original_data[indices_to_keep]
    new_targets = [NEW_CLASS_MAPPING[original_targets[i]] for i in indices_to_keep]

    # Update dataset attributes in place
    dataset.data = new_data
    dataset.targets = new_targets

    print(f"Dataset filtered. New number of samples: {len(dataset)}")
    # Note: dataset.classes still holds original names, but targets are remapped
    # We rely on the remapped targets (0-3) and the global CIFAR10_CLASSES list for lookup

    return dataset

class TripletCIFAR10(torch.utils.data.Dataset):
    """
    Custom Dataset to generate Anchor-Positive-Negative triplets
    from the filtered CIFAR-10 dataset.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.data = self.base_dataset.data # Access data and targets from base dataset
        self.targets = self.base_dataset.targets
        self.transform = self.base_dataset.transform # Keep base dataset transforms

        # Create index mapping: class label -> list of indices for that class
        self.class_indices = defaultdict(list)
        for i, target in enumerate(self.targets):
            self.class_indices[target].append(i)

        self.labels = sorted(list(self.class_indices.keys())) # Get the list of unique labels (0, 1, 2, 3)
        print(f"Triplet Dataset created with {len(self)} samples (based on base dataset size).")
        print(f"Classes found in dataset: {self.labels}")


    def __len__(self):
        # We can iterate based on the number of samples in the base dataset.
        # Each sample can serve as an anchor.
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Generates one triplet: (anchor, positive, negative).
        """
        # 1. Get Anchor
        anchor_img, anchor_label = self.base_dataset[index] # base_dataset applies transforms

        # 2. Get Positive (same class as anchor)
        # Choose a random index from the same class, different from anchor's index
        same_class_indices = self.class_indices[anchor_label]
        positive_index = index # Start with anchor index

        # Loop until a different index is found (handle case where class has only one sample)
        # While loop can be slow if class is very small.
        # Better: sample from indices excluding current, if possible.
        if len(same_class_indices) > 1:
            positive_index = random.choice([i for i in same_class_indices if i != index])

        available_positive_indices = [i for i in same_class_indices if i != index]
        if not available_positive_indices:
             # Handle case where only one sample exists for this class (rare in CIFAR subset)
             # For simplicity, reuse anchor as positive, but triplet won't provide signal
             positive_index = index
        else:
             positive_index = random.choice(available_positive_indices)

        positive_img, _ = self.base_dataset[positive_index] # base_dataset applies transforms

        # 3. Get Negative (different class from anchor)
        # Choose a random label that is NOT the anchor label
        negative_label_candidate = random.choice(self.labels)
        while negative_label_candidate == anchor_label:
             negative_label_candidate = random.choice(self.labels)

        # Choose a random index from the negative class
        negative_index = random.choice(self.class_indices[negative_label_candidate])
        negative_img, _ = self.base_dataset[negative_index] # base_dataset applies transforms

        # Return the triplet images and the anchor's remapped label
        return (anchor_img, positive_img, negative_img), anchor_label


def get_cifar10_datasets(data_dir='./data'):
    """
    Prepares, filters, remaps, and returns data loaders for
    a subset of CIFAR-10 training (as triplets) and testing (standard).

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): The directory to download/load the dataset from.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing (train_loader, test_loader).
               train_loader yields batches of (anchor_img, positive_img, negative_img), anchor_label.
               test_loader yields batches of img, label.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Transformations for the training set with data augmentation
    # These transforms will be applied by the base dataset *before* triplet formation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Transformations for the test set (no augmentation, just normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load the base training and test datasets (these apply the transforms)
    base_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    base_test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # --- Filter and Remap Datasets ---
    print("Filtering and remapping base training dataset...")
    base_train_dataset = filter_and_remap_dataset(base_train_dataset)

    print("Filtering and remapping base testing dataset...")
    base_test_dataset = filter_and_remap_dataset(base_test_dataset)
    # --- End Filtering ---

    # Create the Triplet Dataset for training from the filtered base training dataset
    triplet_train_dataset = TripletCIFAR10(base_train_dataset)

    return triplet_train_dataset, base_test_dataset


def get_dataloader(triplet_train_dataset, base_test_dataset, batch_size=128, num_workers=4):

    train_loader = torch.utils.data.DataLoader(
        triplet_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        base_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Filtered CIFAR-10 Triplet train loader and standard test loader created with {num_workers} workers.")

    return train_loader, test_loader
