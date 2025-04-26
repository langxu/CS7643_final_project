import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import logging
import math
import argparse
from typing import List, Tuple, Callable, Optional
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from collections import namedtuple
from itertools import product
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()

# Constants
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SELECTED_CLASSES = ['ship', 'truck']  # New classes
NUM_CLASSES = len(SELECTED_CLASSES)
IMG_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 210
NUM_LABELED = 250 #250 labeled examples per class.
MU = 7
THRESHOLD = 0.95  # Lowered to reduce bias
LR = 0.03
WD = 0.0005
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0
USE_FOCAL_LOSS = True  # Set to False to use standard CE loss
FOCAL_LOSS_GAMMA = 2  # Moderate focus on hard examples
FOCAL_LOSS_ALPHA = 0.9999  # Balanced class weighting

# Dataset config
NORMALIZATION_VARIABLES = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2616)
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class FilteredCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root, train=train, transform=transform, download=download)
        selected_class_indices = [CIFAR10_CLASSES.index(c) for c in SELECTED_CLASSES]
        mask = np.isin(self.targets, selected_class_indices)
        self.data = self.data[mask]
        self.targets = [SELECTED_CLASSES.index(CIFAR10_CLASSES[t]) for t in np.array(self.targets)[mask]]
        self.classes = SELECTED_CLASSES


class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform=None, return_index=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.return_index = return_index
        self.targets = [self.dataset.targets[idx] for idx in self.indices]
        self.classes = dataset.classes

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        if self.return_index:
            return img, label, self.indices[idx]
        return img, label

    def __len__(self):
        return len(self.indices)


class RandAugment:
    def __init__(self, n=2, m=30, randomized_magnitude=True):
        self.n = n
        self.m = m
        self.randomized_magnitude = randomized_magnitude
        self.augmentation_list = [
            (self.translateX, 0, 0.3),
            (self.translateY, 0, 0.3),
            (self.shearX, 0, 0.3),
            (self.shearY, 0, 0.3),
            (self.rotate, 0, 30),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.equalize, None, None),
            (self.color, 0.1, 1.9),
            (self.autocontrast, 0.05, 0.95),
            (self.posterize, 4, 8),
            (self.solarize, 0, 255),
            (self.contrast, 0.1, 1.9),
        ]

    def __call__(self, img):
        augmentations = random.sample(self.augmentation_list, self.n)
        for transform, min_val, max_val in augmentations:
            if min_val is None:
                img = transform(img, None)
            else:
                magnitude = random.uniform(min_val, max_val) if self.randomized_magnitude else self.m
                img = transform(img, magnitude)
        return img

    def translateX(self, img, mag):
        mag = mag * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, mag * random.choice([-1, 1]), 0, 1, 0))

    def translateY(self, img, mag):
        mag = mag * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, mag * random.choice([-1, 1])))

    def shearX(self, img, mag):
        return img.transform(img.size, Image.AFFINE, (1, mag * random.choice([-1, 1]), 0, 0, 1, 0))

    def shearY(self, img, mag):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, mag * random.choice([-1, 1]), 1, 0))

    def rotate(self, img, mag):
        return img.rotate(mag * random.choice([-1, 1]))

    def brightness(self, img, mag):
        return ImageEnhance.Brightness(img).enhance(mag)

    def sharpness(self, img, mag):
        return ImageEnhance.Sharpness(img).enhance(mag)

    def equalize(self, img, _):
        return ImageOps.equalize(img)

    def color(self, img, mag):
        return ImageEnhance.Color(img).enhance(mag)

    def autocontrast(self, img, mag):
        return ImageOps.autocontrast(img, mag)

    def posterize(self, img, mag):
        return ImageOps.posterize(img, int(mag))

    def solarize(self, img, mag):
        return ImageOps.solarize(img, int(mag))

    def contrast(self, img, mag):
        return ImageEnhance.Contrast(img).enhance(mag)


def get_normalizer():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZATION_VARIABLES)
    ])


def get_weak_augmentation(padding=4):
    return transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=padding, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])


def get_strong_augmentation():
    return RandAugment(n=2, m=30)


class FixMatchTransform:
    def __init__(self, weak_transform, strong_transform=None):
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, img):
        if self.strong is None:
            return self.weak(img)
        else:
            return self.weak(img), self.strong(img)


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=2, dropout_rate=0.0):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.block1 = self._make_block(n, n_channels[0], n_channels[1], 1, dropout_rate, True)
        self.block2 = self._make_block(n, n_channels[1], n_channels[2], 2, dropout_rate)
        self.block3 = self._make_block(n, n_channels[2], n_channels[3], 2, dropout_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(n_channels[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, n, in_planes, out_planes, stride, dropout_rate=0.0, activate_before_residual=False):
        layers = []
        for i in range(int(n)):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes,
                                     out_planes,
                                     i == 0 and stride or 1,
                                     dropout_rate,
                                     activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0, activate_before_residual=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout_rate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            out = self.relu1(self.bn1(x))
        else:
            out = self.bn1(x)
            out = self.relu1(out)
        out = self.conv1(out if self.equalInOut else x)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return torch.add(out, shortcut)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])


def get_datasets(data_dir='./data', num_labeled=NUM_LABELED, num_validation=500):
    normalizer = transforms.Compose([
        get_weak_augmentation(),
        get_normalizer()
    ])
    strong_aug = RandAugment(n=2, m=30)
    unlabeled_transform = transforms.Compose([
        get_weak_augmentation(),
        strong_aug,
        get_normalizer()
    ])
    base_set = FilteredCIFAR10(root=data_dir, train=True, download=True)
    test_set = FilteredCIFAR10(root=data_dir, train=False, transform=normalizer, download=True)
    base_indices = list(range(len(base_set)))
    train_indices, validation_indices = get_uniform_split(base_set.targets, base_indices,
                                                          split_num=len(base_indices) - num_validation)
    labeled_indices, unlabeled_indices = get_uniform_split(base_set.targets, train_indices,
                                                           split_num=num_labeled * NUM_CLASSES)
    labeled_train_set = CustomSubset(base_set, labeled_indices, transform=normalizer)
    unlabeled_train_set = CustomSubset(base_set, unlabeled_indices, transform=FixMatchTransform(
        transforms.Compose([get_weak_augmentation(), get_normalizer()]),
        transforms.Compose([get_weak_augmentation(), strong_aug, get_normalizer()])
    ))
    validation_set = CustomSubset(base_set, validation_indices, transform=normalizer)
    return {"labeled": labeled_train_set, "unlabeled": unlabeled_train_set}, validation_set, test_set


def get_uniform_split(targets, indices, split_num=None, split_pct=None):
    if split_pct is not None:
        samples_per_class = int((split_pct * len(indices)) // len(np.unique(targets)))
    elif split_num is not None:
        samples_per_class = split_num // len(np.unique(targets))
    else:
        raise ValueError('Either split_pct or split_num must be specified')
    split0_indices, split1_indices = [], []
    for class_label in np.unique(targets):
        class_indices = np.where(np.array(targets)[indices] == class_label)[0]
        np.random.shuffle(class_indices)
        split0_indices += list(class_indices[:samples_per_class])
        split1_indices += list(class_indices[samples_per_class:])
    split0_indices = np.array(indices)[split0_indices].tolist()
    split1_indices = np.array(indices)[split1_indices].tolist()
    if split_num is not None and len(split0_indices) < split_num:
        tmp_indices = random.sample(split1_indices, split_num - len(split0_indices))
        split0_indices += tmp_indices
        split1_indices = list(set(split1_indices) - set(tmp_indices))
    return split0_indices, split1_indices


def create_loaders(train_labeled, train_unlabeled, validation_set, test_set, batch_size=BATCH_SIZE, mu=MU):
    train_loader_labeled = DataLoader(
        train_labeled, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_unlabeled = DataLoader(
        train_unlabeled, batch_size=batch_size * mu, shuffle=True, drop_last=True)
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return (train_loader_labeled, train_loader_unlabeled), validation_loader, test_loader


def train_step(model, batch, device, threshold=THRESHOLD, wu=1):
    labeled_batch, unlabeled_batch = batch
    x_weak, labels = labeled_batch
    (u_weak, u_strong), _ = unlabeled_batch
    x_weak = x_weak.to(device)
    labels = labels.to(device)
    u_weak = u_weak.to(device)
    u_strong = u_strong.to(device)
    inputs = torch.cat((x_weak, u_weak, u_strong))
    logits = model(inputs)
    logits_x = logits[:len(x_weak)]
    logits_u_weak, logits_u_strong = logits[len(x_weak):].chunk(2)

    # Labeled loss
    if USE_FOCAL_LOSS:
        loss_fn = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
        labeled_loss = loss_fn(logits_x, labels)
    else:
        labeled_loss = F.cross_entropy(logits_x, labels, reduction='mean', label_smoothing=LABEL_SMOOTHING)

    # Pseudo-labels for unlabeled data
    with torch.no_grad():
        pseudo_labels = torch.softmax(logits_u_weak, dim=1)
        max_probs, targets_u = torch.max(pseudo_labels, dim=1)
        mask = max_probs.ge(threshold).float()

    # Unlabeled loss with focal loss
    if USE_FOCAL_LOSS:
        loss_fn = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction='none')
        unlabeled_loss = (loss_fn(logits_u_strong, targets_u) * mask).mean()
    else:
        unlabeled_loss = (F.cross_entropy(logits_u_strong, targets_u, reduction='none') * mask).mean()

    loss = labeled_loss + wu * unlabeled_loss
    return loss, labeled_loss.item(), unlabeled_loss.item()


def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    per_class_accuracy = []
    for i, class_name in enumerate(SELECTED_CLASSES):
        correct_class = sum((np.array(all_targets) == i) & (np.array(all_preds) == i))
        total_class = sum(np.array(all_targets) == i)
        accuracy = 100. * correct_class / total_class if total_class > 0 else 0.0
        per_class_accuracy.append(accuracy)
    model.train()
    return total_loss / total, 100. * correct / total, per_class_accuracy, all_preds, all_targets


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    torch.manual_seed(42)
    train_sets, validation_set, test_set = get_datasets()
    (train_loader_labeled, train_loader_unlabeled), validation_loader, test_loader = create_loaders(
        train_sets['labeled'], train_sets['unlabeled'], validation_set, test_set)
    model = WideResNet(num_classes=NUM_CLASSES).to(device)
    ema_model = EMA(model, EMA_DECAY)
    optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
    scheduler = LambdaLR(optimizer, lambda step: cosine_lr_decay(step, NUM_EPOCHS * len(train_loader_labeled)))
    val_losses = []
    test_losses = []
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        labeled_loss = 0
        unlabeled_loss = 0
        pbar = tqdm(zip(train_loader_labeled, train_loader_unlabeled),
                    total=len(train_loader_labeled),
                    desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for (labeled_batch, unlabeled_batch) in pbar:
            loss, l_loss, u_loss = train_step(model, (labeled_batch, unlabeled_batch), device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema_model(model)
            total_loss += loss.item()
            labeled_loss += l_loss
            unlabeled_loss += u_loss
            pbar.set_postfix({
                'Loss': f'{total_loss / (pbar.n + 1):.4f}',
                'L Loss': f'{labeled_loss / (pbar.n + 1):.4f}',
                'U Loss': f'{unlabeled_loss / (pbar.n + 1):.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        ema_model.assign(model)
        val_loss, val_acc, val_per_class_acc, _, _ = evaluate(model, validation_loader, device)
        test_loss, test_acc, test_per_class_acc, _, _ = evaluate(model, test_loader, device)
        ema_model.resume(model)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        logger.info(f'Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                    f'{SELECTED_CLASSES[0]} Val Acc: {val_per_class_acc[0]:.2f}%, {SELECTED_CLASSES[1]} Val Acc: {val_per_class_acc[1]:.2f}%, '
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema_model.assign(model)
            torch.save(model.state_dict(), 'best_model_ema_FL.pth')
            logger.info(f'Saved best EMA model at epoch {epoch + 1} with Val Acc: {val_acc:.2f}%')
            ema_model.resume(model)
    model.load_state_dict(torch.load('best_model_ema_FL.pth'))
    ema_model.assign(model)
    test_loss, test_acc, test_per_class_acc, final_preds, final_labels = evaluate(model, test_loader, device)
    logger.info(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                f'{SELECTED_CLASSES[0]} Test Acc: {test_per_class_acc[0]:.2f}%, {SELECTED_CLASSES[1]} Test Acc: {test_per_class_acc[1]:.2f}%')

    # Compute and plot confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FixMatch Validation and Test Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('fixMatch_loss_FL.png')
    plt.close()


def cosine_lr_decay(step, total_steps):
    return max(0.0, math.cos(math.pi * 7 * step / (16 * total_steps)))


if __name__ == '__main__':
    train()