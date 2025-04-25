import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()

# Constants (must match training)
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SELECTED_CLASSES = ['cat', 'dog']
NUM_CLASSES = len(SELECTED_CLASSES)
BATCH_SIZE = 64
NORMALIZATION_VARIABLES = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2470, 0.2435, 0.2616)
}

# model selection
MODEL = MODEL1 # Wei's Model
MODEL = MODEL2 # Tianyi's Model

class FilteredCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root, train=train, transform=transform, download=download)
        selected_class_indices = [CIFAR10_CLASSES.index(c) for c in SELECTED_CLASSES]
        mask = np.isin(self.targets, selected_class_indices)
        self.data = self.data[mask]
        self.targets = [SELECTED_CLASSES.index(CIFAR10_CLASSES[t]) for t in np.array(self.targets)[mask]]
        self.classes = SELECTED_CLASSES


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


def get_normalizer():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZATION_VARIABLES)
    ])


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

    return total_loss / total, 100. * correct / total, per_class_accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create test dataset and loader
    test_transform = get_normalizer()
    test_set = FilteredCIFAR10(root='./data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model and load EMA weights
    model = WideResNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('best_model_ema.pth', map_location=device))

    # Evaluate
    test_loss, test_acc, test_per_class_acc = evaluate(model, test_loader, device)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    for i, class_name in enumerate(SELECTED_CLASSES):
        logger.info(f"{class_name} Test Accuracy: {test_per_class_acc[i]:.2f}%")


if __name__ == '__main__':
    main()