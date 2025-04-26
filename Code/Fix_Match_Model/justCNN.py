import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Constants
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SELECTED_CLASSES = ['cat', 'dog', 'ship', 'truck']
NUM_CLASSES = len(SELECTED_CLASSES)
IMG_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 40
LR = 0.03
WD = 0.0005

# Data config
NORMALIZATION = {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2470, 0.2435, 0.2616]
}

# Your WideResNet model (unchanged)
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

class FilteredCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root, train=train, transform=transform, download=True)
        selected_indices = [CIFAR10_CLASSES.index(c) for c in SELECTED_CLASSES]
        mask = np.isin(self.targets, selected_indices)
        self.data = self.data[mask]
        self.targets = [SELECTED_CLASSES.index(CIFAR10_CLASSES[t]) for t in np.array(self.targets)[mask]]

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZATION)
    ])

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data loading
    transform = get_transforms()
    train_set = FilteredCIFAR10(root='./data', train=True, transform=transform, download=True)
    test_set = FilteredCIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Model setup
    model = WideResNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)

    # Training tracking
    train_losses = []
    val_losses = []
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({
                'train_loss': running_loss / (progress.n + 1),
                'val_loss': val_losses[-1] if val_losses else 'N/A'
            })

        # Calculate epoch metrics
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        epoch_val_loss, epoch_val_acc, _, _ = evaluate(model, test_loader, device)
        val_losses.append(epoch_val_loss)

        logger.info(
            f'Epoch {epoch + 1}: '
            f'Train Loss = {epoch_train_loss:.4f}, '
            f'Val Loss = {epoch_val_loss:.4f}, '
            f'Val Acc = {epoch_val_acc:.2f}%'
        )

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'cnn.pth')
            logger.info(f'New best model saved with val acc {best_val_acc:.2f}%')

    # Final evaluation
    model.load_state_dict(torch.load('cnn.pth'))
    final_val_loss, final_val_acc, final_preds, final_labels = evaluate(model, test_loader, device)
    logger.info(f'\nFinal Validation Accuracy: {final_val_acc:.2f}%')

    # Compute and plot confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cnn_loss.png')
    plt.close()

if __name__ == '__main__':
    train()