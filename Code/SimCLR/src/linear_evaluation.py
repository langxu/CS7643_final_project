import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.simclr_model import SimCLR  # Import your SimCLR model
from src.data_loading import get_trimmed_cifar10_loaders  # For labeled data
import sys
import os
# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # '..' goes one directory up from src/
# print(root_dir)
# Add the root directory to sys.path if it's not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)  # Insert at the beginning
from torchvision import transforms
import numpy as np

def evaluate_linear(encoder, labeled_loader, test_loader, device, epochs=100, lr=0.01):
    """Evaluates the frozen encoder using a linear classifier."""

    encoder.eval()  # Set encoder to evaluation mode
    n_features = encoder.n_features  # Get the number of features from the encoder
    print(f"n_features: {n_features}")
    linear_classifier = nn.Linear(n_features, 4).to(device)  # 4 classes in CIFAR-10
    optimizer = optim.Adam(linear_classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        linear_classifier.train()  # Set classifier to train mode
        for inputs, targets in tqdm(labeled_loader, desc=f"Linear Eval Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():  # No gradients for encoder
                features = encoder.encoder(inputs)  # Get features from the frozen encoder

            outputs = linear_classifier(features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100. * correct / total
        train_accuracies.append(train_accuracy)  # Append accuracy to the list
        print(f"Linear Eval Epoch {epoch+1}, Loss: {total_loss / len(labeled_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Evaluate on the test set
    linear_classifier.eval()  # Set classifier to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = encoder.encoder(inputs)
            outputs = linear_classifier(features)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100. * correct / total
    test_accuracies.append(train_accuracy)  # Append accuracy to the list
    print(f"Linear Evaluation Test Accuracy: {test_accuracy:.2f}%")
    return train_accuracies, test_accuracies

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load your pre-trained SimCLR model
    simclr_model = SimCLR(encoder_name='resnet18',projection_dim=128).to(device)
    # Replace with the path to your saved SimCLR checkpoint
    checkpoint_path = "./checkpoints/simclr_checkpoint_epoch_100.pth"
    checkpoint = torch.load(checkpoint_path)
    simclr_model.load_state_dict(checkpoint['model_state_dict'])


    # Load data loaders for linear evaluation
    labeled_loader = get_trimmed_cifar10_loaders(
        batch_size=64,
        train=True,  # Load training set for labeled data
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_loader = get_trimmed_cifar10_loaders(
        batch_size=64,
        train=False,  # Load training set for labeled data
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )


    train_accuracies, test_accuracies = evaluate_linear(simclr_model, labeled_loader, test_loader, device, epochs=30, lr=0.001)
    
    # Save the evaluation data to files
    np.save('linear_eval_train_accuracies.npy', np.array(train_accuracies))
    np.save('linear_eval_test_accuracies.npy', np.array(test_accuracies))
    #np.save('linear_eval_train_losses.npy', np.array(train_losses))

    print("Linear evaluation complete. Accuracy and loss data saved.")

