import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Base Classifier Head ---
# This remains the same, it just takes an embedding vector
class ClassifierHead(nn.Module):
    """
    A simple classifier head that takes the embedding and outputs class scores.
    """
    def __init__(self, embedding_dim, num_classes=4):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Output: num_classes (10 for CIFAR-10)

    def forward(self, embedding):
        x = self.relu(self.fc1(embedding))
        logits = self.fc2(x)
        return logits

# --- Specific Encoder Implementations ---

class CNNEncoder(nn.Module):
    """
    A simple CNN encoder for 32x32 input, mapping to embedding_dim.
    """
    def __init__(self, embedding_dim=64):
        super(CNNEncoder, self).__init__()
        # Input: 3x32x32 (CIFAR-10)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # Output: 32x32x32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x16x16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Output: 64x16x16
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x8x8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Output: 128x8x8
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x4x4

        # Calculate the size of the flattened features before the linear layer
        self._to_linear = 128 * 4 * 4

        # Final layer maps to the embedding dimension
        self.fc = nn.Linear(self._to_linear, embedding_dim)

        # Store embedding_dim for access by InvariantMappingModel
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the output of the convolutional layers
        x = x.view(-1, self._to_linear) # -1 lets torch infer the batch size

        embedding = self.fc(x)
        return embedding


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 based encoder adapted for 32x32 input (like CIFAR-10),
    mapping to embedding_dim.
    """
    def __init__(self, embedding_dim=64):
        super(ResNet18Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=False) # weights=None for training from scratch

        # Adapt the first layer for 32x32 images.
        # Standard ResNet uses kernel 7, stride 2, padding 3 which reduces
        # spatial dimensions too aggressively for 32x32.
        # We'll use kernel 3, stride 1, padding 1 and remove the initial maxpool.
        # This is a common adaptation for small image datasets.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        # self.maxpool = resnet18.maxpool # Removed

        # Remaining layers (blocks) from ResNet
        self.layer1 = resnet18.layer1 # 64 filters
        self.layer2 = resnet18.layer2 # 128 filters
        self.layer3 = resnet18.layer3 # 256 filters
        self.layer4 = resnet18.layer4 # 512 filters

        # Adaptive average pooling before the final linear layer
        # Original ResNet has AvgPool2d(7x7) for 224x224 input
        # For 32x32 input with our modified conv1 and no maxpool,
        # spatial dimensions after layer4 are 4x4 (32 -> 32 -> 16 -> 8 -> 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Pools down to 1x1 spatial dimensions

        # The output size before the original final FC layer is 512 (from layer4)
        # We replace the original FC layer with one mapping to embedding_dim
        self.fc = nn.Linear(512, embedding_dim)

        # Store embedding_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Forward pass adapted from ResNet-18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # Skip maxpool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten the output from avgpool

        embedding = self.fc(x)
        return embedding

# --- Combined Model ---

class InvariantMappingModel(nn.Module):
    """
    Combines a specified encoder and classifier head.
    Trains the encoder to produce embeddings suitable for classification,
    implicitly learning invariance via data augmentation.
    """
    def __init__(self, encoder: nn.Module, num_classes=4):
        """
        Args:
            encoder (nn.Module): An instance of the encoder model
                                 (e.g., CNNEncoder, ResNet18Encoder).
                                 Must have an 'embedding_dim' attribute.
            num_classes (int): Number of classification classes.
        """
        super(InvariantMappingModel, self).__init__()
        self.encoder = encoder
        # Ensure encoder has the embedding_dim attribute
        if not hasattr(encoder, 'embedding_dim'):
            raise AttributeError("Encoder model must have an 'embedding_dim' attribute.")

        self.classifier = ClassifierHead(self.encoder.embedding_dim, num_classes)
        self.num_classes = num_classes # Store num_classes as well

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits # We output logits directly for CrossEntropyLoss

    def get_embedding(self, x):
        """
        Helper function to get the embedding vector for an input image.
        """
        return self.encoder(x)


# --- Helper function to build encoders ---
def build_encoder(encoder_name: str, embedding_dim: int):
    """
    Builds and returns an encoder instance based on name.

    Args:
        encoder_name (str): Name of the encoder ('cnn', 'resnet18').
        embedding_dim (int): The desired dimensionality of the output embedding.

    Returns:
        nn.Module: An instance of the specified encoder.

    Raises:
        ValueError: If encoder_name is not recognized.
    """
    if encoder_name.lower() == 'cnn':
        return CNNEncoder(embedding_dim)
    elif encoder_name.lower() == 'resnet18':
        return ResNet18Encoder(embedding_dim)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}. Choose from 'cnn', 'resnet18'.")
