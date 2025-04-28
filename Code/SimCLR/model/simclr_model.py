import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    def __init__(self, encoder_name='resnet18', projection_dim=256):
        super().__init__()
        self.encoder = self._get_encoder(encoder_name)
        self.n_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # Remove the classification head
        self.projection_head = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim)
        )

    def _get_encoder(self, encoder_name):
        if encoder_name == 'resnet18':
            encoder = models.resnet18(pretrained=False)
            return encoder
        elif encoder_name == 'resnet50':
            encoder = models.resnet50(pretrained=False)
            return encoder
        else:
            raise ValueError(f"Encoder '{encoder_name}' not supported: {encoder_name}")

    def forward(self, x):
        features = self.encoder(x)
        projected_features = self.projection_head(features)
        return projected_features
