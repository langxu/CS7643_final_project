import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.5):
    """Computes the InfoNCE loss.

    Args:
        features (torch.Tensor): A tensor of shape (2*batch_size, projection_dim) containing the features
            of the augmented pairs.
        temperature (float): Temperature parameter for sharpening the contrastive distribution.

    Returns:
        torch.Tensor: The mean InfoNCE loss.
    """
    #Taken from https://github.com/sthalles/SimCLR/blob/master/simclr.py
    batch_size = features.shape[0] // 2
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0) #2 different views
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() #creating 2d boolean matrix for positive pairs
    labels = labels.to(features.device)
    
    normalized_features = F.normalize(features, dim=1)  # Shape: (2*batch_size, projection_dim)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)  # Shape: (2*batch_size, 2*batch_size)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
    # Concatenate positives and negatives
    logits = torch.cat([positives, negatives], dim=1)  # Shape: (2*batch_size, 2*batch_size)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)  # Shape: (2*batch_size)

    logits = logits / temperature
    loss = F.cross_entropy(logits, labels)
    return loss
