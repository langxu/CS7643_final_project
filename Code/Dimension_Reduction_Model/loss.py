import torch.nn as nn
import torch

def get_loss_function():
    """
    Returns the Cross-Entropy loss function for classification.
    """
    return nn.CrossEntropyLoss()

def get_triplet_loss_function(margin=1.0):
    """
    Returns the Triplet Margin loss function for embedding learning.
    """
    # p=2 means use L2 norm for distance
    # margin is the minimum difference required between d(a,p) and d(a,n)
    return nn.TripletMarginLoss(margin=margin, p=2)

