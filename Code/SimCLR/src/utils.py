# src/utils.py
import torch
import random
import numpy as np
import os

def save_checkpoint(model, optimizer, epoch, checkpoint_path, filename="checkpoint.pth"):
    """Saves a model checkpoint to the specified path.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_path (str): The directory to save the checkpoint in.
        filename (str, optional): The filename for the checkpoint. Defaults to "checkpoint.pth".
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(checkpoint_path, filename))
    torch.save(state, os.path.join(checkpoint_path, "simclr_checkpoint_latest.pth"))  # Save a "latest" checkpoint for convenience
    print(f"Checkpoint saved to {os.path.join(checkpoint_path, filename)}")

def load_checkpoint(model, optimizer, checkpoint_path, filename="checkpoint.pth"):
    """Loads a model checkpoint from the specified path.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str): The directory where the checkpoint is located.
        filename (str, optional): The filename of the checkpoint. Defaults to "checkpoint.pth".

    Returns:
        int: The epoch number from the loaded checkpoint.
    """
    checkpoint = torch.load(os.path.join(checkpoint_path, filename))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {os.path.join(checkpoint_path, filename)} at epoch {epoch}")
    return epoch

def set_seed(seed):
    """Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
