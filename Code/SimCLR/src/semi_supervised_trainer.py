import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # '..' goes one directory up from src/

# Add the root directory to sys.path if it's not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)  # Insert at the beginning

from model.simclr_model import SimCLR
from src.data_loading import get_simclr_data_loaders
from src.loss_functions import info_nce_loss
from src.utils import save_checkpoint, load_checkpoint, set_seed  # Assuming you have utils.py
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_simclr(model, train_loader, optimizer, device, epochs, temperature, checkpoint_path):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            augmented_images, _ = batch

            try:
                view1, view2 = augmented_images # Original incorrect unpacking
                # print("Unpacking successful!")
            except ValueError as e:
                print(f"Unpacking failed: {e}")
                break  # Stop after the first batch for debugging

            view1 = view1.to(device)
            view2 = view2.to(device)
            optimizer.zero_grad()

            projection1 = model(view1)
            projection2 = model(view2)

            # Concatenate projections for InfoNCE loss
            features = torch.cat([projection1, projection2], dim=0)

            loss = info_nce_loss(features, temperature=temperature)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint periodically
        save_checkpoint(model, optimizer, epoch, checkpoint_path, filename=f"simclr_checkpoint_epoch_{epoch+1}.pth")
    
    return train_losses

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 256
    learning_rate = 2e-3
    epochs = 40
    temperature = 0.1 # Important hyperparameter
    num_workers = 4
    projection_dim = 256
    encoder_name = 'resnet50'
    seed = 42  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "./checkpoints"  # Directory to save checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = checkpoint_dir

    # Set seed for reproducibility
    set_seed(seed)

    # Load the SimCLR data loader
    train_loader = get_simclr_data_loaders(batch_size=batch_size, num_workers=num_workers)

    # Initialize the SimCLR model
    model = SimCLR(encoder_name=encoder_name, projection_dim=projection_dim).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if available
    start_epoch = 0
    checkpoint_file = os.path.join(checkpoint_path, "simclr_checkpoint_latest.pth") # Change this if you want to load a specific checkpoint
    if os.path.exists(checkpoint_file):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, filename="simclr_checkpoint_latest.pth") # Change this if you want to load a specific checkpoint
        print(f"Resuming training from epoch {start_epoch}")

    # Train the SimCLR model
    train_losses = train_simclr(model, train_loader, optimizer, device, epochs, temperature, checkpoint_path)

    # Save the loss data to a file
    loss_data_path = 'simclr_loss_data.npy'
    np.save(loss_data_path, np.array(train_losses))
    print(f"Training loss data saved to: {loss_data_path}")


    print("SimCLR pre-training complete!")
