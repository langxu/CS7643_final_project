import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# Import tqdm
from tqdm import tqdm

# Import modules from our project
# Import get_cifar10_dataloaders, NUM_CLASSES, CIFAR10_CLASSES
from Tianyi_Model.data_preparation import NUM_CLASSES, CIFAR10_CLASSES
from Tianyi_Model.model import (
    InvariantMappingModel,
    build_encoder,
    ClassifierHead,
    ResNet18Encoder,
    CNNEncoder,
)

# Import both loss functions
from Tianyi_Model.loss import get_loss_function, get_triplet_loss_function


# --- Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20  # You can adjust this
EMBEDDING_DIM = 64  # The dimensionality of the reduced space
# NUM_CLASSES is imported from data_preparation

# *** Choose your encoder architecture here ***
ENCODER_ARCHITECTURE = "cnn"  # or 'resnet18'

# *** Add num_workers hyperparameter ***
NUM_WORKERS = 4

# *** Hyperparameters for Triplet Loss ***
TRIPLET_MARGIN = 1.0  # Margin for Triplet Loss
TRIPLET_LOSS_WEIGHT = 1.0  # Weight for the triplet loss term
CLASSIFICATION_LOSS_WEIGHT = 0.5  # Weight for the classification loss term


def train_model(
    train_loader,
    test_loader,
    encoder_name,  # Pass encoder_name here
    embedding_dim,
    num_classes,
    learning_rate,
    num_epochs,
    triplet_margin,
    triplet_weight,
    classification_weight,  # Pass loss weights and margin
):
    """
    Handles the training process with triplet loss, evaluation, plotting, and saving.
    Adds tqdm progress bars.
    """
    # --- Output directory ---
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Include encoder name and num_classes in model save path
    MODEL_SAVE_PATH = os.path.join(
        OUTPUT_DIR,
        f"{encoder_name}_{num_classes}cls_{embedding_dim}dim_{triplet_margin}tm_model.pth",
    )

    # Device configuration (moved inside function as provided by user's code)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    print(
        f"learning_rate: {learning_rate}, \nnum_epochs: {num_epochs} \nembedding_dim: {embedding_dim} \nnum_classes: {num_classes}"
    )
    print(f"Encoder Architecture: {encoder_name}")
    print(
        f"Triplet Margin: {triplet_margin}, Triplet Weight: {triplet_weight}, Classification Weight: {classification_weight}"
    )

    # 2. Model, Loss, and Optimizer
    encoder = build_encoder(encoder_name, embedding_dim)  # Use passed encoder_name
    model = InvariantMappingModel(encoder, num_classes).to(device)
    # Get both loss functions
    classification_criterion = get_loss_function()
    triplet_criterion = get_triplet_loss_function(margin=triplet_margin)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training history for plotting
    train_losses = []  # Total training loss (combined)
    train_accuracies = []  # Accuracy based on classification head
    test_losses = []  # Test loss (classification only)
    test_accuracies = []  # Test accuracy

    print("Starting training...")

    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0  # Sum of total loss (combined)
        running_classification_loss = 0.0  # Sum of classification loss component
        running_triplet_loss = 0.0  # Sum of triplet loss component
        correct_predictions = 0
        total_predictions = 0

        # Wrap train_loader (which yields triplets) with tqdm
        train_loop = tqdm(
            train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        )

        # train_loader yields (anchor_img, positive_img, negative_img), anchor_label
        for (
            anchor_images,
            positive_images,
            negative_images,
        ), anchor_labels in train_loop:

            # Move data to device
            anchor_images = anchor_images.to(device)
            positive_images = positive_images.to(device)
            negative_images = negative_images.to(device)
            anchor_labels = anchor_labels.to(device)

            # Combine images into a single batch for efficiency
            combined_images = torch.cat(
                [anchor_images, positive_images, negative_images], dim=0
            )

            # Forward pass through the model
            # Model outputs logits, but we need embeddings for triplet loss
            combined_logits = model(combined_images)
            combined_embeddings = model.get_embedding(
                combined_images
            )  # Get embeddings separately

            # Split the combined outputs/embeddings back
            batch_size = anchor_images.size(0)
            # Fix: Explicitly provide split sizes as a list
            split_sizes = [batch_size, batch_size, batch_size]
            anchor_embeddings, positive_embeddings, negative_embeddings = torch.split(
                combined_embeddings, split_sizes, dim=0
            )
            anchor_logits, positive_logits, negative_logits = torch.split(
                combined_logits, split_sizes, dim=0
            )

            # Calculate Losses
            # 1. Triplet Loss on Embeddings
            triplet_loss = triplet_criterion(
                anchor_embeddings, positive_embeddings, negative_embeddings
            )

            # 2. Classification Loss on Anchor Logits (or average over all if desired)
            classification_loss = classification_criterion(
                anchor_logits, anchor_labels
            )  # Use anchor logits/labels

            # Combined Loss
            total_loss = (
                triplet_weight * triplet_loss
                + classification_weight * classification_loss
            )

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses and accuracy metrics
            batch_size = anchor_images.size(0)  # Get current batch size
            running_train_loss += (
                total_loss.item() * batch_size
            )  # Accumulate total loss per sample
            running_classification_loss += classification_loss.item() * batch_size
            running_triplet_loss += triplet_loss.item() * batch_size

            # Calculate accuracy for the anchor images in the batch (for training accuracy metric)
            _, predicted = torch.max(anchor_logits.data, 1)
            total_predictions += anchor_labels.size(0)  # Use anchor batch size
            correct_predictions += (predicted == anchor_labels).sum().item()

            # Update tqdm description
            if train_loop.n > 0:
                avg_total_loss = (
                    running_train_loss / (train_loop.n * batch_size)
                    if batch_size > 0
                    else 0
                )  # Handle potential empty batch at the very end if drop_last=False
                avg_clf_loss = (
                    running_classification_loss / (train_loop.n * batch_size)
                    if batch_size > 0
                    else 0
                )
                avg_trip_loss = (
                    running_triplet_loss / (train_loop.n * batch_size)
                    if batch_size > 0
                    else 0
                )
                train_loop.set_postfix(
                    total_loss=f"{avg_total_loss:.4f}",
                    clf_loss=f"{avg_clf_loss:.4f}",
                    trip_loss=f"{avg_trip_loss:.4f}",
                    acc=f"{100. * correct_predictions / total_predictions:.2f}%",  # Training accuracy
                )

        epoch_train_loss = running_train_loss / len(
            train_loader.dataset
        )  # Note: len(train_loader.dataset) is # of ANCHORS
        train_losses.append(epoch_train_loss)

        epoch_train_accuracy = 100.0 * correct_predictions / total_predictions
        train_accuracies.append(epoch_train_accuracy)

        # 4. Evaluation Loop (Standard Classification Evaluation)
        model.eval()  # Set model to evaluation mode
        running_test_loss = 0.0  # Classification loss for test set
        correct_predictions = 0
        total_predictions = 0

        # Wrap test_loader (which yields standard batches) with tqdm
        test_loop = tqdm(
            test_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Test ]"
        )

        with torch.no_grad():  # Disable gradient calculation during evaluation
            for images, labels in test_loop:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass (standard classification)
                outputs = model(images)  # Model returns logits

                # Calculate Classification loss for test set
                loss = classification_criterion(outputs, labels)

                running_test_loss += loss.item() * images.size(0)

                # Calculate accuracy for the batch
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update tqdm description with current average test loss
                if test_loop.n > 0:  # Safety check
                    avg_test_batch_loss = (
                        running_test_loss / (test_loop.n * images.size(0))
                        if images.size(0) > 0
                        else 0
                    )  # Handle potential empty batch
                    test_loop.set_postfix(loss=f"{avg_test_batch_loss:.4f}")

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        epoch_test_accuracy = 100 * correct_predictions / total_predictions
        test_accuracies.append(epoch_test_accuracy)

        # Print final epoch summary below the progress bars
        # Use accumulated losses for printing epoch summary
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Summary: "
            f"Total Train Loss: {epoch_train_loss:.4f}, "
            f"Clf Train Loss: {running_classification_loss / len(train_loader.dataset):.4f}, "  # Avg clf loss per anchor
            f"Trip Train Loss: {running_triplet_loss / len(train_loader.dataset):.4f}, "  # Avg trip loss per anchor
            f"Train Acc: {epoch_train_accuracy:.2f}%, "
            f"Test Loss: {epoch_test_loss:.4f}, "  # Test loss is only classification loss
            f"Test Acc: {epoch_test_accuracy:.2f}% \n\n"
        )

    print("Training finished.")

    # 5. Plotting
    plt.figure(figsize=(12, 6))  # Slightly larger figure
    plt.plot(
        train_losses,
        label=f"Train Total Loss (W_trip={triplet_weight}, W_clf={classification_weight})",
    )

    plt.plot(
        test_losses, label="Test Clf Loss"
    )  # Test loss is only classification loss
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{encoder_name.upper()} Training and Test Loss ({num_classes} Classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{encoder_name}_{num_classes}cls_{embedding_dim}dim_{triplet_margin}tm_triplet_loss_plot.png",
        )  # Update filename
    )
    plt.close()

    plt.figure(figsize=(12, 6))  # Slightly larger figure
    plt.plot(test_accuracies, label="Test Accuracy", color="green")
    plt.plot(train_accuracies, label="Train Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"{encoder_name.upper()} Training and Test Accuracy ({num_classes} Classes)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{encoder_name}_{num_classes}cls_{embedding_dim}dim_{triplet_margin}tm_triplet_accuracy_plot.png",
        )  # Update filename
    )
    plt.close()

    print(f"Plots saved to '{OUTPUT_DIR}' directory.")

    training_csv = pd.DataFrame(
        {
            "train_accuracy": train_accuracies,
            "test_accuracy": test_accuracies,
            "train_loss": train_losses,
            "test_loss": test_losses,
        }
    )
    training_csv['model'] = f'{encoder_name}_{num_classes}cls_{embedding_dim}dim_{triplet_margin}'
    training_csv['epochs'] = [i+1 for i in range(num_epochs)]
    training_csv.to_csv(MODEL_SAVE_PATH.replace('.pth', '.csv'), index=False)

    # 6. Saving the Model Parameters
    # MODEL_SAVE_PATH is defined globally based on ENCODER_ARCHITECTURE and NUM_CLASSES
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model parameters saved to '{MODEL_SAVE_PATH}'")
