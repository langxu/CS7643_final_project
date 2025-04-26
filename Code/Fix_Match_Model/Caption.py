import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyperparameters
class Hyperparameters:
    num_epochs = 30
    pretrain_epochs = 15
    batch_size = 64
    learning_rate = 0.001
    rnn_learning_rate = 0.0012
    weight_decay = 2e-4
    cnn_output_dim = 512
    rnn_hidden_dim = 256
    rnn_embedding_dim = 128
    num_layers = 1
    vocab_size = 10  # Updated to accommodate 4 classes (6 fixed + 4 class tokens)
    max_seq_length = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples_per_class = 5
    grad_clip = 5.0
    dropout = 0.5
    teacher_forcing_start = 0.5
    teacher_forcing_end = 0.1
    teacher_forcing_epochs = 20
    class_token_weight = 2.0
    early_stopping_patience = 12
    onlyPred = 0  # New parameter for prediction-only mode

hp = Hyperparameters()

# CIFAR-10 Dataset
class CIFAR10Custom(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.split = split
        self.data = []
        self.labels = []
        # Updated for 4 classes: cat, dog, ship, truck
        self.class_indices = {"cat": 3, "dog": 5, "ship": 8, "truck": 9}
        self.class_names = ["cat", "dog", "ship", "truck"]
        self.class_to_idx = {"cat": 0, "dog": 1, "ship": 2, "truck": 3}
        self.load_data()
        self.vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "this": 3, "is": 4, "a": 5}
        for name in self.class_names:
            self.vocab[name] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def load_data(self):
        if self.split == "train":
            for i in range(1, 6):
                file_path = os.path.join(self.data_dir, f"data_batch_{i}")
                with open(file_path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                self.filter_data(batch)
        else:
            file_path = os.path.join(self.data_dir, "test_batch")
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            self.filter_data(batch)

    def filter_data(self, batch):
        images = batch[b"data"]
        labels = batch[b"labels"]
        images = images.reshape(-1, 3, 32, 32)
        for i, label in enumerate(labels):
            if label in self.class_indices.values():
                self.data.append(images[i])
                class_name = list(self.class_indices.keys())[list(self.class_indices.values()).index(label)]
                self.labels.append(class_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32) / 255.0
        image = torch.tensor(image)
        image = (image - self.mean) / self.std
        label = self.labels[idx]
        caption = ["<START>", "this", "is", "a", label, "<END>"]
        caption_ids = [self.vocab[word] for word in caption if word in self.vocab][:hp.max_seq_length]
        caption_ids += [self.vocab["<PAD>"]] * (hp.max_seq_length - len(caption_ids))
        for cid in caption_ids:
            assert 0 <= cid < hp.vocab_size
        class_idx = self.class_to_idx[label]
        return image, torch.tensor(caption_ids), label, class_idx

# CNN Model
class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(hp.dropout),
        )
        self.fc = nn.Linear(256 * 4 * 4, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# CNN Classifier for Pretraining
class CNNClassifier(nn.Module):
    def __init__(self, cnn):
        super(CNNClassifier, self).__init__()
        self.cnn = cnn
        self.classifier = nn.Linear(hp.cnn_output_dim, 4)  # Updated to 4 classes

    def forward(self, x):
        features = self.cnn(x)
        return self.classifier(features)

# RNN Model
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.feature_projection = nn.Linear(hp.cnn_output_dim, hidden_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=hp.dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions=None, teacher_forcing_ratio=0.0):
        features = self.feature_projection(features)
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            batch_size = features.size(0)
            h0 = features.unsqueeze(0).repeat(hp.num_layers, 1, 1)
            c0 = torch.zeros(hp.num_layers, batch_size, hp.rnn_hidden_dim).to(hp.device)
            outputs = []
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            for t in range(embeddings.size(1)):
                if t == 0 or use_teacher_forcing:
                    input_t = embeddings[:, t:t+1]
                else:
                    predicted = outputs[-1].argmax(dim=1)
                    input_t = self.embedding(predicted).unsqueeze(1)
                rnn_output, (h0, c0) = self.rnn(input_t, (h0, c0))
                output = self.fc(rnn_output.squeeze(1))
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1)
            return outputs
        else:
            batch_size = features.size(0)
            h0 = features.unsqueeze(0).repeat(hp.num_layers, 1, 1)
            c0 = torch.zeros(hp.num_layers, batch_size, hp.rnn_hidden_dim).to(hp.device)
            captions = torch.ones(batch_size, 1, dtype=torch.long).to(hp.device) * dataset.vocab["<START>"]
            outputs = []
            for _ in range(hp.max_seq_length):
                embeddings = self.embedding(captions[:, -1:])
                output, (h0, c0) = self.rnn(embeddings, (h0, c0))
                output = self.fc(output.squeeze(1))
                predicted = output.argmax(dim=1)
                outputs.append(predicted)
                captions = torch.cat([captions, predicted.unsqueeze(1)], dim=1)
                if (predicted == dataset.vocab["<END>"]).all():
                    break
            return torch.stack(outputs, dim=1)

# Combined Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_output_dim, rnn_hidden_dim, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.cnn = CNN(cnn_output_dim)
        self.rnn = RNN(hp.rnn_embedding_dim, rnn_hidden_dim, vocab_size, num_layers)

    def forward(self, images, captions=None, teacher_forcing_ratio=0.0):
        features = self.cnn(images)
        outputs = self.rnn(features, captions, teacher_forcing_ratio)
        return outputs

# Pretraining Function
def pretrain_cnn(cnn, train_loader, val_loader, num_epochs):
    classifier = CNNClassifier(cnn).to(hp.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, _, _, class_idx in train_loader:
            images, class_idx = images.to(hp.device), class_idx.to(hp.device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, class_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), hp.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += class_idx.size(0)
            train_correct += predicted.eq(class_idx).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, _, _, class_idx in val_loader:
                images, class_idx = images.to(hp.device), class_idx.to(hp.device)
                outputs = classifier(images)
                loss = criterion(outputs, class_idx)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += class_idx.size(0)
                val_correct += predicted.eq(class_idx).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_loss)
        print(f"Pretrain Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    return cnn

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer_cnn, optimizer_rnn, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        tf_ratio = hp.teacher_forcing_start - (hp.teacher_forcing_start - hp.teacher_forcing_end) * min(epoch / hp.teacher_forcing_epochs, 1.0)
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (images, captions, _, class_idx) in enumerate(train_loader):
            images, captions, class_idx = images.to(hp.device), captions.to(hp.device), class_idx.to(hp.device)
            optimizer_cnn.zero_grad()
            optimizer_rnn.zero_grad()
            outputs = model(images, captions, teacher_forcing_ratio=tf_ratio)
            loss = criterion(outputs.view(-1, hp.vocab_size), captions[:, 1:].contiguous().view(-1))
            if outputs.size(1) >= 4:
                class_token_pos = 3
                class_loss = criterion(outputs[:, class_token_pos, :], captions[:, class_token_pos+1])
                loss = loss + hp.class_token_weight * class_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.cnn.parameters(), hp.grad_clip)
            torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), hp.grad_clip)
            optimizer_cnn.step()
            optimizer_rnn.step()
            train_loss += loss.item()
            class_token_pos = 4
            if captions.size(1) > class_token_pos:
                pred_class = outputs[:, class_token_pos-1, :].argmax(dim=1)
                true_class = captions[:, class_token_pos]
                train_total += captions.size(0)
                train_correct += pred_class.eq(true_class).sum().item()
            if batch_idx % 100 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(images[:1])
                    caption_ids = pred[0].cpu().numpy()
                    caption = [train_loader.dataset.inverse_vocab.get(id, "<UNK>") for id in caption_ids if id not in [train_loader.dataset.vocab["<PAD>"], train_loader.dataset.vocab["<END>"]]]
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Sample Prediction: {' '.join(caption)}")
                model.train()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (images, captions, _, class_idx) in enumerate(val_loader):
                images, captions, class_idx = images.to(hp.device), captions.to(hp.device), class_idx.to(hp.device)
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, hp.vocab_size), captions[:, 1:].contiguous().view(-1))
                if outputs.size(1) >= 4:
                    class_loss = criterion(outputs[:, class_token_pos, :], captions[:, class_token_pos+1])
                    loss = loss + hp.class_token_weight * class_loss
                val_loss += loss.item()
                if captions.size(1) > class_token_pos:
                    pred_class = outputs[:, class_token_pos-1, :].argmax(dim=1)
                    true_class = captions[:, class_token_pos]
                    val_total += captions.size(0)
                    val_correct += pred_class.eq(true_class).sum().item()
                if batch_idx == 0:
                    pred = model(images[:1])
                    caption_ids = pred[0].cpu().numpy()
                    caption = [val_loader.dataset.inverse_vocab.get(id, "<UNK>") for id in caption_ids if id not in [val_loader.dataset.vocab["<PAD>"], val_loader.dataset.vocab["<END>"]]]
                    print(f"Epoch {epoch+1}, Val Batch 0, Sample Prediction: {' '.join(caption)}")
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Class Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Class Acc: {val_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hp.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return train_losses, val_losses

# Plotting Loss Function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("captioning_loss_plot.png")
    plt.close()

# Plot Sample Images with Captions
def plot_sample_images(model, dataset, num_samples_per_class):
    model.eval()
    class_names = dataset.class_names
    samples = {name: [] for name in class_names}
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for idx in indices:
        _, _, label, _ = dataset[idx]
        if len(samples[label]) < num_samples_per_class:
            samples[label].append(idx)
        if all(len(samples[name]) >= num_samples_per_class for name in class_names):
            break
    fig, axes = plt.subplots(len(class_names), num_samples_per_class, figsize=(num_samples_per_class * 3, len(class_names) * 3))
    if num_samples_per_class == 1:
        axes = np.array([axes]).T
    for i, class_name in enumerate(class_names):
        for j, idx in enumerate(samples[class_name][:num_samples_per_class]):
            image, _, _, _ = dataset[idx]
            image_tensor = image.unsqueeze(0).to(hp.device)
            with torch.no_grad():
                output = model(image_tensor)
            caption_ids = output[0].cpu().numpy()
            caption = []
            for id in caption_ids:
                if id == dataset.vocab["<END>"]:
                    break
                if id not in [dataset.vocab["<PAD>"]]:
                    caption.append(dataset.inverse_vocab.get(id, "<UNK>"))
            caption_text = " ".join(caption)
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
            image_np = np.clip(image_np, 0, 1)
            axes[i, j].imshow(image_np)
            axes[i, j].set_title(caption_text, fontsize=10)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig("sample_images_with_captions.png")
    plt.close()

# Main Execution
def main():
    print(f"Using device: {hp.device}")
    global dataset
    dataset = CIFAR10Custom("data/cifar-10-batches-py", split="test")
    model = ImageCaptioningModel(hp.cnn_output_dim, hp.rnn_hidden_dim, hp.vocab_size, hp.num_layers)
    model = model.to(hp.device)

    if hp.onlyPred == 1:
        # Prediction-only mode
        model.load_state_dict(torch.load("image_captioning_model.pth"))
        model.eval()
        plot_sample_images(model, dataset, hp.num_samples_per_class)
    else:
        # Training mode
        train_dataset = CIFAR10Custom("data/cifar-10-batches-py", split="train")
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=False)
        cnn = CNN(hp.cnn_output_dim)
        print("Pretraining CNN...")
        cnn = pretrain_cnn(cnn, train_loader, val_loader, hp.pretrain_epochs)
        model.cnn = cnn
        criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab["<PAD>"])
        optimizer_cnn = optim.Adam(model.cnn.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        optimizer_rnn = optim.Adam(model.rnn.parameters(), lr=hp.rnn_learning_rate, weight_decay=hp.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer_rnn, mode='min', factor=0.1, patience=7)
        print("Training Captioning Model...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer_cnn, optimizer_rnn, scheduler, hp.num_epochs)
        plot_losses(train_losses, val_losses)
        plot_sample_images(model, dataset, hp.num_samples_per_class)
        torch.save(model.state_dict(), "image_captioning_model.pth")

if __name__ == "__main__":
    main()