import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import argparse
import json

from utils.loader import get_char_loaders, get_img_loaders
from models.cnn import SimpleCNN, PretrainedCNN
from models.rnn import CNNLSTMCTC

CONFIG = None

CNN_MODELS = ['cnn_base', 'cnn_pretrained']
RNN_MODELS = ['rnn_base']

PRINT_EVERY = 250  # Print progress every N batches when running under SLURM

def is_running_under_slurm():
    return 'SLURM_JOB_ID' in os.environ

def save_checkpoint(model, optimizer, epoch, loss_data, checkpoint_path):
    train_losses, val_losses, epochs = loss_data
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': epochs,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint(['train_losses'])
    val_losses = checkpoint(['val_losses'])
    epochs = checkpoint(['epochs'])
    
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch, train_losses, val_losses, epochs

def save_learning_curve(loss_data, model_name, plots_dir):
    train_losses, val_losses, epochs = loss_data 
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss", marker='o', color='blue')
    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s', color='red')
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save to the plots_dir directory
    save_path = os.path.join(plots_dir, f"{model_name}.png")
    plt.savefig(save_path)
    plt.close() 
    print(f"Saved learning curve as {save_path}")

def train():
    print("========== TRAIN.PY START ==========")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    plots_dir = "learning_curves"
    os.makedirs(plots_dir, exist_ok=True)

    model_filename = CONFIG["model"]
    if CONFIG["use_colour"]:
        model_filename += "_color"
    model_path = os.path.join(model_dir, model_filename + ".pt")
    checkpoint_path = os.path.join(model_dir, model_filename + "_checkpoint.pt")

    start_epoch = 0
    use_tqdm = not is_running_under_slurm()

    # Track validation loss
    train_losses = []
    val_losses = []
    epochs = []

    # Load model and data loaders
    if CONFIG["model"] in CNN_MODELS:
        print("Loading data loaders...")
        train_loader, val_loader, classes = get_char_loaders(
            data_path=CONFIG["train_path"],
            batch_size=CONFIG["batch_size"],
            val_split=CONFIG["val_split"],
            colour=CONFIG["use_colour"],
            resize_to=CONFIG["image_size"]
        )

        print(f"Initializing {model_filename} model...")
        in_channels = 3 if CONFIG["use_colour"] else 1
        if CONFIG["model"] == 'cnn_base':
            model = SimpleCNN(num_classes=len(classes), in_channels=in_channels).to(device)
        elif CONFIG["model"] == 'cnn_pretrained':
            model = PretrainedCNN(num_classes=len(classes)).to(device)
        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    elif CONFIG["model"] in RNN_MODELS:
        print("Loading data loaders...")
        train_loader, val_loader, vocab = get_img_loaders(
            data_path=CONFIG["train_path"],
            batch_size=CONFIG["batch_size"],
            val_split=CONFIG["val_split"],
            colour=CONFIG["use_colour"],
            resize_to=CONFIG["image_size"]
        )

        print(f"Initializing {model_filename} model...")
        in_channels = 3 if CONFIG["use_colour"] else 1
        if CONFIG["model"] == 'rnn_base':
            model = CNNLSTMCTC(len(vocab), in_channels=in_channels).to(device)
        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    else:
        raise ValueError(f"Unknown model '{CONFIG['model']}'")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        start_epoch, train_losses, val_losses, epochs = load_checkpoint(model, optimizer, checkpoint_path, device)

    print("Begin training...")
    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        loop = train_loader
        if use_tqdm:
            loop = tqdm(loop, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}]", ncols=100, file=sys.stdout)

        for batch_idx, batch in enumerate(loop):
            if CONFIG["model"] in CNN_MODELS:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if not use_tqdm and batch_idx % PRINT_EVERY == 0:
                    acc = 100. * correct / total if total > 0 else 0.0
                    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
                    sys.stdout.flush()

                if use_tqdm:
                    loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

            elif CONFIG["model"] in RNN_MODELS:
                images, labels, label_lengths = batch
                images, labels = images.to(device), labels.to(device)
                label_lengths = label_lengths.to(device)

                logits = model(images)
                input_lengths = torch.full((images.size(0),), logits.size(0), dtype=torch.long).to(device)

                log_probs = F.log_softmax(logits, dim=2)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if not use_tqdm and batch_idx % PRINT_EVERY == 0:
                    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                    sys.stdout.flush()

                if use_tqdm:
                    loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        epochs.append(epoch + 1)

        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                if CONFIG["model"] in CNN_MODELS:
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_batches += 1

                elif CONFIG["model"] in RNN_MODELS:
                    for images, labels, label_lengths in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        label_lengths = label_lengths.to(device)

                        logits = model(images)
                        input_lengths = torch.full(size=(images.size(0),), fill_value=logits.size(0), dtype=torch.long).to(device)

                        log_probs = F.log_softmax(logits, dim=2)
                        loss = criterion(log_probs, labels, input_lengths, label_lengths)

                        val_loss += loss.item()
                        val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            if CONFIG["model"] in CNN_MODELS:
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
            else:
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}")
        
        else:
            if CONFIG["model"] in CNN_MODELS:
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Accuracy: {acc:.2f}%")
            else:
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint at end of each epoch
        loss_data = (train_losses, val_losses, epochs)
        save_checkpoint(model, optimizer, epoch, loss_data, checkpoint_path)

    print("Training complete!")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save the learning curve
    loss_data = (train_losses, val_losses, epochs)
    save_learning_curve(loss_data, model_filename, plots_dir)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint deleted after successful training.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        CONFIG = json.load(f)

    if is_running_under_slurm():
        max_key_len = max(len(key) for key in CONFIG)
        for key, value in CONFIG.items():
            print(f"{key:<{max_key_len}} : {value}")

    train()
