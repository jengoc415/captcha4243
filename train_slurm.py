import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys

from utils.loader import get_char_loaders, get_img_loaders
from models.cnn import SimpleCNN, PretrainedCNN
from models.rnn import CNNLSTMCTC
from config import CONFIG

CNN_MODELS = ['cnn_base', 'cnn_pretrained']
RNN_MODELS = ['rnn_base']

PRINT_EVERY = 50  # Print progress every N batches when running under SLURM

def is_running_under_slurm():
    return 'SLURM_JOB_ID' in os.environ

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch

def train():
    print("========== TRAIN.PY START ==========")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    model_filename = CONFIG["model"]
    if CONFIG["use_colour"]:
        model_filename += "_color"
    model_path = os.path.join(model_dir, model_filename + ".pt")
    checkpoint_path = os.path.join(model_dir, model_filename + "_checkpoint.pt")

    start_epoch = 0
    use_tqdm = not is_running_under_slurm()

    # Load model and data loaders
    if CONFIG["model"] in CNN_MODELS:
        print("Loading data loaders...")
        train_loader, _, classes = get_char_loaders(
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
        train_loader, _, vocab = get_img_loaders(
            data_path=CONFIG["train_path"],
            batch_size=CONFIG["batch_size"],
            val_split=CONFIG["val_split"],
            colour=CONFIG["use_colour"]
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
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

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
        if CONFIG["model"] in CNN_MODELS:
            acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}")

        # Save checkpoint at end of each epoch
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    print("Training complete!")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint deleted after successful training.")

    return model

if __name__ == "__main__":
    train()
