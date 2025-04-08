import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.loader import get_char_loaders, get_img_loaders
from models.cnn import SimpleCNN, PretrainedCNN
from models.rnn import CNNLSTMCTC
from config import CONFIG

CNN_MODELS = ['cnn_base', 'cnn_pretrained']
RNN_MODELS = ['rnn_base']

def train():
    print("========== TRAIN.PY START ==========")

    # Setup
    print("Getting device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if CONFIG["model"] in CNN_MODELS:
        # Data Loaders
        print("Loading data loaders...")
        train_loader, _, classes = get_char_loaders(
            data_path=CONFIG["data_path"],
            batch_size=CONFIG["batch_size"],
            val_split=CONFIG["val_split"],
            colour=CONFIG["use_colour"]
        )

        # Model
        print("Initializing model...")
        if CONFIG["model"] == 'cnn_base':
            in_channels = 3 if CONFIG["use_colour"] else 1
            model = SimpleCNN(num_classes=len(classes), in_channels=in_channels).to(device)
        
        elif CONFIG["model"] == 'cnn_pretrained':
            model = PretrainedCNN(num_classes=len(classes)).to(device)


        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

        # Optimizer and Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

        # Training loop
        print("Begin training model")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
            for images, labels in loop:
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

                loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

        print("Training complete!")

    elif CONFIG["model"] in RNN_MODELS:
        # Data Loaders
        print("Loading data loaders...")
        train_loader, _, vocab = get_img_loaders(
            data_path=CONFIG["data_path"],
            batch_size=CONFIG["batch_size"],
            val_split=CONFIG["val_split"],
            colour=CONFIG["use_colour"]
        )

        # Model
        print("Initializing model...")
        if CONFIG["model"] == 'rnn_base':
            in_channels = 3 if CONFIG["use_colour"] else 1
            model = CNNLSTMCTC(len(vocab), in_channels=in_channels).to(device)
        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

        # Optimizer and Loss
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

        # Training loop
        print("Begin training model")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            num_batches = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
            for images, labels, label_lengths in loop:
                images, labels = images.to(device), labels.to(device)
                label_lengths = label_lengths.to(device)

                logits = model(images)  # (T, B, C)
                input_lengths = torch.full(size=(images.size(0),), fill_value=logits.size(0), dtype=torch.long).to(device)

                log_probs = F.log_softmax(logits, dim=2)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Avg Loss: {avg_loss:.4f}")

    else:
        raise ValueError(f"Unknown model '{CONFIG['model']}'")


if __name__ == "__main__":
    train()