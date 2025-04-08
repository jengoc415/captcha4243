import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.loader import get_loaders
from models.cnn import SimpleCNN, PretrainedCNN
from config import CONFIG

def train():
    print("========== TRAIN.PY START ==========")

    # Setup
    print("Getting device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Loaders
    print("Loading data loaders...")
    train_loader, val_loader, classes = get_loaders(
        data_path=CONFIG["data_path"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        colour=CONFIG["use_colour"]
    )

    # Model
    print("Building model...")
    if CONFIG["use_pretrained"]:
        model = PretrainedCNN(num_classes=len(classes)).to(device)
    else:
        in_channels = 3 if CONFIG["use_colour"] else 1
        model = SimpleCNN(num_classes=len(classes), in_channels=in_channels).to(device)

    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Training loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {total_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")

    print("Training complete!")

if __name__ == "__main__":
    train()
