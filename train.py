import torch
import torch.nn as nn
import torch.optim as optim
from config import CONFIG
from utils.loader import get_loaders
from models.cnn import SimpleCNN

def get_model(name, num_classes):
    if name == "cnn":
        from models.cnn import SimpleCNN
        return SimpleCNN(num_classes)
    elif name == "rnn":
        raise NotImplementedError("RNN model coming soon.")
    else:
        raise ValueError(f"Unknown model: {name}")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, classes = get_loaders(
        data_path=CONFIG["data_path"],
        batch_size=CONFIG["batch_size"]
    )

    model = get_model(CONFIG["model"], num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train()
