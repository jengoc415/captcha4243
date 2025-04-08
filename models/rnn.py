import torch.nn as nn

class CNNLSTMCTC(nn.Module):
    def __init__(self, vocab_size, in_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),  # -> (32, 50, 200)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (32, 25, 100)
            nn.Conv2d(32, 64, 3, padding=1), # -> (64, 25, 100)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (64, 12, 50)
        )
        self.lstm = nn.LSTM(input_size=64*12, hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(128 * 2, vocab_size + 1)  # +1 for blank

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # (W, B, C, H)
        x = x.view(w, b, c * h)  # (T, B, input_size)
        x, _ = self.lstm(x)      # (T, B, 256)
        x = self.fc(x)           # (T, B, vocab+1)
        return x
