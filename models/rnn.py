import torch.nn as nn
from torchvision import models

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


class ResNetCTCModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetCTCModel, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        self.conv = nn.Conv2d(512, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # Force height=1, keep width

        self.rnn = nn.LSTM(256, 128, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)       # (B, 512, H, W)
        features = self.conv(features)    # (B, 256, H, W)
        features = self.pool(features)    # (B, 256, 1, W)
        features = features.squeeze(2)    # (B, 256, W)
        features = features.permute(0, 2, 1)  # (B, W, 256)

        rnn_out, _ = self.rnn(features)   # (B, W, 256)
        output = self.classifier(rnn_out) # (B, W, num_classes)

        return output.permute(1, 0, 2)    # (W, B, num_classes)