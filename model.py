import torch
import torch.nn as nn

class BacteriaBranch(nn.Module):
    """CNN branch for bacterial DNA sequence (channels-first input)."""
    def __init__(self):
        super(BacteriaBranch, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=5)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=10, stride=5, bias=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        # x shape: (batch, 4, BACTERIUM_THRESHOLD)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        # Permute to (batch, length, channels) to mimic Keras channels-last flatten, then flatten
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        return x

class PhageBranch(nn.Module):
    """CNN branch for phage DNA sequence (channels-first input)."""
    def __init__(self):
        super(PhageBranch, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=30, stride=10, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=25, stride=10, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        # x shape: (batch, 4, PHAGE_THRESHOLD)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), -1)
        return x

class PerphectInteractionModel(nn.Module):
    """Dual-input CNN for phage-bacterium interaction prediction."""
    def __init__(self):
        super(PerphectInteractionModel, self).__init__()
        self.bacteria_branch = BacteriaBranch()
        self.phage_branch = PhageBranch()
        # Flattened feature lengths: 8928 (bacteria) + 6368 (phage) = 15296
        self.fc1 = nn.Linear(15296, 100, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 1, bias=True)
    def forward(self, bacteria_input, phage_input):
        # Pass each input through its branch
        bact_features = self.bacteria_branch(bacteria_input)
        phage_features = self.phage_branch(phage_input)
        # Concatenate features (batch, 15296)
        combined = torch.cat([bact_features, phage_features], dim=1)
        # Fully-connected layers and sigmoid output
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        out = torch.sigmoid(self.fc2(x))
        return out