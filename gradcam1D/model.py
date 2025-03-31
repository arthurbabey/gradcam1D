import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DConvNet(nn.Module):
    def __init__(self):
        super(Simple1DConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)  # Assume binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc(x)
        return x