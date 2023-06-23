import torch.nn as nn
import torch.nn.functional as F
from torch.nn import InstanceNorm2d

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self, normalization='batch'):
        super(Net, self).__init__()

        if normalization == 'batch':
            Norm2d = nn.BatchNorm2d
        elif normalization == 'group':
            Norm2d = lambda num_features: nn.GroupNorm(2, num_features)
        elif normalization == 'layer':
            Norm2d = lambda num_features: nn.GroupNorm(1, num_features)
        else:
            raise ValueError("Invalid Normalization Type")

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            Norm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            Norm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # TRANSITION BLOCK 1
        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 28
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            Norm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 12
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            Norm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            Norm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 4

        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            Norm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        # OUTPUT BLOCK
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x = x1 + x2
        x = self.convblock2a(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.gap(x)  # Use the GAP layer here
        x = self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)