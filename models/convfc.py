import math
import torch.nn as nn

__all__ = [
    'ConvFC', 'ConvFCSimple'
]


class ConvFCBase(nn.Module):

    def __init__(self, num_classes):
        super(ConvFCBase, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(1152, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x

class ConvFCSimpleBase(nn.Module):

    def __init__(self, num_classes):
        super(ConvFCSimpleBase, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(392, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


class ConvFC:
    base = ConvFCBase
    kwargs = {}
    
class ConvFCSimple:
    base = ConvFCSimpleBase
    kwargs = {}
