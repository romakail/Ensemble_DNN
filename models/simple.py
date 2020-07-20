import torch.nn as nn

__all__ = [
    'TwoLayersNet',
]

class Flatten(nn.Module):
    
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)

class TwoLayersNetBase(nn.Module):

    def __init__(self, num_classes):
        super(TwoLayersNetBase, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
class TwoLayersNet:
    base = TwoLayersNetBase
    kwargs = {}