import numpy as np

import torch
import torchvision

def select_idxs(data, target, idxs=[0,1]):
    new_data   = []
    new_target = []
    for input, trg in zip(data, target):
        if trg in idxs: 
#             new_data  .append(input.unsqueeze(0))
            new_data.append  (np.expand_dims(input, axis=0))
            new_target.append(trg)
    return np.concatenate(new_data, axis=0), np.array(new_target)

class CIFAR2(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = select_idxs(self.data, self.targets)
        
CIFAR10  = torchvision.datasets.CIFAR10
CIFAR100 = torchvision.datasets.CIFAR100
MNIST    = torchvision.datasets.MNIST
