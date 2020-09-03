import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

import datasets

class Transforms:
    
    class MNIST:
        
        class VGG:
            
            train = transforms.Compose([
                transforms.Resize(size=[32, 32]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            
            test = transforms.Compose([
                transforms.Resize(size=[32, 32]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
    
    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR2   = CIFAR10
    CIFAR100 = CIFAR10

class GradBoostDataset (Dataset):
    def __init__(self, inputs_tensor, targets_tensor, logits_tensor, transform=None):
        self.inputs  = inputs_tensor
        self.logits  = logits_tensor
        self.targets = targets_tensor
        self.transform = transform
        assert len(inputs_tensor) == len(logits_tensor)
        assert len(logits_tensor) == len(targets_tensor)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
#         print ("Inputs :", type(self.inputs[idx]))
#         print ("Target :", type(self.targets[idx]))
#         print ("Logits :", type(self.logits[idx]))
        
        img, target, logit = self.inputs[idx], self.targets[idx], self.logits[idx]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target, logit
    
    def update_logits(self, boost_lr=1.0, logits_generator=None):
#         print ('I am updating logits')
        if logits_generator is not None:
            self.logits += boost_lr * logits_generator(self.inputs)
#         print ('Shape :', self.logits.shape, 'Logits_mean :', self.logits.mean().item())
#         print ('Max :'  , self.logits.max().item(), 'Min :' , self.logits.min().item())

def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, weights_generator=None):
#     ds = getattr(torchvision.datasets, dataset)
    ds = getattr(datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True, download=True, transform=transform.train)
    
    n_classes = max(train_set.targets) + 1
    if weights_generator is not None:
        weights = weights_generator (train_set)
        train_set.targets = [{'label': label, 'weight': weight.item()} for label, weight in zip(train_set.targets, weights)]
    
    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
#         print (vars(train_set).keys())
#         print (train_set.data.shape)
#         print (len(train_set.targets))
        
        train_set.data = train_set.data[:-5000]
        train_set.targets = train_set.targets[:-5000]

        test_set = ds(path, train=True, download=True, transform=transform.test)
        test_set.train = False
        test_set.data = test_set.data[-5000:]
        test_set.targets = test_set.targets[-5000:]
        
    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, int(n_classes)

def loaders_gb(dataset, path, batch_size, num_workers, transform_name, use_test=False, shuffle_train=True, logits_generator=None):
#     ds = getattr(torchvision.datasets, dataset)
    ds = getattr(datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    train_set = ds(path, train=True , download=True, transform=transform.train)
    
    n_classes = max(train_set.targets) + 1
        
    logits_train = logits_generator(train_set.data).cpu().detach()
    print ('Initial logits :')
    print ('Shape :', logits_train.shape, 'Logits_mean :', logits_train.mean().item())
    print ('Max :'  , logits_train.max().item(), 'Min :' , logits_train.min().item())

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        test_set  = ds(path, train=False, download=True, transform=transform.test)
        logits_test = logits_generator(test_set.data).cpu().detach()
        
        train_set = GradBoostDataset(
            train_set.data, 
            train_set.targets,
            logits_train,
            transform=transform.train)
        test_set = GradBoostDataset(
            test_set.data,
            test_set.targets,
            logits_test,
            transform=transform.test)
    else:
        print("Using train (45000) + validation (5000)")
        train_set = GradBoostDataset(
            train_set.data[:-5000],
            train_set.target[:-5000],
            logits_train[:-5000],
            transform=transform.train)
        test_set = GradBoostDataset(
            train_set.data[-5000:],
            train_set.target[-5000:],
            logits_train[-5000:],
            transform=transform.test)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, int(n_classes)
