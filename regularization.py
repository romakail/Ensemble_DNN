import os
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

from torch.nn.functional import softmax

import utils

class TwoModelsMSE ():
    def __init__(self, orig_model, weight_decay):
        self.orig_model = deepcopy(orig_model)
        for p in self.orig_model.parameters():
            p = p.detach()
        self.wd = weight_decay
        print (self.wd)

    def reg (self, model):
        l2 = 0.0
        x_sum = 0.
        y_sum = 0.
        for (x, y) in zip(self.orig_model.parameters(), model.parameters()):
            x_sum += x.sum().item()
            y_sum += y.sum().item()
            l2 += torch.sum((x - y) ** 2)
        return - self.wd * l2



def linear_weigh_func(logits, labels, coef=1, normalize=False):
    assert logits.shape[0] == labels.shape[0]
    prediction = softmax(logits, dim=1)
    probas = prediction.take(
        (labels + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
    w = coef * (1 - probas)
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

def exponential_weight_func(logits, labels, coef=1, normalize=False):
    prediction = softmax(logits, dim=1)
    probas = prediction.take(
        (labels + torch.arange(0, prediction.numel(), prediction.shape[1])).cuda())
    w = torch.exp(coef * (1 - probas))
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

def adalast_weight_func(logits, labels, coef=1, normalize=False):
    true_logits = logits.take(
        (labels + torch.arange(0, logits.numel(), logits.shape[1])).cuda())
    true_logits = true_logits * (logits.shape[1] - 1) / logits.shape[1] - logits.mean(dim=1)
    correct = 2 * torch.eq(labels.cuda(), logits.argmax(dim=1)) - 1
    w = torch.exp(- coef * true_logits * correct)
    if normalize:
        w /= torch.sum(w)
        w *= w.shape[0]
    return w

class AdaBoost ():
    def __init__(self):
        self.logits = None
    def weight_func(self, logits, labels, coef=1, normalize=False):
        if self.logits is None:
            self.logits = logits
        else:
            self.logits += logits
        true_logits = self.logits.take(
            (labels + torch.arange(0, self.logits.numel(), self.logits.shape[1])).cuda())
        true_logits = true_logits * (self.logits.shape[1] - 1) / self.logits.shape[1] - logits.mean(dim=1)
        correct = 2 * torch.eq(labels.cuda(), self.logits.argmax(dim=1)) - 1
        w = torch.exp(- coef * true_logits)
        if normalize:
            w /= torch.sum(w)
            w *= w.shape[0]
        return w

def dataset_weights_generator(model, coef, func_type=None, normalize=False, batch_size=64):
    print ('Func_type : ', '[', func_type, ']', sep='')
    if func_type is None:
        return None
    elif func_type == 'Lin':
        weight_func = linear_weigh_func
    elif func_type == 'Exp':
        weight_func = exponential_weight_func
    elif func_type == 'AdaLast':
        weight_func = adalast_weight_func
    elif func_type == 'AdaBoost':
        adaboost_state_holder = AdaBoost()
        weight_func = adaboost_state_holder.weight_func
    # elif func_type == 'GradBoost':
    #     return None
    else:
        print ("I don't know this function, choose on of [Lin/Exp/AdaLast/AdaBoost]")
        return None

    def weights_generator(dataset):
        with torch.no_grad():
            input_batch = []

            logits = []
            labels = []

            for idx, (input, label) in enumerate(dataset):
                input_batch.append(input.unsqueeze(0))
                labels.append(label)
                if (idx+1) % batch_size == 0:
                    input_batch = torch.cat(input_batch, dim=0)
                    logits_batch = model(input_batch.cuda())
                    logits.append(logits_batch)
                    input_batch = []

            input_batch = torch.cat(input_batch, dim=0)
            logits_batch = model(input_batch.cuda())
            logits.append(logits_batch)

            logits = torch.cat(logits, dim=0)
            labels = torch.tensor(labels, dtype=torch.long)

            weights = weight_func(logits, labels, coef=coef, normalize=normalize)

            print ('Shape :', weights.shape, 'Weights_sum :', weights.sum().item())
            print ('Max :', weights.max().item(), 'Min :', weights.min().item())
        return weights

    return weights_generator

def dataset_logits_generator(model, transform, batch_size=64):
    def logits_generator(input_tensor):
        with torch.no_grad():
            input_batch = []
            logits = []
            for idx, input in enumerate(input_tensor):
                input = Image.fromarray(input)
                input = transform(input)
                input_batch.append(input.unsqueeze(0))
                if (idx+1) % batch_size == 0:
                    input_batch = torch.cat(input_batch, dim=0)
                    logits_batch = model(input_batch.cuda())
                    logits.append(logits_batch.cpu())
                    input_batch = []
    
            input_batch = torch.cat(input_batch, dim=0)
            logits_batch = model(input_batch.cuda())
            logits.append(logits_batch.cpu())

            logits = torch.cat(logits, dim=0)

#             print ('Shape :', logits.shape, 'Logits_mean :', logits.mean().item())
#             print ('Max :'  , logits.max().item(), 'Min :' , logits.min().item())
        return logits

    return logits_generator

def logits_info(logits, logits_sum=None):
    print ("+---------------")
    print ("|Mean :", logits.mean().item(),
           "Std", ((logits - logits.mean(dim=0, keepdim=True))**2).mean().sqrt().item())
    print ("|Max :", logits.max().item(),
           "Min :", logits.min().item())
    if logits_sum is not None:
        print ("|Sum Mean :", logits_sum.mean().item(),
               "Sum Std", ((logits_sum - logits_sum.mean(dim=0, keepdim=True))**2).mean().sqrt().item())
        print ("|Sum Max :", logits_sum.max().item(),
               "|Sum Min :", logits_sum.min().item())
    print ('+--------------')
    
def adjust_boost_lr(
    dataloader,
    model,
    criterion=torch.nn.CrossEntropyLoss(),
    device=torch.device('cuda'),
    lr_initialization = [0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000.],
    n_steps = 1000,
    step_size = 1000000.):
    
    model.eval().to(device)#.detach()
    with torch.no_grad():
        logits = []
        targets = []
        predicts = []
        for iter, (input, target, logit) in enumerate(dataloader):
            input  = input.to(device)
            predict = model(input)
            
            logits  .append(logit)
            targets .append(target)
            predicts.append(predict.cpu())
            
        logits   = torch.cat(logits, dim=0).detach()
        targets  = torch.cat(targets, dim=0).detach()
        predicts = torch.cat(predicts, dim=0).detach()
        
        lr_initialization = torch.tensor(lr_initialization)
        results = torch.zeros(lr_initialization.shape, dtype=torch.float)
        for iter, lr in enumerate(lr_initialization):
            results[iter] = criterion(logits + lr * predicts, targets)
        
    learning_rate = float(lr_initialization[torch.argmin(results)])
    learning_rate = torch.tensor([learning_rate], requires_grad=True)
    optim = torch.optim.SGD([learning_rate], lr=1., momentum=0.5)
    learning_rates = np.arange(1., 0., -1/n_steps, dtype=float) * float(learning_rate) * step_size
    
    for iter, loc_lr in enumerate(learning_rates):
        utils.adjust_learning_rate(optim, loc_lr)
        loss = criterion(logits + learning_rate * predicts, targets)
            
        loss.backward()
        optim.step()
        if iter%(100) == 99:
            print ('[', iter, '] lr :', learning_rate, 'grad :', learning_rate.grad)
        optim.zero_grad()
    
    return float(learning_rate.detach())
        
        
        
        