import sys
import math as m
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
from scipy.stats import wasserstein_distance

epsilon = 1e-45

def predictions_correlation(prediction1, prediction2):
    assert len(prediction1.shape) == 2
    assert prediction1.shape == prediction2.shape

    correlation = (
        ( (prediction1 - prediction1.mean(dim=0, keepdim=True))
        * (prediction2 - prediction2.mean(dim=0, keepdim=True))).mean(dim=0)
        / prediction1.var(dim=0).sqrt()
        / prediction2.var(dim=0).sqrt())
    return correlation.cpu().numpy()

def correlation_over_classes (model1, model2, dataloader, device=torch.device('cpu')):
    model1.eval().to(device)
    model2.eval().to(device)

    prediction1 = []
    prediction2 = []

    for input, _ in dataloader:
        prediction1.append(model1(input.detach().to(device)).detach().cpu())
        prediction2.append(model2(input.detach().to(device)).detach().cpu())

    prediction1 = torch.cat(prediction1, dim=0)
    prediction2 = torch.cat(prediction2, dim=0)

    return predictions_correlation(prediction1, prediction2)

def correlation_Nmodels (model_list, dataloader, device=torch.device('cpu'), one_hot=False):

    predictions = []
    for iter, model in tqdm(enumerate(model_list)):
        model.eval().to(device)
        predictions.append([])
        for input, _ in dataloader:
            predictions[iter].append(model(input.detach().to(device)).detach().cpu())
        predictions[iter] = torch.cat(predictions[iter], dim=0)

    if one_hot:
        for idx, prediction in enumerate(predictions):
            predictions[idx] = F.one_hot(prediction.argmax(dim=1)).float()

    n_model_list = len(model_list)
    cor_matrix = np.zeros([n_model_list, n_model_list], dtype=float)
    for i in range(n_model_list):
        for j in range(i, n_model_list):
            cor_matrix[i, j] = cor_matrix[j, i] = (
                predictions_correlation(predictions[i], predictions[j]).mean())
    return cor_matrix

def cross_entropy (pred, target):
    assert (pred.shape == target.shape)
    print ('1',  pred[100] + epsilon)
    print ('2', (pred[100] + epsilon).log())
#     print ('2', (target * pred.log()).sum())
#     print ('3', - (target * pred.log()).sum(dim=1).mean().item())
    return - (target * (pred + epsilon).log()).sum(dim=1).mean().item()

def cross_entropy_2models (pred_model, ground_truth_model, dataloader, device=torch.device('cpu')):
    pred_model.eval().to(device)
    ground_truth_model.eval().to(device)
    
    pred   = []
    target = []
    
    for input, _ in dataloader:
        pred  .append(pred_model        (input.detach().to(device)).detach().cpu())
        target.append(ground_truth_model(input.detach().to(device)).detach().cpu())
    
    pred   = torch.cat(pred  , dim=0)
    target = torch.cat(target, dim=0)
    
    return cross_entropy(pred, target)

def cross_entropy_Nmodels (models, dataloader, device=torch.device('cpu'), mean=False):
    
    predictions = []
    for iter, model in tqdm(enumerate(models)):
        model.eval().to(device)
        predictions.append([])
        for input, _ in dataloader:
            predictions[iter].append(model(input.detach().to(device)).detach().cpu())
        predictions[iter] = torch.cat(predictions[iter], dim=0)
    
    n_model_list = len(models)
    CE_matrix = np.zeros([n_model_list, n_model_list], dtype=float)
    
    if mean:
        for i in range(n_model_list):
            for j in range(n_model_list):
                CE_matrix[i, j] = (
                    cross_entropy(predictions[i], predictions[j]) +
                    cross_entropy(predictions[j], predictions[i])) / 2
    else:
        for i in range(n_model_list):
            for j in range(n_model_list):
                CE_matrix[i, j] = cross_entropy(predictions[i], predictions[j])
    return CE_matrix

def layers_diff (model1, model2, device=torch.device('cpu')):
    model1.to(device)
    model2.to(device)
    
    with torch.no_grad():
        table = []
        for idx, (module1, module2) in enumerate(zip(model1.modules(), model2.modules())):
            if type(module1) == type(module2) == torch.nn.modules.conv.Conv2d:
                for key in module1.state_dict().keys():
                    p1 = module1.state_dict()[key]
                    p2 = module2.state_dict()[key]
                    diff = p1 - p2
                    numel = diff.numel()
                    L1 = diff.abs().mean().item()
                    L2 = m.sqrt((diff ** 2).mean().item())
                    cos = ((p1 * p2).sum() / (p1**2).sum().sqrt() / (p2**2).sum().sqrt()).item()
                    was = wasserstein_distance(p1.view(-1).numpy(), p2.view(-1).numpy())
                    table.append({
                        'Num'      : idx,
                        'Layer'    : key,
                        'N_params' : numel,
                        'L1'       : round(L1, 3),
                        'L2'       : round(L2, 3),
                        'Cos'      : round(cos, 3),
                        'Was'      : round(was, 3)})
    return table
            
    
    
