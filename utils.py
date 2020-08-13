import numpy as np
import os
import torch
import torch.nn.functional as F

def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


# def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
#     def schedule(iter):
#         t = ((epoch % cycle) + iter) / cycle
#         if t < 0.5:
#             return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
#         else:
#             return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
#     return schedule

def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * 2.0 * t + alpha_2 * (1.0 - 2.0 * t)
        else:
            return alpha_1 * (2.0 - 2.0 * t) + alpha_2 * (2.0 * t - 1.0)
    return schedule

def linear_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        return alpha_1 + (alpha_2 - alpha_1) * t
    return schedule

def slide_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t <= 0.5:
            return alpha_1
        elif t <= 0.9:
            return alpha_1 + (t - 0.5) / 0.4 * (alpha_2 - alpha_1)
        else:
            return alpha_2
    return schedule

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(device=None, non_blocking=False)
        target = target.cuda(device=None, non_blocking=False)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_weighted(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(device=None, non_blocking=False)
        labels = target['label'].cuda(device=None, non_blocking=False)
        output = model(input)
        loss = criterion(output, labels)
        loss = torch.mean(loss * target['weight'].cuda(device=None, non_blocking=False))

        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(labels.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_gb (train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None, gb_version='classic', boost_lr=1.):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, labels, logits) in enumerate(train_loader):
#         print ("Input  :", input.shape)
#         print ("Labels :", labels.shape)
#         print ("Logits :", logits.shape)
        if isinstance(labels, dict):
            weights = labels['weights']
            labels  = labels['labels']
        else:
            weights = torch.ones(
                labels.shape,
                dtype=torch.float,
                device=torch.device('cuda'),
                requires_grad=False)
        
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input  = input .cuda(device=None, non_blocking=False)
        logits = logits.cuda(device=None, non_blocking=False).detach()
        labels = labels.cuda(device=None, non_blocking=False)
        output = model(input)
        
#         print("Logits :", type(logits), logits.shape, logits.dtype)
#         print("Labels :", type(labels), labels.shape, labels.dtype)
#         print("Output :", type(output), output.shape, output.dtype)
#         raise AssertionError('STOP')
        
        if   gb_version == 'simple':
            loss = criterion(logits + boost_lr * output, labels)
        elif gb_version == 'classic':
            antigrad = one_hot(labels, logits.shape[1]) - F.softmax(logits, dim=1)
            loss = criterion(output, antigrad).mean(dim=1)

        loss = torch.mean(loss * weights)

        if regularizer is not None:
            loss += boost_lr * regularizer(model)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = (logits + boost_lr * output).data.argmax(1, keepdim=True)
        correct += pred.eq(labels.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(device=None, non_blocking=False)
        if isinstance (target, dict):
            target = target['label'].cuda(device=None, non_blocking=False)
        else:
            target = target.cuda(device=None, non_blocking=False)

        output = model(input, **kwargs)
        nll = criterion(output, target).mean()
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def test_gb(test_loader, model, criterion, regularizer=None, gb_version='classic', boost_lr=1., **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target, logits in test_loader:
        
        input  = input .cuda(device=None, non_blocking=False)
        target = target.cuda(device=None, non_blocking=False)
        logits = logits.cuda(device=None, non_blocking=False)
        
        output = model(input, **kwargs)
        if   gb_version == 'simple':
            nll = criterion(logits + boost_lr * output, labels).mean()
            loss = nll.clone()
        elif gb_version == 'classic':
            antigrad = one_hot(target, logits.shape[1]) - F.softmax(logits, dim=1)
            nll = criterion(output, antigrad).mean(dim=1).mean()
            loss = torch.nn.CrossEntropyLoss()(logits + boost_lr * output, target)

        nll_sum  += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
#         pred = output.data.argmax(1, keepdim=True)
#         correct += pred.eq(target.data.view_as(pred)).sum().item()
        pred = (logits + boost_lr * output).data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll'     : nll_sum / len(test_loader.dataset),
        'loss'    : loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(device=None, non_blocking=False)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def logits(test_loader, model, **kwargs):
    preds = []
    targets = []
    print ('TestLoader :', type(test_loader))
    for data in test_loader:
        input = data[0]
        target = data[1]
        input = input.cuda(device=None, non_blocking=False)
        output = model(input, **kwargs)
        preds.append(output.cpu().data)
        targets.append(target)
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm)

def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(device=None, non_blocking=False)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
    
def one_hot(batch ,depth):
    ones = torch.eye(depth, dtype=batch.dtype, device=batch.device)
    return ones.index_select(0,batch)
