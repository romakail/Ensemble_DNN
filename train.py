import argparse
import os
import random
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import data
import models
import utils

# from torchsummary import summary


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--device', type=int, default=0, metavar='N',
                    help='number of device to train on (default: 0)')

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: random)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=False)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True

if args.seed == 0:
    args.seed = random.randint(0, 1000000)   
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

train_len = 0
test_len = 0
for (x, _) in loaders['train']:
    train_len += x.shape[0]
for (x, _) in loaders['test']:
    test_len += x.shape[0]
print ('Train_len = ', train_len, 'test_len = ', test_len)

# print (dir(models))
architecture = getattr(models, args.model)


model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model.cuda()
# summary(model, (3, 32, 32))



def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


criterion = F.cross_entropy
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd if args.curve is None else 0.0
)


start_epoch = 1
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}
for epoch in range(start_epoch, args.epochs + 1):
    time_ep = time.time()

    lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    utils.adjust_learning_rate(optimizer, lr)

    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
    if args.curve is None or not has_bn:
        test_res = utils.test(loaders['test'], model, criterion, regularizer)

    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
#     for idx, module in enumerate(model.modules()):
#         if type(module) == torch.nn.modules.conv.Conv2d:
#             p = module.state_dict()['weight']
#             print ('[', idx, '] ', round(p.sum().item(), 3), end=' ', sep='')
#     print (' ')
    

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )
