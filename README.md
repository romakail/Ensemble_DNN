# Ensemble_DNN
In this repository I explore different methods of ensembling of DNN

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

The code in this repository implements Fast Geometric Ensembling (FGE) and gradient boosting Ensembling, with examples on the CIFAR-10 and CIFAR-100 datasets.

## Training the initial models

To run the ensembling procedure, you first need to train a network that will serve as the starting point of the ensemble. You can train it using the following command

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --device=<DEVICE> \
                 [--use_test]
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```TRANSFORM``` &mdash; type of data transformation [VGG/ResNet] (default: VGG)
* ```MODEL``` &mdash; DNN model name:
    - ConvFC
    - vgg16/vgg16_bn/vgg19/vgg19_bn 
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```DEVICE``` &mdash; GPU number

Use the `--use_test` flag if you want to use the test set instead of validation set (formed from the last 5000 training objects) to evaluate performance.

For example, use the following commands to train VGG16, PreResNet or Wide ResNet:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=vgg16_bn --epochs=200 --lr=0.05 --wd=5e-4 --use_test --transform=VGG --device=0
#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=150  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --device=0
#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --device=0
```

## Fast Geometric Ensembling (FGE)

In order to run FGE you need to pre-train the network to initialize the procedure. To do so follow the instructions in the previous section. Then, you can run FGE with the following command:

```bash
python3 fge.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --ckpt=<CKPT> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --ckpt=<CKPT> \
                 --lr_1=<LR1> \
                 --lr_2=<LR2> \
                 --cycle=<CYCLE> \
                 --device=<DEVICE> \
                 [--use_test]
```
Parameters:

* ```CKPT``` path to the checkpoint saved by `train.py`
* ```LR1, LR2``` the minimum and maximum learning rates in the cycle
* ```CYCLE``` cycle length in epochs (default:4)

For example, use the following commands to train VGG16 FGE ensemble:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=CIFAR100 --model=vgg16_bn --data_path=<PATH> --epochs=200 --cycle=10 --device=1 --use_test
#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=400 --cycle=10  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --device=0
#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --cycle=20 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --device=0
```

## Gradien boosting Ensembling

In order to run a gradient boosting ensemble you need to pre-train the network to initialize the procedure. To do so follow the instructions in the first section. Then, you can run GB ensembling with the following command:


```bash
python3 fge_gradboost.py --dir=<DIR> \
                         --dataset=<DATASET> \
                         --data_path=<PATH> \
                         --transform=<TRANSFORM> \
                         --model=<MODEL> \
                         --ckpt=<CKPT> \
                         --epochs=<EPOCHS> \
                         --cycle=<CYCLE> \
                         --lr_1=<LR1> \
                         --lr_2=<LR2> \
                         --boost_lr=<BOOST_LR> \
                         --scheduler=<SCHEDULER> \
                         --independent=<INDEP> \
                         --device=<DEVICE> \
                         [--use_test]
```

* ```CKPT``` path to the checkpoint saved by `train.py`
* ```LR1, LR2``` the minimum and maximum learning rates in the cycle
* ```EPOCHS``` the total number of epochs
* ```CYCLE``` number of epochs spent on one model (default:4)
* ```BOOST_LR``` lenght of boosting learning rate. Can be a number or 'auto'. If 'auto' learning rate is chosen as a solution of one-dimensional optimization problem. (default:auto)
* ```SCHEDULER``` type of learning rate scheduler, used to train a new model (cyclic/linear/slide)
* ```INDEP``` can be true or false. If False a new model for the ensemble is initialized as a copy of previous one. If True new models are initialized from the scratch

For example, use the following commands to train VGG16 gradient boosting ensemble:

```bash
#VGG16
python3 fge_gradboost.py --use_test --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --transform=VGG --model=vgg16_bn --ckpt=<CKPT> --cycle=50 --epochs=800 --lr_1=0.01 --lr_2=0.0001 --device=0 --boost_lr=auto --scheduler=slide --independent=False
```

# References
 This repo inherits a lot from this repo
 * FGE ensembling: [github.com/timgaripov/dnn-mode-connectivity/](https://github.com/timgaripov/dnn-mode-connectivity/)
 
 Provided model implementations were adapted from
 * VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
 * PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 * WideResNet: [github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)

## Other Relevant Papers

 * [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109) by Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger
 * [Loss Surfaces, Mode Connectivity and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026) by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
 * [Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757) by Stanislav Fort, Huiyi Hu, Balaji Lakshminarayanan