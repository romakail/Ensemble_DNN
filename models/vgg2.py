import torch
import torch.nn as nn

__all__ = ['VGG', 'vgg2', 'vgg2_bn', 'vgg5', 'vgg5_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]

cfgs = {
    2  : [64, 'M', 'M', 'M', 'M', 512, 'M'],
    5  : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):

    def __init__(self, num_classes, depth=16, batch_norm=False, init_weights=True):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs[depth], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
#             nn.Softmax(dim=1)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

class vgg2:
    base = VGG
    kwargs = {
        'depth': 2,
        'batch_norm': False,
        'init_weights': True,
    }
    
class vgg2_bn:
    base = VGG
    kwargs = {
        'depth': 2,
        'batch_norm': True,
        'init_weights': True,
    }

class vgg5:
    base = VGG
    kwargs = {
        'depth': 5,
        'batch_norm': False,
        'init_weights': True,
    }
    
class vgg5_bn:
    base = VGG
    kwargs = {
        'depth': 5,
        'batch_norm': True,
        'init_weights': True,
    }

class vgg16:
    base = VGG
    kwargs = {
        'depth': 16,
        'batch_norm': False,
        'init_weights': True,
    }
    
class vgg16_bn:
    base = VGG
    kwargs = {
        'depth': 16,
        'batch_norm': True,
        'init_weights': True,
    }
    
class vgg19:
    base = VGG
    kwargs = {
        'depth': 19,
        'batch_norm': False,
        'init_weights': True,
    }
    
class vgg19_bn:
    base = VGG
    kwargs = {
        'depth': 19,
        'batch_norm': True,
        'init_weights': True,
    }