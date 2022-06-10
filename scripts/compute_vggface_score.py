import argparse
import os
import sys
import torch
import torch.nn as nn
import math
import pickle
import numpy as np
import torch.nn.functional as F
import requests
from requests.adapters import HTTPAdapter


__all__ = ['ResNet', 'resnet50']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if not self.include_top:
            return x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def resnet50(**kwargs):
        """Constructs a ResNet-50 model.
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        return model


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def transform(img):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= self.mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

def download_weight_for_small_file(weight_file):
    '''
    Not working for large files
    '''
    weight_url = 'https://drive.google.com/uc?export=download' + \
        '&id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU'
    if not os.path.exists(weight_file):
        print('Downloading VGGFace resnet50_ft_weight.pkl (1/1)')
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=10))
        r = s.get(weight_url, allow_redirects=True)
        with open(weight_file, 'wb') as f:
            f.write(r.content)

def download_weight(weight_file):
    '''
    Reference:
        https://medium.com/@acpanjan/
            download-google-drive-files-using-wget-3c2c025a8b99
    '''
    if not os.path.exists(weight_file):
        cmd = "bash scripts/download_vggface_weights.sh"
        print(cmd)
        os.system(cmd)

def get_vggface_score(imgs):
    N_IDENTITY = 8631
    # include_top = True if args.cmd != 'extract' else False
    include_top = True
    weight_file = 'scripts/weights/resnet50_ft_weight.pkl'
    # mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    download_weight(weight_file)
    load_state_dict(model, weight_file)
    model = model.cuda()
    model.eval()
    mean_rgb = np.array([131.0912, 103.8827, 91.4953])
    # mean_rgb = torch.Tensor([131.0912, 103.8827, 91.4953]).cuda()
    preds = []
    for img in imgs:
        # img = img.permute(1, 2, 0) - mean_rgb
        img = img - mean_rgb
        img = torch.Tensor(img).cuda().permute(2, 0, 1)
        img = nn.UpsamplingBilinear2d(size=(224, 224))(img.view(1, 3, img.size(1), img.size(2)))
        pred = model(img)
        pred = F.softmax(pred, dim=1)
        preds.append(pred.data.cpu().numpy())
    preds = np.concatenate(preds, 0)
    # preds = model(imgs)

    np.random.shuffle(preds)
    splits = 5
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


def get_vggface_act(imgs):
    N_IDENTITY = 8631
    # include_top = True if args.cmd != 'extract' else False
    include_top = True
    weight_file = 'scripts/weights/resnet50_ft_weights.pkl'
    # mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    load_state_dict(model, weight_file)
    model = model.cuda()
    model.eval()
    mean_rgb = np.array([131.0912, 103.8827, 91.4953])
    # mean_rgb = torch.Tensor([131.0912, 103.8827, 91.4953]).cuda()
    preds = []
    for img in imgs:
        # img = img.permute(1, 2, 0) - mean_rgb
        img = img - mean_rgb
        img = torch.Tensor(img).cuda().permute(2, 0, 1)
        img = nn.UpsamplingBilinear2d(size=(224, 224))(img.view(1, 3, img.size(1), img.size(2)))
        pred = model(img)
        pred = F.softmax(pred, dim=1)
        preds.append(pred.data.cpu().numpy())
    preds = np.concatenate(preds, 0)
    return preds


def main():
    N_IDENTITY = 8631
    # include_top = True if args.cmd != 'extract' else False
    include_top = False
    weight_file = './weights/resnet50_ft_weights.pkl'
    model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    load_state_dict(model, weight_file)
    model = model.cuda()

if __name__ == '__main__':
    main()
