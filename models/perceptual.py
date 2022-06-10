import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models
try:
    from .inception_resnet_v1 import InceptionResnetV1
except:
    from inception_resnet_v1 import InceptionResnetV1


class FaceNetLoss(nn.Module):
    def __init__(self, cos_loss_weight=0.0):
        super(FaceNetLoss, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.cos_loss_weight = cos_loss_weight
        if self.cos_loss_weight > 0.0:
            self.cos_emb_loss = torch.nn.CosineEmbeddingLoss()
            self.cos_target = torch.ones((1,), dtype=torch.int8).cuda()

    def forward(self, x, y):
        x_facenet = self.facenet.get_hidden_states(x)
        # we do not store gradients for y
        with torch.no_grad():
            y_facenet = self.facenet.get_hidden_states(y)
        loss = 0
        for i in range(len(x_facenet)):
            #print(x_facenet[i].shape)
            #print(y_facenet[i].shape)
            loss += self.weights[i] * \
                self.criterion(x_facenet[i], y_facenet[i])
        if self.cos_loss_weight > 0.0:
            loss += self.cos_loss_weight * \
                self.cos_emb_loss(
                    x_facenet[-1],
                    y_facenet[-1],
                    self.cos_target)
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        # we do not store gradients for y
        with torch.no_grad():
            y_vgg = self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, ignore_last=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        if ignore_last:
            self.slice5 = None
        else:
            self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if self.slice5 is not None:
            for x in range(21, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        if self.slice5 is None or h_relu4.size(3) < 3:
            return [h_relu1, h_relu2, h_relu3, h_relu4,]
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


if __name__ == '__main__':
    demo_imgs = torch.ones((4, 3, 224, 224))
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    facenet_out = facenet.get_hidden_states(demo_imgs)
    for i, feature in enumerate(facenet_out):
        print('FaceNet Feature Shape {}:'.format(i), feature.shape)

    vgg_19 = Vgg19()
    vgg_out = vgg_19(demo_imgs)
    for i, feature in enumerate(vgg_out):
        print('VGG Feature Shape {}:'.format(i), feature.shape)
