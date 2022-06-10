import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bilinear import crop_bbox_batch, uncrop_bbox
from utils.box_utils import box_union, box_in_region
from .layers import GlobalAvgPool, Flatten, get_activation, build_cnn
import models


class PatchDiscriminator(nn.Module):
    def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
                 padding='same', pooling='avg', input_size=(128, 128),
                 layout_dim=0):
        super(PatchDiscriminator, self).__init__()
        input_dim = 3 + layout_dim
        arch = 'I%d,%s' % (input_dim, arch)
        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        self.cnn, output_dim = build_cnn(**cnn_kwargs)
        self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

    def forward(self, x):
        return self.classifier(self.cnn(x))


class AcDiscriminator(nn.Module):
    def __init__(self, arch, normalization='none', activation='relu',
                 padding='same', pooling='avg', num_id=None):
        super(AcDiscriminator, self).__init__()
        # self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        self.num_id = num_id

        self.real_classifier = nn.Linear(1024, 1)
        self.id_classifier = nn.Linear(1024, num_id)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]

        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        id_scores = self.id_classifier(vecs)
        ac_loss = F.cross_entropy(id_scores, y)
        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.discriminator = AcDiscriminator(arch, normalization,
                                             activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes, obj_to_img,
                object_crops=None, **kwargs):
        if object_crops is None:
            object_crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, ac_loss = self.discriminator(object_crops, objs)
        return real_scores, ac_loss, object_crops


# Conditional Discriminator
class CondPatchDiscriminator(nn.Module):
    def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
                 padding='same', pooling='avg', input_size=(128, 128),
                 cond_dim=0):
        super(CondPatchDiscriminator, self).__init__()
        input_dim = 3
        arch = 'I%d,%s' % (input_dim, arch)
        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        self.cnn, output_dim = build_cnn(**cnn_kwargs)
        self.joint_conv = nn.Sequential(
            nn.Conv2d(output_dim+cond_dim, output_dim, \
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

    def forward(self, x, cond_vecs):
        if len(cond_vecs) == 2:
            cond_vecs = cond_vecs.view(-1, cond_vecs.size(1), 1, 1)
        x = self.cnn(x)
        cond_vecs = cond_vecs.expand(-1, cond_vecs.size(1), x.size(2), x.size(3))
        x = torch.cat([x, cond_vecs], dim=1)
        x = self.joint_conv(x)
        return self.classifier(x)
