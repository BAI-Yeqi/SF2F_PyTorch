import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import get_normalization_2d
from .layers import get_activation


"""
Cascaded refinement network architecture, as described in:
Qifeng Chen and Vladlen Koltun,
"Photographic Image Synthesis with Cascaded Refinement Networks",
ICCV 2017
"""


class RefinementModule(nn.Module):
    def __init__(self, layout_dim, input_dim, output_dim,
                 normalization='instance', activation='leakyrelu'):
        super(RefinementModule, self).__init__()

        layers = []
        layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                                kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(output_dim, output_dim,
                                kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, layout, feats):
        _, _, HH, WW = layout.size()
        _, _, H, W = feats.size()
        assert HH >= H
        if HH > H:
            factor = round(HH // H)
            assert HH % factor == 0
            assert WW % factor == 0 and WW // factor == W
            layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        net_input = torch.cat([layout, feats], dim=1)
        out = self.net(net_input)
        return out


class RefinementNetwork(nn.Module):
    def __init__(self, dims, normalization='instance',
                 activation='leakyrelu',
                 use_tanh=False,
                 use_deconv=False,
                 multi_resolution=False,):
        super(RefinementNetwork, self).__init__()
        layout_dim = dims[0]
        self.refinement_modules = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        self.multi_resolution = multi_resolution
        for i in range(1, len(dims)):
            input_dim = 1 if i == 1 else (dims[i] if use_deconv else dims[i-1])
            output_dim = dims[i]
            mod = RefinementModule(layout_dim, input_dim, output_dim,
                                   normalization=normalization, activation=activation)
            self.refinement_modules.append(mod)
            if use_deconv:
                if i == 1:
                    mod = nn.Sequential()
                else:
                    mod = nn.Sequential(
                        nn.ConvTranspose2d(dims[i-1], dims[i],
                            kernel_size=2, stride=2,),
                        get_normalization_2d(dims[i], normalization),
                        get_activation(activation)
                    )
                self.upsample_modules.append(mod)
        output_conv_layers = None
        if not multi_resolution:
            output_conv_layers = [
                nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
                get_activation(activation),
                nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0),
            ]
            if use_tanh:
                ## added according to Pix2Pix-HD
                output_conv_layers.append(nn.Tanh(),)
            nn.init.kaiming_normal_(output_conv_layers[0].weight)
            nn.init.kaiming_normal_(output_conv_layers[2].weight)
        else:
            output_conv_layers_low = [
                nn.Conv2d(dims[-2], dims[-2], kernel_size=3, padding=1),
                get_activation(activation),
                nn.Conv2d(dims[-2], 3, kernel_size=1, padding=0),
            ]
            output_conv_layers_high = [
                nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
                get_activation(activation),
                nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0),
            ]
            if use_tanh:
                ## added according to Pix2Pix-HD
                output_conv_layers_low.append(nn.Tanh(),)
                output_conv_layers_high.append(nn.Tanh(),)

        if output_conv_layers:
            self.output_conv = nn.Sequential(*output_conv_layers)
        else:
            assert output_conv_layers_low
            assert output_conv_layers_high
            self.output_conv_low = nn.Sequential(*output_conv_layers_low)
            self.output_conv_high = nn.Sequential(*output_conv_layers_high)

    def forward(self, layout):
        """
        Output will have same size as layout
        """
        # H, W = self.output_size
        N, _, H, W = layout.size()
        self.layout = layout

        # Figure out size of input
        input_H, input_W = H, W
        for _ in range(len(self.refinement_modules)):
            input_H //= 2
            input_W //= 2

        assert input_H != 0
        assert input_W != 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)
        if not self.multi_resolution:
            for i, mod in enumerate(self.refinement_modules):
                if i == 0 or len(self.upsample_modules) == 0:
                    feats = F.interpolate(feats, scale_factor=2, mode='nearest')
                else:
                    feats = self.upsample_modules[i](feats)
                feats = mod(layout, feats)
            out = self.output_conv(feats)
            return (out, feats)
        elif self.multi_resolution:
            for i, mod in enumerate(self.refinement_modules):
                if i == 0 or len(self.upsample_modules) == 0:
                    feats = F.interpolate(feats, scale_factor=2, mode='nearest')
                else:
                    feats = self.upsample_modules[i](feats)
                feats = mod(layout, feats)
                if i == len(self.refinement_modules) - 2:
                    out_low = self.output_conv_low(feats)
            out_high = self.output_conv_high(feats)
            return (out_high, out_low)
