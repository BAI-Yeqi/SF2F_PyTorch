'''
Implementations of different Face Decoders
'''


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import functools
import numpy as np
from collections import OrderedDict
try:
    from .attention import AttnBlock
except:
    from attention import AttnBlock
try:
    from .layers import get_activation, get_normalization_2d
except:
    from layers import get_activation, get_normalization_2d
try:
    from .networks import upBlock, GLU
except:
    from networks import upBlock, GLU


class DeConv2DBLK(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size,
                 stride,
                 padding,
                 activation='relu',
                 normalization='none'):
        super(DeConv2DBLK, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(
                input_channel,
                output_channel,
                kernel_size,
                stride,
                padding,
                bias=True))
        self.layers.append(get_activation(activation))
        self.layers.append(
            get_normalization_2d(
                output_channel,
                normalization))
        self.layers = [layer for layer in self.layers if \
            layer is not None]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ExtraFeatureBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 normalization='none'):
        super(ExtraFeatureBlock, self).__init__()
        self.projection = nn.Linear(dim_in, dim_hidden, bias=True)
        self.norm_layer = get_normalization_2d(
            dim_hidden // 16, normalization)

    def forward(self, embedding, cat_tensor):
        '''
        Transform the embedding, then concat it to the hidden tensor map
        '''
        # Projected Embedding
        #print('embedding:', embedding.shape)
        N, D_in, _, _ = embedding.shape
        proj_emb = self.projection(embedding.view(N, -1))
        N, D_hid = proj_emb.shape
        feat_map = proj_emb.view(N, -1, 4, 4)
        if self.norm_layer is not None:
            feat_map = self.norm_layer(feat_map)
        N, C, H, W = cat_tensor.shape
        feat_map = F.interpolate(feat_map, (H, W))
        map_out = torch.cat([cat_tensor, feat_map], 1)

        return map_out


class V2FDecoder(nn.Module):
    def __init__(self,
                 input_channel,
                 channels,
                 output_channel,
                 normalization='none'):
        super(V2FDecoder, self).__init__()
        output_size = 2 ** (len(channels) + 1)
        print('V2FDecoder: According to your channels list, ' + \
            'the output will be generated with shape ({}, {})'.format(
                output_size, output_size))
        #torch.nn.ConvTranspose2d(
        #in_channels, out_channels, kernel_size, stride=1, padding=0,
        #output_padding=0, groups=1, bias=True,
        #dilation=1, padding_mode='zeros')
        self.layers = []
        # First Layer
        layer_id = 0
        self.layers.append(
            DeConv2DBLK(input_channel,
                        channels[0],
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        activation='relu',
                        normalization=normalization))
        # Hidden Layers
        for layer_id in range(1, len(channels)):
            self.layers.append(
                DeConv2DBLK(channels[layer_id - 1],
                            channels[layer_id],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            activation='relu',
                            normalization=normalization))
        # Output Layer
        layer_id = layer_id + 1
        self.layers.append(
            DeConv2DBLK(channels[layer_id - 1],
                        output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation='none',
                        normalization='none'))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class MRDecoder(nn.Module):
    def __init__(self,
                 input_channel,
                 base_block_channels,
                 mid_block_channel=32,
                 high_block_channel=None,
                 output_channel=3,
                 normalization='none',
                 use_up_block=False,
                 extra_feature_map=False):
        super(MRDecoder, self).__init__()
        output_sizes = []
        output_sizes.append(
            2 ** (len(base_block_channels) + 1))
        output_sizes.append(
            2 ** (len(base_block_channels) + 2))
        if high_block_channel is not None:
            self.build_high_img = True
            output_sizes.append(
                2 ** (len(base_block_channels) + 3))
        else:
            self.build_high_img = False
        print('MRDecoder: According to your channels list:')
        for i, output_size in enumerate(output_sizes):
            print('\tOutput {} will be generated with shape ({}, {})'.format(
                i, output_size, output_size))

        self.input_channel = input_channel
        self.base_block_channels = base_block_channels
        self.mid_block_channel = mid_block_channel
        self.high_block_channel = high_block_channel
        self.output_channel = output_channel
        self.normalization = normalization
        self.use_up_block = use_up_block
        self.extra_feature_map = extra_feature_map

        # Output Layer
        self.build_base_block()
        self.build_low_block()
        self.build_mid_block()
        if self.build_high_img:
            self.build_high_block()

    def build_base_block(self):
        #torch.nn.ConvTranspose2d(
        #in_channels, out_channels, kernel_size, stride=1, padding=0,
        #output_padding=0, groups=1, bias=True,
        #dilation=1, padding_mode='zeros')
        self.base_block_layers = []
        # First Layer
        layer_id = 0
        self.base_block_layers.append(
            DeConv2DBLK(self.input_channel,
                        self.base_block_channels[0],
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        activation='relu',
                        normalization=self.normalization))
        # Hidden Layers
        for layer_id in range(1, len(self.base_block_channels)):
            self.base_block_layers.append(
                DeConv2DBLK(self.base_block_channels[layer_id - 1],
                            self.base_block_channels[layer_id],
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            activation='relu',
                            normalization=self.normalization))
        self.base_block = nn.Sequential(*self.base_block_layers)

    def build_low_block(self):
        self.low_output_layer = \
            DeConv2DBLK(self.base_block_channels[-1],
                        self.output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation='none',
                        normalization='none')

    def build_mid_block(self):
        extra_channel = 0
        if self.extra_feature_map:
            # Mid-resolution ExtraFeatureBlock
            self.mid_ef_block = ExtraFeatureBlock(
                dim_in=self.input_channel,
                dim_hidden=self.input_channel,
                normalization=self.normalization)
            extra_channel = self.input_channel // 16
        if self.use_up_block:
            self.mid_hidden_layer = \
                upBlock(self.base_block_channels[-1] + extra_channel,
                        self.mid_block_channel)
        else:
            self.mid_hidden_layer = \
                DeConv2DBLK(self.base_block_channels[-1] + extra_channel,
                            self.mid_block_channel,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            activation='relu',
                            normalization=self.normalization)
        self.mid_output_layer = \
            DeConv2DBLK(self.mid_block_channel,
                        self.output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation='none',
                        normalization='none')

    def build_high_block(self):
        extra_channel = 0
        if self.extra_feature_map:
            # High-resolution ExtraFeatureBlock
            self.high_ef_block = ExtraFeatureBlock(
                dim_in=self.input_channel,
                dim_hidden=self.input_channel,
                normalization=self.normalization)
            extra_channel = self.input_channel // 16
        if self.use_up_block:
            self.high_hidden_layer = \
                upBlock(self.mid_block_channel + extra_channel,
                        self.high_block_channel)
        else:
            self.high_hidden_layer = \
                DeConv2DBLK(self.mid_block_channel + extra_channel,
                            self.high_block_channel,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            activation='relu',
                            normalization=self.normalization)
        self.high_output_layer = \
            DeConv2DBLK(self.high_block_channel,
                        self.output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation='none',
                        normalization='none')

    def forward(self, x):
        low_code = self.base_block(x)
        low_imgs = self.low_output_layer(low_code)
        #print('low_code:', low_code.shape)
        if self.extra_feature_map:
            low_code = self.mid_ef_block(x, low_code)
            #print('low_code:', low_code.shape)
        mid_code = self.mid_hidden_layer(low_code)
        mid_imgs = self.mid_output_layer(mid_code)
        if self.build_high_img:
            #print('mid_code:', mid_code.shape)
            if self.extra_feature_map:
                mid_code = self.high_ef_block(x, mid_code)
                #print('mid_code:', mid_code.shape)
            high_code = self.high_hidden_layer(mid_code)
            high_imgs = self.high_output_layer(high_code)
            return (low_imgs, mid_imgs, high_imgs)
        else:
            return (low_imgs, mid_imgs)


class AttnV2FDecoder(nn.Module):
    def __init__(self,
                 input_channel,
                 cnn_channels,
                 attn_channels,
                 output_channel,
                 seq_in_dim,
                 normalization='none'):
        super(AttnV2FDecoder, self).__init__()
        output_size = 2 ** (len(cnn_channels) + 1)
        print('V2FDecoder: According to your channels list, ' + \
            'the output will be generated with shape ({}, {})'.format(
                output_size, output_size))
        #torch.nn.ConvTranspose2d(
        #in_channels, out_channels, kernel_size, stride=1, padding=0,
        #output_padding=0, groups=1, bias=True,
        #dilation=1, padding_mode='zeros')
        self.layers = []
        self.layer_names = []
        # First Layer
        layer_id = 0
        self.layers.append(
            DeConv2DBLK(input_channel,
                        cnn_channels[0],
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        activation='relu',
                        normalization=normalization))
        self.layer_names.append('DeConv2DBLK{}'.format(layer_id))
        # Hidden Layers
        for layer_id in range(1, len(cnn_channels)):
            self.layers.append(
                DeConv2DBLK(
                    cnn_channels[layer_id - 1]+attn_channels[layer_id - 1],
                    cnn_channels[layer_id],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation='relu',
                    normalization=normalization))
            self.layer_names.append('DeConv2DBLK{}'.format(layer_id))
            attn_channel = attn_channels[layer_id]
            if attn_channel > 0:
                self.layers.append(
                    AttnBlock(attn_dim_in=seq_in_dim,
                              attn_dim_out=attn_channel,
                              attn_num_query=16,
                              normalization=normalization))
                self.layer_names.append('AttnBlock{}'.format(layer_id))
        # Output Layer
        layer_id = layer_id + 1
        self.layers.append(
            DeConv2DBLK(cnn_channels[layer_id - 1]+attn_channels[layer_id - 1],
                        output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation='none',
                        normalization='none'))
        self.layer_names.append('DeConv2DBLK{}'.format(layer_id))

        for layer_id, layer in enumerate(self.layers):
            self.add_module(self.layer_names[layer_id], layer)

    def forward(self, x, seq_feat):
        for layer, name in zip(self.layers, self.layer_names):
            if 'AttnBlock' in name:
                x = layer(x, seq_feat)
            else:
                x = layer(x)
            #print(x.shape)
        return x

    def return_attn_weights(self):
        for layer, name in zip(self.layers, self.layer_names):
            if 'AttnBlock' in name:
                return layer.return_attn_weights()


class FaceGanDecoder(nn.Module):
    def __init__(self,
                 image_size=(64, 64),
                 normalization='batch',
                 activation='leakyrelu-0.2',
                 mlp_normalization='none',
                 noise_dim=128,
                 **kwargs):

        super(FaceGanDecoder, self).__init__()
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.dim = noise_dim
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.dim * 4 * 4 * 2),
            GLU())

        up_num, up_layers = [], []
        num_in, num_out = 1, 1
        up_num = range(1, int(np.log2(image_size[0]))-1)

        for i, num in enumerate(up_num):
            if i == 0:
                num_in = num
            num_out = 2**num
            up_layers.append(upBlock(self.dim//num_in, self.dim//num_out))
            num_in = num_out

        self.up_layers = nn.Sequential(*up_layers)

        gnet_layers = [
            nn.Conv2d(self.dim//num_out, self.dim//num_out, kernel_size=3, padding=1),
            get_activation('leakyrelu'),
            nn.Conv2d(self.dim//num_out, 3, kernel_size=1, padding=0),
        ]
        self.gnet = nn.Sequential(*gnet_layers)

    def forward(self, x):
        """
        Required Inputs:
        - placeholder: This is a placeholder.

        Optional Inputs:
        - placeholder: This is a placeholder.
        """

        vecs = self.fc(x.view(-1, 512))
        vecs = vecs.view(x.size()[0], -1, 4, 4)

        img_code = self.up_layers(vecs)
        img_pred = self.gnet(img_code)

        return img_pred


class FaceGanDecoder_v2(nn.Module):
    def __init__(self,
                 image_size=(64, 64),
                 normalization='batch',
                 activation='leakyrelu-0.2',
                 mlp_normalization='none',
                 noise_dim=128,
                 **kwargs):

        super(FaceGanDecoder_v2, self).__init__()
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.dim = noise_dim
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.dim * 4 * 4 * 2),
            GLU())

        up_num, up_layers = [], []
        num_in, num_out = 1, 1
        up_num = range(1, int(np.log2(image_size[0]))-1)

        for i, num in enumerate(up_num):
            if i == 0:
                num_in = num
            num_out = 2**num
            up_layers.append(upBlock(self.dim//num_in, self.dim//num_out))
            num_in = num_out

        self.up_low_layers = nn.Sequential(*up_layers)
        self.up_mid_layer = upBlock(self.dim//num_out, self.dim//num_out//2)

        gnet_layers = [
            nn.Conv2d(self.dim//num_out, self.dim//num_out, kernel_size=3, padding=1),
            get_activation('leakyrelu'),
            nn.Conv2d(self.dim//num_out, 3, kernel_size=1, padding=0),
        ]
        self.gnet_low = nn.Sequential(*gnet_layers)

        gnet_layers = [
            nn.Conv2d(self.dim//num_out//2, self.dim//num_out//2, kernel_size=3, padding=1),
            get_activation('leakyrelu'),
            nn.Conv2d(self.dim//num_out//2, 3, kernel_size=1, padding=0),
        ]
        self.gnet_mid = nn.Sequential(*gnet_layers)

    def forward(self, x):
        """
        Required Inputs:
        - placeholder: This is a placeholder.

        Optional Inputs:
        - placeholder: This is a placeholder.
        """

        vecs = self.fc(x.view(-1, 512))
        vecs = vecs.view(x.size()[0], -1, 4, 4)

        img_code_low = self.up_low_layers(vecs)
        img_pred_low = self.gnet_low(img_code_low)

        img_code_mid = self.up_mid_layer(img_code_low)
        img_pred_mid = self.gnet_mid(img_code_mid)

        return (img_pred_low, img_pred_mid)


if __name__ == '__main__':
    # Demo Input
    voice_embeddings = torch.ones((16, 512, 1, 1))
    seq_voice_embeddings = torch.ones((16, 512, 270))

    # Test V2F 1D CNN
    mr_decoder_kwargs = {
        'input_channel': 512,
        'base_block_channels': [1024, 512, 256, 128, 64],
        'mid_block_channel': 32,
        'high_block_channel': 32,
        'output_channel': 3,
        'normalization': 'batch',
        'use_up_block': True,
        'extra_feature_map': True,
        }
    mr_decoder = MRDecoder(**mr_decoder_kwargs)
    print(mr_decoder)
    for i, imgs in enumerate(mr_decoder(voice_embeddings)):
        print('MRDecoder Output {} shape:'.format(i), imgs.shape)

    # Test V2F Decoder
    v2f_decoder_kwargs = {
        'input_channel': 512,
        'channels': [1024, 512, 256, 128, 64],
        'output_channel': 3,
        'normalization': 'batch',
        }
    v2f_decoder = V2FDecoder(**v2f_decoder_kwargs)
    print(v2f_decoder)
    print('V2FDecoder Output shape:', v2f_decoder(voice_embeddings).shape)
    print('V2F Sequential:')
    print(v2f_decoder.model[5])
    for name, param in v2f_decoder.model[5].named_parameters():
        print(name)
    #for module in v2f_decoder.model:
    #    print(module)

    v2f_decoder_kwargs = {
        'input_channel': 512,
        'cnn_channels': [1024, 512, 256, 128, 64],
        'attn_channels': [0, 0, 128, 0, 32],
        'output_channel': 3,
        'seq_in_dim': 512,
        'normalization': 'none',
        }
    attn_v2f_decoder = AttnV2FDecoder(**v2f_decoder_kwargs)
    print(attn_v2f_decoder)
    print('AttnV2FDecoder Output shape:', \
        attn_v2f_decoder(voice_embeddings, seq_voice_embeddings).shape)
    print('Return Attention Weights:',
          attn_v2f_decoder.return_attn_weights().shape,
          attn_v2f_decoder.return_attn_weights()[0][0][0:10])

    dimLatentVector = 512
    depthScale0 = 512

    pgandecoder = PganDecoder(dimLatent=dimLatentVector,
                          depthScale0=depthScale0,
                          initBiasToZero=True,
                          leakyReluLeak=0.2,
                          normalization=True,
                          generationActivation=None,
                          dimOutput=3,
                          equalizedlR=True,
                          depthOtherScales=[512,512,512,256])

    img = pgandecoder(voice_embeddings)
    
