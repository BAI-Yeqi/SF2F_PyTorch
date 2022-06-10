'''
Implementations of different Fusers
'''


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

try:
    from attention import Attention
except:
    from .attention import Attention


class AttentionFuserV1(nn.Module):
    def __init__(self,
                 dimensions,
                 dim_out,
                 attention_type='general',
                 linear_out_layer=True,
                 ignore_tanh=False):
        '''
        1 Layer Attnetion Fuser
        '''
        nn.Module.__init__(self)
        self.attention = Attention(
            dimensions,
            dim_out,
            attention_type=attention_type,
            linear_out_layer=linear_out_layer,
            ignore_tanh=ignore_tanh)

    def forward(self, embeddings):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        # Embeddings: [B, N, C]
        embeddings, attention_weights = \
            self.attention(embeddings, embeddings)
        # [B, C, N]
        embeddings = embeddings.transpose(1, 2)
        #print('Transposed Shape:', embeddings.shape)
        embeddings = F.avg_pool1d(
            embeddings, embeddings.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)
        #print('Attention Fuser Output Shape:', embeddings.shape)
        return embeddings


class AttentionFuserV2(nn.Module):
    def __init__(self,
                 dimensions,
                 attention_type='general'):
        '''
        2-Layer Attention
        '''
        nn.Module.__init__(self)
        self.attention_1 = Attention(
            dimensions,
            dimensions,
            attention_type=attention_type,
            linear_out_layer=True,
            ignore_tanh=False)
        self.attention_2 = Attention(
            dimensions,
            dimensions,
            attention_type=attention_type,
            linear_out_layer=True,
            ignore_tanh=True)

    def forward(self, embeddings):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        # Embeddings: [B, N, C]
        embeddings, attention_weights = \
            self.attention_1(embeddings, embeddings)
        embeddings, attention_weights = \
            self.attention_2(embeddings, embeddings)
        # [B, C, N]
        embeddings = embeddings.transpose(1, 2)
        #print('Transposed Shape:', embeddings.shape)
        embeddings = F.avg_pool1d(
            embeddings, embeddings.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)
        #print('Attention Fuser Output Shape:', embeddings.shape)
        return embeddings


class AttentionFuserV3(nn.Module):
    def __init__(self,
                 dimensions,
                 attention_type='general'):
        '''
        2-Layer Attention with Residual Connection
        '''
        nn.Module.__init__(self)
        self.attention_1 = Attention(
            dimensions,
            dimensions,
            attention_type=attention_type,
            linear_out_layer=True,
            ignore_tanh=False)
        self.attention_2 = Attention(
            dimensions*2,
            dimensions,
            attention_type=attention_type,
            linear_out_layer=True,
            ignore_tanh=True)

    def forward(self, input):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        # Embeddings: [B, N, C]
        hidden, attention_weights = \
            self.attention_1(input, input)
        hidden = F.normalize(hidden, dim=2)
        # Residual Connection
        combined = torch.cat((hidden, input), dim=2)
        # 2nd Attention
        embeddings, attention_weights = \
            self.attention_2(combined, combined)
        # [B, C, N]
        embeddings = embeddings.transpose(1, 2)
        #print('Transposed Shape:', embeddings.shape)
        embeddings = F.avg_pool1d(
            embeddings, embeddings.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)
        #print('Attention Fuser Output Shape:', embeddings.shape)
        return embeddings
