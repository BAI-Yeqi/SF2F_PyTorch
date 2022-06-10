'''
Implementations of different Voice Encoders
'''


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

try:
    from fusers import AttentionFuserV1, AttentionFuserV2, AttentionFuserV3
except:
    from .fusers import AttentionFuserV1, AttentionFuserV2, AttentionFuserV3


class EncoderModelCollection():
    def __init__(self):
        self.AttentionFuserV1 = AttentionFuserV1
        self.AttentionFuserV2 = AttentionFuserV2
        self.AttentionFuserV3 = AttentionFuserV3

encoder_model_collection = EncoderModelCollection()


class TransposeLayer(nn.Module):
    def __init__(self, dim_1, dim_2):
        '''
        Transpose Layer
        '''
        nn.Module.__init__(self)
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        x = x.transpose(self.dim_1, self.dim_2)
        return x


class CNN1DBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel=3,
                 stride=2,
                 padding=1):
        '''
        1D CNN block
        '''
        nn.Module.__init__(self)
        self.layers = []
        self.layers.append(
            nn.Conv1d(channel_in,
                      channel_out,
                      kernel,
                      stride,
                      padding,
                      bias=False))
        self.layers.append(
            nn.BatchNorm1d(channel_out, affine=True))
        self.layers.append(
            nn.ReLU(inplace=True))
        self.model = nn.Sequential(*self.layers)

    def forward(self, seq_in):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        seq_out = self.model(seq_in)
        return seq_out


class Inception1DBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_k2,
                 channel_k3,
                 channel_k5,
                 channel_k7):
        '''
        Basic building block of 1D Inception Encoder
        '''
        nn.Module.__init__(self)
        self.conv_k2 = None
        self.conv_k3 = None
        self.conv_k5 = None

        if channel_k2 > 0:
            self.conv_k2 = CNN1DBlock(
                channel_in,
                channel_k2,
                2, 2, 1)
        if channel_k3 > 0:
            self.conv_k3 = CNN1DBlock(
                channel_in,
                channel_k3,
                3, 2, 1)
        if channel_k5 > 0:
            self.conv_k5 = CNN1DBlock(
                channel_in,
                channel_k3,
                5, 2, 2)
        if channel_k7 > 0:
            self.conv_k7 = CNN1DBlock(
                channel_in,
                channel_k3,
                7, 2, 3)

    def forward(self, input):
        output = []
        if self.conv_k2 is not None:
            c2_out = self.conv_k2(input)
            output.append(c2_out)
        if self.conv_k3 is not None:
            c3_out = self.conv_k3(input)
            output.append(c3_out)
        if self.conv_k5 is not None:
            c5_out = self.conv_k5(input)
            output.append(c5_out)
        if self.conv_k7 is not None:
            c7_out = self.conv_k7(input)
            output.append(c7_out)
        #print(c2_out.shape, c3_out.shape, c5_out.shape, c7_out.shape)
        if output[0].shape[-1] > output[1].shape[-1]:
            output[0] = output[0][:, :, 0:-1]
        output = torch.cat(output, 1)
        #print(output.shape)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PosEmbLayer(nn.Module):
    def __init__(self, max_length=2000, pos_embedding_dim=5):
        '''
        Postional embedding block
        '''
        nn.Module.__init__(self)

        self.max_length = max_length
        self.pos_embedding_dim = pos_embedding_dim
        # Position Embedding
        # Each position index in range (0, 2 * max_length - 1) have its own embedding
        self.pos_embedding = nn.Embedding(
            max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, seq_in):
        '''
        seq_in: tensor with shape [N, L, D]
        '''
        N, L, D = seq_in.shape
        # (L, )
        pos_idx = torch.tensor(np.arange(0, L))
        # (1, L)
        pos_idx = pos_idx.unsqueeze(0)
        # (N, L)
        pos_idx = pos_idx.repeat(N, 1).cuda()
        # (N, L, D_pos)
        pos_emb = self.pos_embedding(pos_idx).type(seq_in.type())
        #pos_emb = torch.transpose(pos_emb, 1, 2)
        seq_out = torch.cat([seq_in, pos_emb], 2)
        # (N, L, D + D_pos)
        return seq_out


class TransEncoder(nn.Module):
    def __init__(self,
                 input_channel=40,
                 cnn_channels=[512, ],
                 transformer_dim=512,
                 transformer_depth=3,
                 return_seq=True,
                 sin_pos_encoding=False,
                 pos_embedding_dim=8,
                 add_noise=False,
                 normalize_embedding=False):
        super(TransEncoder, self).__init__()
        self.layers = []

        if pos_embedding_dim > 0:
            # Postional Embedding Layer
            # (N, L, D)
            self.layers.append(TransposeLayer(1, 2))
            # (N, D, L)
            self.layers.append(
                PosEmbLayer(2000, pos_embedding_dim))
            # (N, L, D)
            self.layers.append(TransposeLayer(1, 2))

        # CNN Layer for dimension conversion
        self.layers.append(
            CNN1DBlock(input_channel+pos_embedding_dim, cnn_channels[0]))
        for cnn_layer_id in range(1, len(cnn_channels)):
            self.layers.append(CNN1DBlock(
                cnn_channels[cnn_layer_id - 1],
                cnn_channels[cnn_layer_id]))

        self.layers.append(TransposeLayer(1, 2))
        # Sin Positional Encoding
        if sin_pos_encoding:
            self.layers.append(PositionalEncoding(transformer_dim))

        # Transformer
        trans_layer = nn.TransformerEncoderLayer(
            transformer_dim,
            8,
            transformer_dim)
        transformer_encoder = nn.TransformerEncoder(
            trans_layer, transformer_depth)
        self.layers.append(transformer_encoder)
        self.layers.append(TransposeLayer(1, 2))

        self.model = nn.Sequential(*self.layers)
        self.add_noise = add_noise
        self.normalize_embedding = normalize_embedding
        self.return_seq = return_seq

    def forward(self, seq_in):
        seq_out = self.model(seq_in)
        #for layer in self.layers:
        #    print(layer)
        #    x = layer(x)
        #    print(x.shape)
        embeddings = F.avg_pool1d(seq_out, seq_out.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)

        if self.normalize_embedding:
            embeddings = F.normalize(embeddings)
        if self.add_noise:
            noise = 0.05 * torch.randn(
                embeddings.shape[0], embeddings.shape[1], 1, 1)
            noise = noise.type(embeddings.type())
            embeddings = embeddings + noise
            if self.normalize_embedding:
                embeddings = F.normalize(embeddings)

        if self.return_seq:
            return embeddings, seq_out
        else:
            return embeddings


class V2F1DCNN(nn.Module):
    def __init__(self,
                 input_channel,
                 channels,
                 output_channel,
                 add_noise=False,
                 normalize_embedding=False,
                 return_seq=False,
                 inception_mode=False,
                 segments_fusion=False,
                 normalize_fusion=False,
                 fuser_arch='Attention',
                 fuser_kwargs=None):
        super(V2F1DCNN, self).__init__()
        if inception_mode:
            self.model = nn.Sequential(
                Inception1DBlock(
                    channel_in=input_channel,
                    channel_k2=channels[0]//4,
                    channel_k3=channels[0]//4,
                    channel_k5=channels[0]//4,
                    channel_k7=channels[0]//4),
                Inception1DBlock(
                    channel_in=channels[0],
                    channel_k2=channels[1]//4,
                    channel_k3=channels[1]//4,
                    channel_k5=channels[1]//4,
                    channel_k7=channels[1]//4),
                Inception1DBlock(
                    channel_in=channels[1],
                    channel_k2=channels[2]//4,
                    channel_k3=channels[2]//4,
                    channel_k5=channels[2]//4,
                    channel_k7=channels[2]//4),
                Inception1DBlock(
                    channel_in=channels[2],
                    channel_k2=channels[3]//4,
                    channel_k3=channels[3]//4,
                    channel_k5=channels[3]//4,
                    channel_k7=channels[3]//4),
                nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
                nn.BatchNorm1d(channels[0], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
                nn.BatchNorm1d(channels[1], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
                nn.BatchNorm1d(channels[2], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
                nn.BatchNorm1d(channels[3], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
            )
        self.add_noise = add_noise
        self.normalize_embedding = normalize_embedding
        self.return_seq = return_seq
        self.output_channel = output_channel

        self.segments_fusion = segments_fusion
        self.normalize_fusion = normalize_fusion
        if segments_fusion:
            #self.attn_fuser = Attention(
            #    output_channel,
            #    output_channel,
            #    ignore_tanh=True)
            self.attn_fuser = \
                getattr(encoder_model_collection, fuser_arch)(**fuser_kwargs)

    def forward(self, x):
        # In case more than one mel segment per person is passed
        if len(x.shape) == 4:
            fusion_mode = True
            B, N, C, L = x.shape
            #print('Fusion Mode On! Input Shape:', x.shape)
            x = x.view(B*N, C, L)
        else:
            fusion_mode = False
            B, C, L = x.shape
        x = self.model(x)
        embeddings = F.avg_pool1d(x, x.size()[2], stride=1)
        embeddings = embeddings.view(embeddings.size()[0], -1, 1, 1)

        if self.normalize_embedding:
            embeddings = F.normalize(embeddings)
        if self.add_noise:
            noise = 0.05 * torch.randn(x.shape[0], x.shape[1], 1, 1)
            noise = noise.type(embeddings.type())
            embeddings = embeddings + noise
            if self.normalize_embedding:
                embeddings = F.normalize(embeddings)

        # Restore Tensor shape
        if fusion_mode:
            #print(embeddings.shape)
            C_emb = embeddings.shape[1]
            embeddings = embeddings.view(B, N, C_emb)
            # Attention fusion
            embeddings = self.attn_fuser(embeddings)
            if self.normalize_fusion:
                embeddings = F.normalize(embeddings)

        if self.return_seq:
            return embeddings, x
        else:
            return embeddings

    def print_param(self):
        print('All parameters:')
        for name, param in self.named_parameters():
            print(name)

    def print_trainable_param(model):
        print('Trainable Parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    def train_fuser_only(self):
        print('Training Attention Fuser Only')
        for name, param in self.named_parameters():
            if 'attn_fuser' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def init_attn_fusion(self):
        self.attn_fuser = Attention(
            self.output_channel,
            self.output_channel,
            ignore_tanh=True)

if __name__ == '__main__':
    import time
    # Demo Input
    log_mels = torch.ones((16, 40, 151))
    pos_log_mel = torch.ones((16, 48, 150)).transpose(1, 2)

    # Test Inception1DBlock
    incept_block = Inception1DBlock(
        channel_in=40,
        channel_k2=64,
        channel_k3=64,
        channel_k5=64,
        channel_k7=64)
    incept_block(log_mels)

    # Test V2F 1D CNN
    log_mel_segs = torch.ones((16, 20, 40, 150))
    v2f_id_cnn_kwargs = {
        'input_channel': 40,
        'channels': [256, 384, 576, 864],
        'output_channel': 64,
        'segments_fusion': True,
        'inception_mode': True,
        }
    v2f_id_cnn_fuse = V2F1DCNN(**v2f_id_cnn_kwargs)
    print(v2f_id_cnn_fuse)
    print('V2F1DCNN Output shape:', v2f_id_cnn_fuse(log_mel_segs).shape)
    v2f_id_cnn_fuse.print_param()
    v2f_id_cnn_fuse.train_fuser_only()
    v2f_id_cnn_fuse.print_trainable_param()

    # test a simple transformer layer
    trans_layer = nn.TransformerEncoderLayer(48, 8,
        dim_feedforward=48, dropout=0.1, activation='relu')
    print(trans_layer(pos_log_mel).shape)


    # Transformer
    trans_kwargs = {
        'input_channel': 40,
        'cnn_channels': [512, 512],
        'transformer_dim': 512,
        'transformer_depth': 2,
        'return_seq': True,
        'pos_embedding_dim': 0, #8
        'sin_pos_encoding': True, #False
    }
    trans_encoder = TransEncoder(**trans_kwargs).cuda()
    print(trans_encoder)
    emb, seq_emb = trans_encoder(log_mels.cuda())
    print('TransEncoder Output shape:', emb.shape, seq_emb.shape)

    # Test V2F 1D CNN
    v2f_id_cnn_kwargs = {
        'input_channel': 40,
        'channels': [256, 384, 576, 864],
        'output_channel': 64,
        }
    v2f_id_cnn = V2F1DCNN(**v2f_id_cnn_kwargs)
    print(v2f_id_cnn)
    print('V2F1DCNN Output shape:', v2f_id_cnn(log_mels).shape)

    # Test V2F 1D CNN Sequential Return
    v2f_id_cnn_kwargs = {
        'input_channel': 40,
        'channels': [256, 384, 576, 864],
        'output_channel': 64,
        'return_seq': True,
        }
    v2f_id_cnn = V2F1DCNN(**v2f_id_cnn_kwargs)
    print(v2f_id_cnn)
    emb, seq_emb = v2f_id_cnn(log_mels)
    print('V2F1DCNN Output shape:', emb.shape, seq_emb.shape)
