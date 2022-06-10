'''
Flexible Encoder-Decoder Framework
'''


import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

try:
    from .model_collection import model_collection
except:
    from model_collection import model_collection
#from .voice_encoders import V2F1DCNN
#from .face_decoders import V2FDecoder


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder_arch,
                 encoder_kwargs,
                 decoder_arch,
                 decoder_kwargs,
                 image_size):
        super(EncoderDecoder, self).__init__()
        # Initialize Encoder Decoder
        self.encoder = getattr(model_collection, encoder_arch)(**encoder_kwargs)
        self.decoder = getattr(model_collection, decoder_arch)(**decoder_kwargs)
        #self.model = nn.Sequential(
        #    self.encoder,
        #    self.decoder
        #)

    def forward(self, x, average_voice_embedding=False):
        '''
        average_encoder_embedding: an option for evaluation, not needed for
        training
        '''
        others = {}
        if self.encoder.return_seq:
            emb, seq_emb = self.encoder(x)
            imgs_pred = self.decoder(emb, seq_emb)
        else:
            emb = self.encoder(x)
            if average_voice_embedding:
                emb = torch.mean(emb, 0, keepdim=True)
            imgs_pred = self.decoder(emb)
        others['cond'] = emb
        return imgs_pred, others

    def train_fuser_only(self):
        print('Training Attention Fuser Only')
        for name, param in self.named_parameters():
            if 'attn_fuser' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


if __name__ == '__main__':
    # Demo Input
    log_mels = torch.ones((16, 40, 127))

    # Test V2F 1D CNN
    v2f_id_cnn_kwargs = {
        'input_channel': 40,
        'channels': [256, 384, 576, 864],
        'output_channel': 64,
        }
    # Test V2F 1D CNN
    v2f_decoder_kwargs = {
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64],
        'output_channel': 3,
        }

    baseline_v2f = EncoderDecoder(
        encoder_arch='V2F1DCNN',
        encoder_kwargs=v2f_id_cnn_kwargs,
        decoder_arch='V2FDecoder',
        decoder_kwargs=v2f_decoder_kwargs
    )

    print(baseline_v2f)
    print('baseline_v2f Output shape:', baseline_v2f(log_mels).shape)
