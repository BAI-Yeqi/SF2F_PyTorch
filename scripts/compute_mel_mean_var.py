# -*- coding: utf-8 -*-
'''
Compute the mean and variance of VoxCeleb dataset's mel spectrograms
'''


import os
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('./')
print(sys.path)
from datasets.vox_dataset import VoxDataset


def mean():
    epoch_num = 100
    vox_dataset = VoxDataset(
        data_dir='./data/VoxCeleb',
        image_size=(64, 64),
        image_normalize_method=None,
        mel_normalize_method=None)
    loader_kwargs = {
        'batch_size': 128,
        'num_workers': 8,
        'shuffle': False,
        "drop_last": True,
    }
    print(vox_dataset)
    vox_loader = DataLoader(vox_dataset, **loader_kwargs)

    total_mean = torch.tensor(0.0)
    total_std = torch.tensor(0.0)
    min_length = 10000000
    for epoch in range(epoch_num):
        for iter, batch in enumerate(vox_loader):
            #images, log_mels, mel_length = batch
            images, log_mels = batch
            total_mean = total_mean + torch.mean(log_mels)
            total_std = total_std + torch.std(log_mels)
            #cur_min_length = torch.min(mel_length)
            #if cur_min_length < min_length:
            #    min_length = cur_min_length
        print(epoch)

    avg_mean = (total_mean / epoch_num) / (iter + 1)
    avg_std = (total_std / epoch_num) / (iter + 1)

    print('Dataset Mel Spectrogram Mean:', avg_mean)
    print('Dataset Mel Spectrogram Standard Deviation:', avg_std)
    print('Dataset minimum mel spectrogram length:', min_length)

if __name__ == '__main__':
    mean()
