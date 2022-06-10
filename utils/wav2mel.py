'''
Convert *.wav or *.m4a into mel spectrogram\

for unit test, run:
    python utils/wav2mel.py
'''


import random
import numpy as np
import argparse
import time
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import webrtcvad
try:
    import vad_ex
except:
    from utils import vad_ex


def vad_process(path):
    # VAD Process
    if path.endswith('.wav'):
        audio, sample_rate = vad_ex.read_wave(path)
    elif path.endswith('.m4a'):
        audio, sample_rate = vad_ex.read_m4a(path)
    else:
        raise TypeError('Unsupported file type: {}'.format(path.split('.')[-1]))

    vad = webrtcvad.Vad(1)
    frames = vad_ex.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
    total_wav = b""
    for i, segment in enumerate(segments):
        total_wav += segment
    # Without writing, unpack total_wav into numpy [N,1] array
    # 16bit PCM 기준 dtype=np.int16
    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
    #print("read audio data from byte string. np array of shape:" + \
    #    str(wav_arr.shape))
    return wav_arr, sample_rate


def wav_to_mel(path, nfilt=40):
    '''
    Output shape: (nfilt, length)
    '''
    wav_arr, sample_rate = vad_process(path)
    #print("sample_rate:", sample_rate)
    logmel_feats = logfbank(
        wav_arr,
        samplerate=sample_rate,
        nfilt=nfilt)
    #print("created logmel feats from audio data. np array of shape:" \
    #    + str(logmel_feats.shape))
    return np.transpose(logmel_feats)


if __name__ == '__main__':
    import os
    VOX_DIR = os.path.join('./data', 'VoxCeleb')
    WAV_DIR = os.path.join('./data', 'VoxCeleb', 'raw_wav')
    #wav_path = 'vox1/test/id10270/zjwijMp0Qyw/00001.wav'
    wav_path = 'vox1/dev/id10001/J9lHsKG98U8/00026.wav'
    wav_path = os.path.join(WAV_DIR, wav_path)
    m4a_path = 'vox2/test/id04253/dfbCPe2xOPA/00257.m4a'
    m4a_path = os.path.join(WAV_DIR, m4a_path)

    #wav_arr, sample_rate = vad_process(wav_path)
    #print(wav_arr, sample_rate)
    #wav_arr, sample_rate = vad_process(m4a_path)
    #print(wav_arr, sample_rate)

    print(wav_to_mel(wav_path).shape)
    print(wav_to_mel(m4a_path).shape)

    # Compare with preprocessed pickles
    import sys
    sys.path.append('./')
    from datasets.vox_dataset import VoxDataset
    # Config
    image_size = (256, 256)
    #image_normalize_method = 'imagenet'
    image_normalize_method = 'standard'
    mel_normalize_method = 'vox_mel'
    test_case_dir = os.path.join('./data', 'test_cases')
    os.makedirs(test_case_dir, exist_ok=True)

    # Dataset
    vox_dataset = VoxDataset(
        data_dir=VOX_DIR,
        image_size=image_size,
        image_normalize_method=image_normalize_method,
        mel_normalize_method=mel_normalize_method)
    np_mel = vox_dataset.load_mel_gram(mel_pickle= \
        './data/VoxCeleb/vox1/mel_spectrograms/A.J._Buckley' + \
            '/id10001_J9lHsKG98U8_00026.pickle')
    print('Load from pickle:', np_mel.shape)
    print('Calculated by program:', wav_to_mel(wav_path).shape)

    print(np_mel - wav_to_mel(wav_path))
