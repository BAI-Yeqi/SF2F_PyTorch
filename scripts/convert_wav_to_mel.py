'''
Convert raw_wavs in VoxCeleb dataSet to mel_gram

    1. Build two id2name mapping for vox1 & vox2
    2. Create a folder for each identity with his/her name
    3. for each raw wav file, convert it to mel_gram,
        save it under the identity folder
'''


import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import shutil
import gc
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from tensorflow.io import gfile
import sys
sys.path.append('./')
from utils.wav2mel import wav_to_mel
from concurrent.futures import ProcessPoolExecutor, as_completed


VOX_DIR = os.path.join('./data/VoxCeleb')
vox1_raw = os.path.join(VOX_DIR, 'raw_wav', 'vox1')
vox2_raw = os.path.join(VOX_DIR, 'raw_wav', 'vox2')
vox1_meta_csv = os.path.join(VOX_DIR, 'vox1', 'vox1_meta.csv')
vox2_meta_csv = os.path.join(VOX_DIR, 'vox2', 'full_vox2_meta.csv')

parser = argparse.ArgumentParser()

# arguments:
#parser.add_argument('--output_folder',
#    default=os.path.join(PROC_DIR, 'VoxCeleb1', 'faces'),
#    help='folder that contains the outputs')

args = parser.parse_args()

class WavConvertor:
    def __init__(self):
        self.load_metadata()
        self.get_id2name()
        self.get_wav_dirs()
        self.create_output_dirs()

    def load_metadata(self):
        self.vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
        self.vox2_df = pd.read_csv(vox2_meta_csv, sep=',')
        #print(self.vox1_df.head())
        #print(self.vox1_df.columns)
        #print(self.vox2_df.head())
        #print(self.vox2_df.columns)

    def get_id2name(self):
        self.vox1_id2name = dict(
            zip(self.vox1_df['VoxCeleb1 ID'], self.vox1_df['VGGFace1 ID']))
        self.vox2_id2name = dict(
            zip(self.vox2_df['VoxCeleb2ID'], self.vox2_df['Name']))
        #print((self.vox2_id2name))

    def convert_identity(self, wav_dir, mel_home_dir, dataset):
        '''
        Create a mel_gram folder for the given identity

        Arguments:
            1. wav_dir (str): path to the identity's raw_wav folder
            2. mel_home_dir (str): path to the identity's raw_wav folder
            3. dataset (str): 'vox1' or 'vox2'
        '''
        spkid = wav_dir.split('/')[-1]
        # In case the input path ends with '/'
        if spkid == '':
            spkid = wav_dir.split('/')[-2]
        if dataset == 'vox1':
            name = self.vox1_id2name[spkid]
        elif dataset == 'vox2':
            name = self.vox2_id2name[spkid]
        else:
            raise ValueError("Invalid dataset argument")
        #print(name)

        # Create mel_gram directory of the speaker
        mel_dir = os.path.join(mel_home_dir, name)
        gfile.mkdir(mel_dir)

        clipids = os.listdir(wav_dir)
        for clipid in clipids:
            clip_dir = os.path.join(wav_dir, clipid)
            wav_files = os.listdir(clip_dir)
            for wav_file in wav_files:
                # Read and process the wav
                wav_path = os.path.join(clip_dir, wav_file)
                try:
                    wavid = wav_file.replace('.wav', '').replace('.m4a', '')
                    pickle_name = "{}_{}_{}.pickle".format(
                        spkid, clipid, wavid)
                    pickle_path = os.path.join(
                        mel_dir, pickle_name)
                    if os.path.exists(pickle_path):
                        # Skip if exists
                        continue
                    # Vox1 use .wav format, Vox2 use .m4a format
                    log_mel = wav_to_mel(wav_path)
                    pickle_dict = {
                        'LogMel_Features': log_mel,
                        'spkid': spkid,
                        'clipid': clipid,
                        'wavid': wavid
                    }
                    pickle.dump(
                        pickle_dict,
                        open(pickle_path, "wb")
                    )
                except IndexError:
                    pass
        gc.collect()

    def get_wav_dirs(self):
        '''
        Generate a list containing paths to the wav_dir of all the speakers
        '''
        # The original VoxCeleb dataset comes with test set and dev set
        vox1_dev = os.path.join(vox1_raw, 'dev')
        vox1_test = os.path.join(vox1_raw, 'test')
        vox2_dev = os.path.join(vox2_raw, 'dev')
        vox2_test = os.path.join(vox2_raw, 'test')
        # Get Vox1 wav_dir paths
        vox1_wav_dirs = []
        for wav_dir in os.listdir(vox1_dev):
            vox1_wav_dirs.append(os.path.join(vox1_dev, wav_dir))
        for wav_dir in os.listdir(vox1_test):
            vox1_wav_dirs.append(os.path.join(vox1_test, wav_dir))
        # Get Vox2 wav_dir paths
        vox2_wav_dirs = []
        for wav_dir in os.listdir(vox2_dev):
            vox2_wav_dirs.append(os.path.join(vox2_dev, wav_dir))
        for wav_dir in os.listdir(vox2_test):
            vox2_wav_dirs.append(os.path.join(vox2_test, wav_dir))
        self.vox1_wav_dirs = vox1_wav_dirs
        self.vox2_wav_dirs = vox2_wav_dirs
        return 0

    def create_output_dirs(self):
        self.vox1_mel = os.path.join(VOX_DIR, 'vox1', 'mel_spectrograms')
        self.vox2_mel = os.path.join(VOX_DIR, 'vox2', 'mel_spectrograms')
        gfile.mkdir(self.vox1_mel)
        gfile.mkdir(self.vox2_mel)

    def _worker(self, job_id, infos):
            for i, info in enumerate(infos):
                self.convert_identity(info[0], info[1], info[2])
                print('job #{} prcess {} {} done'.format(job_id, i, info[0]))
                
    def convert_wav_to_mel(self, n_jobs=1):

        infos = []
        for wav_dir in self.vox1_wav_dirs:
            infos.append((wav_dir, self.vox1_mel, 'vox1'))
        print(len(infos))
        # l1 = len(infos)
        for wav_dir in self.vox2_wav_dirs:
            infos.append((wav_dir, self.vox2_mel, 'vox2'))
        # print(len(infos) - l1)
        
        n_wav_dirs = len(infos)
        n_jobs = n_jobs if n_jobs <= n_wav_dirs else n_wav_dirs
        n_wav_dirs_per_job = n_wav_dirs // n_jobs
        process_index = []
        for ii in range(n_jobs):
            process_index.append([ii*n_wav_dirs_per_job, (ii+1)*n_wav_dirs_per_job])
        if n_jobs * n_wav_dirs_per_job != n_wav_dirs:
            process_index[-1][-1] = n_wav_dirs
        
        futures = set()
        with ProcessPoolExecutor() as executor:
            for job_id in range(n_jobs):
                # future = executor.submit(_worker, process_index[job_id][0], process_index[job_id][1])
                future = executor.submit(self._worker, job_id, infos[process_index[job_id][0]:process_index[job_id][1]])
                futures.add(future)
                print('submit job {}, {}-{}'.format(job_id, process_index[job_id][0], process_index[job_id][1]))
            for future in as_completed(futures):
                pass
        
        print("Done.")



def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', '-n_jobs', type=int, default=1)
    args = parser.parse_args()
    
    wav_convertor = WavConvertor()
    wav_convertor.convert_wav_to_mel(args.n_jobs)
    #gfile.mkdir('./data/test')
    #wav_convertor.convert_identity(
    #    'data/VoxCeleb/raw_wav/vox1/dev/id10001/',
    #    './data/test', 'vox1')


if __name__ == '__main__':
    print(os.listdir(VOX_DIR))
    main()
