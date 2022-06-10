'''
Build a demo inference dataset
'''


import os
import shutil
import json
import pandas as pd
from PIL import Image
from pydub import AudioSegment


DATA_DIR = './data'
vox2_face_dir = os.path.join(DATA_DIR, 'VoxCeleb', 'vox2', 'masked_faces')
vox2_wav_dir = os.path.join(DATA_DIR, 'VoxCeleb', 'raw_wav', 'vox2')
vox2_meta_csv = os.path.join(DATA_DIR, 'VoxCeleb', 'vox2', 'full_vox2_meta.csv')
vox_split_json = os.path.join(DATA_DIR, 'VoxCeleb', 'split.json')
demo_dir = os.path.join(DATA_DIR, 'VoxCeleb', 'demo_data')

def main():
    try:
        shutil.rmtree(demo_dir)
    except:
        pass
    os.makedirs(demo_dir, exist_ok=True)
    #print(os.listdir(vox2_face_dir))
    #print(os.listdir(vox2_wav_dir))
    with open(vox_split_json) as json_file:
        split_dict = json.load(json_file)
    # A list of names in the vox2 test set (in out splition)
    test_list = split_dict['vox2']['test']
    # meta_csv
    meta_df = pd.read_csv(vox2_meta_csv)
    print(meta_df.head(5))
    name2id = {}
    name2set = {}
    for i in range(len(meta_df)):
        name = meta_df['Name'][i]
        id = meta_df['VoxCeleb2ID'][i]
        split_set = meta_df['Set'][i]
        name2id[name] = id
        name2set[name] = split_set
    #print(name2id)
    available_names = set(test_list).intersection(
        set(os.listdir(vox2_face_dir)))
    available_names = list(available_names)
    for name in available_names[:50]:
        id = name2id[name]
        split_set = name2set[name]
        face_dir = os.path.join(vox2_face_dir, name)
        #print(os.listdir(face_dir))
        wav_dir = os.path.join(vox2_wav_dir, split_set, id)
        # Avoid PermissionError
        try:
            #print(os.listdir(wav_dir))
            wav_sub_dir_list = os.listdir(wav_dir)
        except PermissionError:
            continue
        save_dir = os.path.join(demo_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        for i, jpg in enumerate(os.listdir(face_dir)):
            face_jpg = os.path.join(face_dir, jpg)
            img = Image.open(face_jpg)
            for reso, w in [('low', 64), ('mid', 128), ('high', 256)]:
                save_path = os.path.join(save_dir, '{}_{}.jpg'.format(i, reso))
                cur_img = img.resize((w, w))
                cur_img.save(save_path)
        # select 2 wavs
        for j, wav_sub_dir in enumerate(wav_sub_dir_list[:2]):
            #print(wav_sub_dir)
            wav_sub_dir = os.path.join(wav_dir, wav_sub_dir)
            wav_path = os.path.join(wav_sub_dir, os.listdir(wav_sub_dir)[0])
            wav_save_dir = os.path.join(save_dir, 'audio_{}'.format(j))
            os.makedirs(wav_save_dir, exist_ok=True)
            try:
                track = AudioSegment.from_file(wav_path, 'm4a')
                wav_save_path = os.path.join(wav_save_dir, '{}.wav'.format(j))
                file_handle = track.export(wav_save_path, format='wav')
            except:
                print("ERROR CONVERTING " + str(wav_path))
        print(name)

if __name__ == '__main__':
    main()
