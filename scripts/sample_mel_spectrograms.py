'''
This file samples several mel spectrograms from the VoxCeleb Dataset, and
    visualize them as png images
'''


import os
import PIL

import sys
sys.path.append('./')
#print(sys.path)
from datasets import VoxDataset, fast_mel_deprocess_batch
from utils.visualization.plot import plot_mel_spectrogram #get_np_plot,
from tensorflow.io.gfile import mkdir


VOX_DIR = os.path.join('./data', 'VoxCeleb')
output_dir = os.path.join('./output', 'sampled_mel_spectrograms')
mkdir(output_dir)


def main():
    vox_dataset = VoxDataset(
        data_dir=VOX_DIR,
        image_size=(64, 64),
        nframe_range=(300, 600),
        face_type='masked',
        image_normalize_method='imagenet',
        mel_normalize_method='vox_mel',
        mel_seg_window_stride=(125, 63),
        split_set='test',
        split_json=os.path.join(VOX_DIR, 'split.json'))

    log_mels = vox_dataset.get_all_mel_segments_of_id(5)

    log_mels_de = fast_mel_deprocess_batch(log_mels, 'vox_mel')

    for i in range(len(log_mels_de)):
        log_mel = log_mels_de[i].cpu().detach().numpy()
        buf = plot_mel_spectrogram(
            log_mel,
            colorbar=False,
            label=False,
            coordinate=False,
            remove_boarder=True)
        log_mel_img = PIL.Image.open(buf)
        log_mel_img.save(os.path.join(output_dir, '{}.png'.format(i)))


if __name__ == '__main__':
    main()
