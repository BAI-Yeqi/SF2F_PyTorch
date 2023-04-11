'''
VoxCeleb DataSet
'''


import json, os
import numpy as np
import random
#import pandas as pd
#import h5py
import pickle
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
#from skimage.transform import resize as imresize
try:
    from .utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch
except:
    from utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch


VOX_DIR = os.path.join('./data', 'VoxCeleb')

class VoxDataset(Dataset):
    def __init__(self,
                 data_dir,
                 image_size=(64, 64),
                 face_type='masked',
                 image_normalize_method='imagenet',
                 mel_normalize_method='vox_mel',
                 nframe_range=(100, 150),
                 split_set='train',
                 split_json=os.path.join(VOX_DIR, 'split.json'),
                 return_mel_segments=False,
                 mel_seg_window_stride=(125, 125),
                 image_left_right_avg=False,
                 image_random_hflip=False):
        '''
        A PyTorch Dataset for loading VoxCeleb 1 & 2 human speech
        (as mel spectrograms) and face (as image)

        Inputs:
        - data: Path to a directory where vox1 & vox2 data are held
        - face_type: 'masked' or 'origin'
        - image_size: Shape (h, w) to output the image
        - image_normalize_method: Method to normalize the image, 'imagenet' or
            'standard'
        - return_mel_segments: Return several segments of mel spectrogram
        - mel_seg_window_stride: Tuple (int, int), defines the window size and
            stride size when segmenting the mel spectrogram with sliding window
        - image_left_right_avg: flip the image, average the original and flipped
            image
        - image_random_hflip: Add random horizontal image flip to training set

        '''
        self.data_dir = data_dir
        self.image_size = image_size
        self.face_type = face_type
        self.face_dir = self.face_type + '_faces'
        self.image_normalize_method = image_normalize_method
        self.mel_normalize_method = mel_normalize_method
        self.nframe_range = nframe_range
        self.split_set = split_set
        self.split_json = split_json
        self.return_mel_segments = return_mel_segments
        self.mel_seg_window_stride = mel_seg_window_stride
        self.shuffle_mel_segments = True
        # This attribute is added to make the return segment mode to start from
        # a random time
        # Thus improve the randomness in fuser data
        self.mel_segments_rand_start = False
        self.image_left_right_avg = image_left_right_avg
        self.image_random_hflip = image_random_hflip

        self.load_split_dict()
        self.list_available_names()
        self.set_image_transform()
        self.set_mel_transform()

    def __len__(self):
        return len(self.available_names)

    def set_length(self, length):
        self.available_names = self.available_names[0:length]

    def __getitem__(self, index):
        '''
        Given an index, randomly return a face and a mel_spectrogram of this guy
        '''
        sub_dataset, name = self.available_names[index]
        # Face Image
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        image_jpg = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_jpg)

        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                #print("PIL Image:", np.array(image))
                image = image.convert('RGB')
                if self.image_left_right_avg:
                    arr = (np.array(image) / 2.0 + \
                        np.array(T.functional.hflip(image)) / 2.0).astype(
                            np.uint8)
                    image = PIL.Image.fromarray(arr, mode="RGB")
                image = self.image_transform(image)

        # Mel Spectrogram
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        mel_gram_pickle = random.choice(os.listdir(mel_gram_dir))
        mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
        if not self.return_mel_segments:
            # Return single segment
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
        else:
            log_mel = self.get_all_mel_segments_of_id(
                index, shuffle=self.shuffle_mel_segments)

        human_id = torch.tensor(index)

        return image, log_mel, human_id

    def get_all_faces_of_id(self, index):
        '''
        Given a id, return all the faces of him as a batch tensor, with shape
        (N, C, H, W)
        '''
        sub_dataset, name = self.available_names[index]
        faces = []
        # Face Image
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        for image_jpg in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_jpg)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    #print("PIL Image:", np.array(image))
                    image = self.image_transform(image.convert('RGB'))
                    faces.append(image)
        faces = torch.stack(faces)

        return faces

    def get_all_mel_segments_of_id(self,
                                   index,
                                   shuffle=False):
        '''
        Given a id, return all the speech segments of him as a batch tensor,
        with shape (N, C, L)
        '''
        sub_dataset, name = self.available_names[index]
        window_length, stride_length = self.mel_seg_window_stride
        segments = []
        # Mel Spectrogram
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        mel_gram_list = os.listdir(mel_gram_dir)
        if shuffle:
            random.shuffle(mel_gram_list)
        else:
            mel_gram_list.sort()
        seg_count = 0
        for mel_gram_pickle in mel_gram_list:
            mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
            mel_length = log_mel.shape[1]
            if self.mel_segments_rand_start:
                start = np.random.randint(mel_length - window_length) if mel_length >= window_length else 0
                log_mel = log_mel[:, start:]
                mel_length = log_mel.shape[1]
            # Calulate the number of windows that can be generated
            num_window = 1 + (mel_length - window_length) // stride_length
            # Sliding Window
            for i in range(0, num_window):
                start_time = i * stride_length
                segment = log_mel[:, start_time:start_time + window_length]
                segments.append(segment)
                seg_count = seg_count + 1
                if seg_count == 20: # 20
                    segments = torch.stack(segments)
                    return segments
        segments = torch.stack(segments)
        return segments

    def set_image_transform(self):
        print('Dataloader: called set_image_size', self.image_size)
        image_transform = [T.Resize(self.image_size), T.ToTensor()]
        if self.image_random_hflip and self.split_set == 'train':
            image_transform = [T.RandomHorizontalFlip(p=0.5),] + \
                image_transform
        if self.image_normalize_method is not None:
            print('Dataloader: called image_normalize_method',
                self.image_normalize_method)
            image_transform.append(imagenet_preprocess(
                normalize_method=self.image_normalize_method))
        self.image_transform = T.Compose(image_transform)

    def set_mel_transform(self):
        mel_transform = [T.ToTensor(), ]
        print('Dataloader: called mel_normalize_method',
            self.mel_normalize_method)
        if self.mel_normalize_method is not None:
            mel_transform.append(imagenet_preprocess(
                normalize_method=self.mel_normalize_method))
        mel_transform.append(torch.squeeze)
        self.mel_transform = T.Compose(mel_transform)

    def load_split_dict(self):
        '''
        Load the train, val, test set information from split.json
        '''
        with open(self.split_json) as json_file:
            self.split_dict = json.load(json_file)

    def list_available_names(self):
        '''
        Find the intersection of speech and face data
        '''
        self.available_names = []
        # List VoxCeleb1 data:
        for sub_dataset in ('vox1', 'vox2'):
            mel_gram_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, 'mel_spectrograms'))
            face_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, self.face_dir))
            available = \
                set(mel_gram_available).intersection(face_available)
            for name in available:
                if name in self.split_dict[sub_dataset][self.split_set]:
                    self.available_names.append((sub_dataset, name))

        self.available_names.sort()

    def load_mel_gram(self, mel_pickle):
        '''
        Load a speech's mel spectrogram from pickle file.

        Format of the pickled data:
            LogMel_Features
            spkid
            clipid
            wavid

        Inputs:
        - mel_pickle: Path to the mel spectrogram to be loaded.
        '''
        # open a file, where you stored the pickled data
        file = open(mel_pickle, 'rb')
        # dump information to that file
        data = pickle.load(file)
        # close the file
        file.close()
        log_mel = data['LogMel_Features']
        #log_mel = np.transpose(log_mel, axes=None)

        return log_mel

    def crop_or_pad(self, log_mel, out_frame):
        '''
        Log_mel padding/cropping function to cooperate with collate_fn
        '''
        freq, cur_frame = log_mel.shape
        if cur_frame >= out_frame:
            # Just crop
            start = np.random.randint(0, cur_frame-out_frame+1)
            log_mel = log_mel[..., start:start+out_frame]
        else:
            # Padding
            zero_padding = np.zeros((freq, out_frame-cur_frame))
            zero_padding = self.mel_transform(zero_padding)
            if len(zero_padding.shape) == 1:
                zero_padding = zero_padding.view([-1, 1])
            log_mel = torch.cat([log_mel, zero_padding], -1)

        return log_mel

    def collate_fn(self, batch):
        min_nframe, max_nframe = self.nframe_range
        assert min_nframe <= max_nframe
        np.random.seed()
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        #start = np.random.randint(0, max_nframe-num_frame+1)
        #batch = [(item[0], item[1][..., start:start+num_frame], item[2])
        #         for item in batch]

        batch = [(item[0],
                  self.crop_or_pad(item[1], num_frame),
                  item[2]) for item in batch]
        return default_collate(batch)

    def count_faces(self):
        '''
        Count the number of faces in the dataset
        '''
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            # Face Image
            image_dir = os.path.join(
                self.data_dir, sub_dataset, self.face_dir, name)
            cur_count = len(os.listdir(image_dir))
            total_count = total_count + cur_count
        print('Number of faces in current dataset: {}'.format(total_count))
        return total_count

    def count_speech(self):
        '''
        Given a id, return all the speech segments of him as a batch tensor,
        with shape (N, C, L)
        '''
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            window_length, stride_length = self.mel_seg_window_stride
            # Mel Spectrogram
            mel_gram_dir = os.path.join(
                self.data_dir, sub_dataset, 'mel_spectrograms', name)
            mel_gram_list = os.listdir(mel_gram_dir)
            cur_count = len(mel_gram_list)
            total_count = total_count + cur_count
        print('Number of speech in current dataset: {}'.format(total_count))
        return total_count


if __name__ == '__main__':
    '''
    from utils import imagenet_deprocess, deprocess_and_save

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
    print('np_mel.shape:', np_mel.shape)

    print('dataset length:', len(vox_dataset))

    # Try out 1 case
    # image, log_mel, mel_len = vox_dataset[220]
    image, log_mel, human_id = vox_dataset[220]
    print('image shape:', image.shape)
    print('log_mel shape:', log_mel.shape)
    print('human_id:', human_id)
    # print(mel_len)
    deprocess_and_save(image, image_normalize_method,
        os.path.join(test_case_dir, 'vox_dataset_image_test_1.jpg'))

    log_mel = np.array(log_mel)
    print(log_mel)
    print(np.max(log_mel), np.min(log_mel))

    # Unit test after train test split update:
    print('### Testing splitted dataset ###')
    print('Test identities:', vox_dataset.split_dict['vox1']['test'])
    '''

    for split_set in ['train', 'val', 'test']:
        vox_dataset = VoxDataset(
            data_dir=VOX_DIR,
            image_size=(64, 64),
            nframe_range=(300, 600),
            face_type='masked',
            image_normalize_method='imagenet',
            mel_normalize_method='vox_mel',
            split_set=split_set,
            split_json=os.path.join(VOX_DIR, 'split.json'))
        print('Length of {} set: {}'.format(split_set, len(vox_dataset)))
        vox_dataset.count_faces()
        vox_dataset.count_speech()

    # Collate Function dataloader test
    loader_kwargs = {
        'batch_size': 16,
        'num_workers': 8,
        'shuffle': False,
        "drop_last": True,
        'collate_fn': vox_dataset.collate_fn,
    }
    val_loader = DataLoader(vox_dataset, **loader_kwargs)
    for iter, batch in enumerate(val_loader):
        images, log_mels, human_ids = batch
        print('log_mels.shape:', log_mels.shape)
        if iter > 10000:
            break

    for e in range(1000):
        for iter, batch in enumerate(val_loader):
            images, log_mels, human_ids = batch
            print('log_mels.shape:', log_mels.shape)
            if iter > 10000:
                break

    '''

    #### Test utils imagenet_deprocess_batch and fast_imagenet_deprocess_batch
    imgs_de = imagenet_deprocess_batch(
        images,
        rescale=False,
        normalize_method='imagenet')
    imgs_de_fast = fast_imagenet_deprocess_batch(
        images,
        normalize_method='imagenet')

    print('imgs_de:', imgs_de[0])
    print('imgs_de_fast:', imgs_de_fast[0])

    #### Test utils fast_mel_deprocess_batch
    log_mels_de = fast_mel_deprocess_batch(
        log_mels,
        normalize_method='vox_mel')

    print('log_mels_de:', log_mels_de[0])

    # Test DataLoader Deepcopy
    from copy import deepcopy
    val_loader_copy = deepcopy(val_loader)
    val_loader_copy.dataset.set_length(100)
    print('length of copied val dataset: {}'.format(
        len(val_loader_copy.dataset)))
    print('length of val dataset: {}'.format(
        len(val_loader.dataset)))

    # Test get all faces
    faces = val_loader.dataset.get_all_faces_of_id(3)
    print('get_all_faces_of_id:', faces.shape)

    # Test get all mel segments
    segments = val_loader.dataset.get_all_mel_segments_of_id(3)
    print('get_all_mel_segments_of_id:', segments.shape)

    # Test return many segments mode
    val_loader.collate_fn = default_collate
    val_loader.dataset.return_mel_segments = True
    val_loader.dataset.mel_seg_window_stride = (125, 60)
    for iter, batch in enumerate(val_loader):
        images, log_mels, human_ids = batch
        print('log_mels.shape (many segments mode):', log_mels.shape)
        if iter > 10000:
            break

    '''
