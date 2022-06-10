import json
import os
import os.path as osp
import numpy as np
from datasets import VoxDataset
from torch.utils.data import DataLoader


VOX_DIR = os.path.join('./data', 'VoxCeleb')


def build_vox_dsets(data_opts, batch_size, image_size):
    dset_kwargs = {
        'data_dir': osp.join(data_opts['root_dir']),
        'image_size': image_size,
        'face_type': data_opts.get('face_type', 'masked'),
        'image_normalize_method': \
            data_opts.get('image_normalize_method', 'imagenet'),
        'mel_normalize_method': \
            data_opts.get('mel_normalize_method', 'vox_mel'),
        'split_set': 'train',
        'split_json': \
            data_opts.get('split_json', os.path.join(VOX_DIR, 'split.json'))
    }
    train_dset = VoxDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    # make some modification to dset_kwargs so that it can be used to create val_dset
    # dset_kwargs['max_samples'] = data_opts["num_val_samples"]
    # val_dset = visual_genome(**dset_kwargs)
    dset_kwargs['split_set'] = 'val'
    val_dset = VoxDataset(**dset_kwargs)

    dset_kwargs['split_set'] = 'test'
    test_dset = VoxDataset(**dset_kwargs)

    return train_dset, val_dset, test_dset


def build_dataset(opts):
    if opts["dataset"] == "vox":
        return build_vox_dsets(opts["data_opts"], opts["batch_size"],
                              opts["image_size"])
    else:
        raise ValueError("Unrecognized dataset: {}".format(opts["dataset"]))


def build_loaders(opts):
    train_dset, val_dset, test_dset = build_dataset(opts)

    loader_kwargs = {
        'batch_size': opts["batch_size"],
        'num_workers': opts["workers"],
        'shuffle': True,
        "drop_last": True,
        'collate_fn': train_dset.collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    val_loader = DataLoader(val_dset, **loader_kwargs)
    test_loader = DataLoader(test_dset, **loader_kwargs)
    return train_loader, val_loader, test_loader
