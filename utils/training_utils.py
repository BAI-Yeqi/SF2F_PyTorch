import os
import json
import math
from collections import defaultdict
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from torchvision.models import inception_v3
from pyprind import prog_bar

from datasets import imagenet_deprocess_batch, fast_mel_deprocess_batch
import datasets
import models
from utils.metrics import jaccard
from utils import tensor2im
from utils.visualization.plot import get_np_plot, plot_mel_spectrogram, \
    plot_attention

from scripts.compute_vggface_score import get_vggface_score


def add_loss_with_tensor(total_loss,
                         curr_loss,
                         loss_dict,
                         loss_name,
                         weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def visualize_sample(model,
                     imgs,
                     log_mels,
                     image_normalize_method,
                     mel_normalize_method='vox_mel',
                     visualize_attn=False):
    '''
    Prepare data for tensorboard
    '''
    samples = []
    # add the ground-truth images
    samples.append(imgs[:1])

    with torch.no_grad():
        model_out = model(log_mels)
        imgs_pred, others = model_out
        # add the reconstructed images
        if isinstance(imgs_pred, tuple):
            if len(imgs_pred) >= 2:
                low_imgs_pred = imgs_pred[0]
                # Rescale
                while low_imgs_pred.size()[2] != imgs_pred[-1].size()[2]:
                    low_imgs_pred = F.interpolate(
                        low_imgs_pred, scale_factor=2, mode='nearest')
                samples.append(low_imgs_pred[:1])
            if len(imgs_pred) == 3:
                mid_imgs_pred = imgs_pred[1]
                while mid_imgs_pred.size()[2] != imgs_pred[-1].size()[2]:
                    mid_imgs_pred = F.interpolate(
                        mid_imgs_pred, scale_factor=2, mode='nearest')
                samples.append(mid_imgs_pred[:1])
            imgs_pred = imgs_pred[-1]
        samples.append(imgs_pred[:1])

        log_mels_de = fast_mel_deprocess_batch(log_mels, mel_normalize_method)
        log_mel = log_mels_de[0].cpu().detach().numpy()

        if visualize_attn:
            attn_weights = model.module.decoder.return_attn_weights()
            attn_weight = attn_weights[0].cpu().detach().numpy()

    samples = torch.cat(samples, dim=3)
    samples = {
            "samples": tensor2im(
                imagenet_deprocess_batch(
                    samples,
                    rescale=False,
                    normalize_method=image_normalize_method
                ).squeeze(0)
            ),
            #"mel_spectrogram": get_np_plot(
            #    plot_mel_spectrogram(
            #        log_mel)
            #)
    }
    if visualize_attn:
        samples['attention_weights'] = get_np_plot(
            plot_attention(attn_weight))

    # Draw Scene Graphs
    #sg_array = draw_scene_graph(objs[obj_to_img == 0],
    #                 triples[triple_to_img == 0],
    #                 vocab=vocab)
    #samples["scene_graph"] = sg_array

    return samples


def check_model(args,
                options,
                t,
                loader,
                model):
    training_status = model.training
    model.eval()
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    num_samples = 0
    all_losses = defaultdict(list)
    inception_module = nn.DataParallel(
        inception_v3(
            pretrained=True,
            transform_input=False).cuda())
    inception_module.eval()
    preds = []
    images = []
    with torch.no_grad():
        # To avoid influence to the running_mean/var of BatchNorm Layers
        for batch in prog_bar(
            loader,
            title="[Validating Inception Score and Pixel Loss]",
            width=50):
            # Loop logic
            ######### unpack the data #########
            imgs, log_mels, human_ids = batch
            imgs = imgs.cuda()
            log_mels = log_mels.type(float_dtype)
            human_ids = human_ids.type(long_dtype)
            ###################################
            # Run the model as it has been run during training
            model_out = model(log_mels)
            imgs_pred, others = model_out

            skip_pixel_loss = False
            total_loss, losses = calculate_model_losses(
                options["optim"],
                skip_pixel_loss,
                imgs,
                imgs_pred,
                get_item=True)

            if isinstance(imgs_pred, tuple):
                imgs_pred = imgs_pred[-1]
            image = imagenet_deprocess_batch(
                imgs_pred,
                normalize_method=options["data"]["data_opts"]["image_normalize_method"])
            for i in range(image.shape[0]):
                img_np = image[i].numpy().transpose(1, 2, 0)
                images.append(img_np)

            # check inception scores
            x = F.interpolate(imgs_pred, (299, 299), mode="bilinear")
            x = inception_module(x)
            preds.append(F.softmax(x).cpu().numpy())

            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)

        samples = visualize_sample(
            model,
            imgs,
            log_mels,
            options["data"]["data_opts"]["image_normalize_method"],
            visualize_attn=options['eval'].get('visualize_attn', False))
        mean_losses = {k: np.mean(v) for k, v in all_losses.items()}

        # calculate the inception scores
        splits = 5
        preds = np.concatenate(preds, axis=0)
        # Now compute the mean kl-div
        split_scores = []
        N = preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        inception_score = (np.mean(split_scores), np.std(split_scores))

    vf_score = get_vggface_score(images)

    out = [mean_losses, samples, inception_score, vf_score]
    model.train(mode=training_status)
    return tuple(out)


def calculate_model_losses(opts,
                           skip_pixel_loss,
                           imgs,
                           imgs_pred,
                           get_item=False):
    if get_item:
        add_loss_fn = add_loss
    else:
        add_loss_fn = add_loss_with_tensor
    total_loss = torch.zeros(1).to(imgs)
    losses = {}

    l1_pixel_weight = opts["l1_pixel_loss_weight"]
    if skip_pixel_loss:
        l1_pixel_weight = 0

    if isinstance(imgs_pred, tuple):
        # Multi-Resolution Pixel Loss
        for i, img_pred in enumerate(imgs_pred):
            loss_name = 'L1_pixel_loss_%d' % i
            img = imgs
            while img.size()[2] != img_pred.size()[2]:
                img = F.interpolate(img, scale_factor=0.5, mode='nearest')
            l1_pixel_loss = F.l1_loss(img_pred, img)
            total_loss = add_loss_fn(
                total_loss,
                l1_pixel_loss,
                losses,
                loss_name,
                l1_pixel_weight)
    else:
        # Single Resolution
        l1_pixel_loss = F.l1_loss(imgs_pred, imgs)
        total_loss = add_loss_fn(
            total_loss,
            l1_pixel_loss,
            losses,
            'L1_pixel_loss',
            l1_pixel_weight)

    return total_loss, losses
