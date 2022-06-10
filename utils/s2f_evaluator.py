'''
This file contains the utils for evaluation, and is used to cooperate with \
    training_utils.py and test.py
'''


import os
import json
import math
from collections import defaultdict
import time
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from torchvision.models import inception_v3
from pyprind import prog_bar
from tensorflow import gfile
from imageio import imwrite

from datasets import fast_imagenet_deprocess_batch, imagenet_deprocess_batch
import datasets
import models
from models import InceptionResnetV1, fixed_image_standardization


# left, top, right, bottom
VGG_BOX = [0.235, 0.195, 0.765, 0.915]


class S2fEvaluator:
    def __init__(self,
                 loader,
                 options,
                 nframe_range=None,
                 extraction_size=100,
                 hq_emb_dict=True,
                 face_gen_mode='naive',
                 facenet_return_pooling=False):
        '''
        Inputs:
            loader
            options
        '''
        self.loader = deepcopy(loader)
        # This makes sure same group of meg_spectrograms is used for fuser
        self.loader.dataset.shuffle_mel_segments = False
        if nframe_range is not None:
            self.loader.dataset.nframe_range = nframe_range
        self.facenet = InceptionResnetV1(
            pretrained='vggface2',
            auto_input_resize=True,
            return_pooling=facenet_return_pooling).cuda().eval()
        self.float_dtype = torch.cuda.FloatTensor
        self.long_dtype = torch.cuda.LongTensor
        self.options = options
        self.image_normalize_method= \
                self.options["data"]["data_opts"]["image_normalize_method"]
        self.do_deprocess_and_preprocess = \
            self.options["eval"]["facenet"]["deprocess_and_preprocess"]
        # crop faces according to VGG average bounding box
        self.crop_faces = \
            self.options["eval"]["facenet"]["crop_faces"]
        self.extraction_size = extraction_size
        self.hq_emb_dict = hq_emb_dict
        self.face_gen_mode = face_gen_mode

        self.get_dataset_embeddings()

        # Implement Evaluation Metrics
        self.L2_dist = \
            nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        self.L1_dist = \
            nn.PairwiseDistance(p=1.0, eps=1e-06, keepdim=False)
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def deprocess_and_preprocess(self, imgs):
        '''
        For a batch of real / generated image, perform 'imagenet' or 'standard'
        deprocess, and FaceNet Preprocess
        '''
        #print('Begin:', imgs[0])
        imgs = fast_imagenet_deprocess_batch(
            imgs,
            normalize_method=self.image_normalize_method)
        #print('Our distribution:', imgs[0])
        imgs = fixed_image_standardization(imgs)
        #print('fixed_image_standardization:', imgs[0])
        return imgs

    def crop_vgg_box(self, imgs):
        # with correct cropping & correct processing
        left, top, right, bottom = VGG_BOX
        # = [0.235015, 0.19505739, 0.76817876, 0.9154963]
        N, C, H, W = imgs.shape
        left = int(left * W)
        right = int(right * W)
        top = int(top * H)
        bottom = int(bottom * H)
        imgs = imgs[:, :, top:bottom+1, left:right+1]
        return imgs

    def get_dataset_embeddings(self):
        with torch.no_grad():
            # To avoid influence to the running_mean/var of BatchNorm Layers
            embedding_batches = []
            if self.hq_emb_dict:
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Preparing FaceNet Embedding Dictionary]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    imgs = self.loader.dataset.get_all_faces_of_id(i)
                    imgs = imgs.cuda()
                    ###################################
                    if self.do_deprocess_and_preprocess:
                        imgs = self.deprocess_and_preprocess(imgs)
                    if self.crop_faces:
                        imgs = self.crop_vgg_box(imgs)
                    embeddings = self.facenet(imgs)
                    embeddings = torch.mean(embeddings, 0, keepdim=True)
                    embedding_batches.append(embeddings)
            else:
                for batch in prog_bar(
                    self.loader,
                    title="[S2fEvaluator: " + \
                        "Preparing FaceNet Embedding Dictionary]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    imgs, log_mels, human_ids = batch
                    imgs = imgs.cuda()
                    ###################################
                    if self.do_deprocess_and_preprocess:
                        imgs = self.deprocess_and_preprocess(imgs)
                    if self.crop_faces:
                        imgs = self.crop_vgg_box(imgs)
                    embeddings = self.facenet(imgs)
                    embedding_batches.append(embeddings)
            self.dataset_embedding = torch.cat(embedding_batches, 0)
        print("S2fEvaluator: dataset_embedding shape:",
              self.dataset_embedding.shape)

    def get_pred_img_embeddings(self, model):
        '''
        This function focus on evaluating the speech-to-face specific metrics,
        Including:
            1. Deep feature (FaceNet Feature) Similarity
            2. Extraction Recall@K
            3. Landmark Correlation
        '''
        training_status = model.training
        model.eval()
        pred_img_embedding_batches = []
        with torch.no_grad():
            # To avoid influence to the running_mean/var of BatchNorm Layers
            if self.face_gen_mode == 'naive':
                for batch in prog_bar(
                    self.loader,
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    imgs, log_mels, human_ids = batch
                    imgs = imgs.cuda()
                    log_mels = log_mels.type(self.float_dtype)
                    human_ids = human_ids.type(self.long_dtype)
                    ###################################
                    # Run the model as it has been run during training
                    model_out = model(log_mels)
                    imgs_pred, others = model_out
                    # Multi-Reso Case
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                    pred_img_embeddings = self.facenet(imgs_pred)
                    pred_img_embedding_batches.append(pred_img_embeddings)
            elif self.face_gen_mode == 'average_facenet_embedding':
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images, " + \
                            "with average Facenet embedding policy]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    log_mels = self.loader.dataset.get_all_mel_segments_of_id(i)
                    log_mels = log_mels.type(self.float_dtype)
                    ###################################
                    # Run the model as it has been run during training
                    model_out = model(log_mels)
                    imgs_pred, others = model_out
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                    pred_img_embeddings = self.facenet(imgs_pred)
                    pred_img_embeddings = torch.mean(
                        pred_img_embeddings, 0, keepdim=True)
                    pred_img_embedding_batches.append(pred_img_embeddings)
            elif self.face_gen_mode == 'average_voice_embedding':
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images, " + \
                            "with average voice embedding policy]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    log_mels = self.loader.dataset.get_all_mel_segments_of_id(i)
                    log_mels = log_mels.type(self.float_dtype)
                    ###################################
                    # Run the model as it has been run during training
                    model_out = model(log_mels, average_voice_embedding=True)
                    imgs_pred, others = model_out
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                    pred_img_embeddings = self.facenet(imgs_pred)
                    #pred_img_embeddings = torch.mean(
                    #    pred_img_embeddings, 0, keepdim=True)
                    pred_img_embedding_batches.append(pred_img_embeddings)
            pred_img_embedding = torch.cat(pred_img_embedding_batches, 0)
        #print("S2fEvaluator: pred_img_embedding shape:",
        #      pred_img_embedding.shape)
        model.train(mode=training_status)

        return pred_img_embedding

    def L1_query(self, x, y):
        '''
        Given x, extract top K similar features from y, based on L1 distance
        x in shape (N_x, D)
        y in shape (N_y, D)
        '''
        # (N_x, D) --> (N_x, 1, D)
        x = x.unsqueeze(1)
        # Initialize: (N_x, )
        x_ids = torch.tensor(np.arange(x.shape[0])).cpu()
        # (N_y, D) --> (1, N_y, D)
        y = y.unsqueeze(0)
        # Output: (N_x, N_y, D)
        L1_table = torch.abs(x - y)
        # (N_x, N_y, D) --> (N_x, N_y)
        L1_table = torch.mean(L1_table, dim=-1)
        L1_table = torch.neg(L1_table)
        # Top K: (N_x, K)
        top_1_vals, top_1_indices = torch.topk(L1_table, 1, dim=-1)
        top_5_vals, top_5_indices = torch.topk(L1_table, 5, dim=-1)
        top_10_vals, top_10_indices = torch.topk(L1_table, 10, dim=-1)
        top_50_vals, top_50_indices = torch.topk(L1_table, 50, dim=-1)

        recall_at_1 = self.in_top_k(top_1_indices.cpu(), x_ids)
        recall_at_5 = self.in_top_k(top_5_indices.cpu(), x_ids)
        recall_at_10 = self.in_top_k(top_10_indices.cpu(), x_ids)
        recall_at_50 = self.in_top_k(top_50_indices.cpu(), x_ids)

        return recall_at_1, recall_at_5, recall_at_10, recall_at_50

    def cos_query(self, x, y):
        '''
        Given x, extract top K similar features from y, based on L1 distance
        x in shape (N_x, D)
        y in shape (N_y, D)
        '''
        # (N_x, D) --> (N_x, 1, D)
        x = x.unsqueeze(1)
        # Initialize: (N_x, )
        x_ids = torch.tensor(np.arange(x.shape[0])).cpu()
        # (N_y, D) --> (1, N_y, D)
        y = y.unsqueeze(0)
        # Output: (N_x, N_y)
        cos_table = self.cos_sim(x, y)

        # Top K: (N_x, K)
        top_1_vals, top_1_indices = torch.topk(cos_table, 1, dim=-1)
        top_2_vals, top_2_indices = torch.topk(cos_table, 2, dim=-1)
        top_5_vals, top_5_indices = torch.topk(cos_table, 5, dim=-1)
        top_10_vals, top_10_indices = torch.topk(cos_table, 10, dim=-1)
        top_20_vals, top_20_indices = torch.topk(cos_table, 20, dim=-1)
        top_50_vals, top_50_indices = torch.topk(cos_table, 50, dim=-1)

        recall_at_1 = self.in_top_k(top_1_indices.cpu(), x_ids)
        recall_at_2 = self.in_top_k(top_2_indices.cpu(), x_ids)
        recall_at_5 = self.in_top_k(top_5_indices.cpu(), x_ids)
        recall_at_10 = self.in_top_k(top_10_indices.cpu(), x_ids)
        recall_at_20 = self.in_top_k(top_20_indices.cpu(), x_ids)
        recall_at_50 = self.in_top_k(top_50_indices.cpu(), x_ids)

        recall_tuple = (recall_at_1, recall_at_2, recall_at_5, recall_at_10, \
            recall_at_20, recall_at_50)

        return recall_tuple

    def cal_ih_sim(self, x, y):
        '''
        For an existing face embedding distribution, calculate the inter-human
        similarity

        Arguments:
            x in shape (N_x, D)
            x in shape (N_y, D)
        '''
        # (N_x, D) --> (N_x, 1, D)
        y = y.unsqueeze(0)
        x = x.unsqueeze(1)
        # Output: (N_x, N_y)
        cos_table = self.cos_sim(x, y)
        cos_table = cos_table.detach().cpu().numpy()
        ih_sum = 0.0
        for i in range(cos_table.shape[0]):
            for j in range(cos_table.shape[1]):
                if i != j:
                    ih_sum = ih_sum + cos_table[i, j]
        ih_sim = ih_sum / float(cos_table.shape[0] * (cos_table.shape[1] - 1))
        return ih_sim

    def in_top_k(self, top_k_indices, gt_labels):
        results = []
        for i, top_k_id in enumerate(top_k_indices):
            gt_label = gt_labels[i]
            #print(gt_label, top_k_id)
            if gt_label in top_k_id:
                results.append(1.0)
            else:
                results.append(0.0)
            #print('results:', results)
        return np.mean(results)

    def get_metrics(self, model, recall_method='cos_sim', get_ih_sim=False):
        '''
        Arguments:
            recall_method: the similarity metric to use for retrival
            get_ih_sim: get the inter-human similarity
        '''
        pred_img_embedding = self.get_pred_img_embeddings(model)
        L2_dist = self.L2_dist(self.dataset_embedding, pred_img_embedding)
        L2_dist = torch.mean(L2_dist).item()
        L1_dist = self.L1_dist(self.dataset_embedding, pred_img_embedding)
        L1_dist = torch.mean(L1_dist).item()
        cos_sim = self.cos_sim(self.dataset_embedding, pred_img_embedding)
        cos_sim = torch.mean(cos_sim).item()

        # We might use only 100 id for extraction
        if self.extraction_size is None:
            pred_emb_to_use = pred_img_embedding
            data_emb_to_use = self.dataset_embedding
        elif isinstance(self.extraction_size, list):
            pred_emb_to_use = [pred_img_embedding[0:self.extraction_size[0]], \
                pred_img_embedding[self.extraction_size[0]:self.extraction_size[1]], \
                pred_img_embedding[self.extraction_size[1]:self.extraction_size[2]]]
            data_emb_to_use = [self.dataset_embedding[0:self.extraction_size[0]], \
                self.dataset_embedding[self.extraction_size[0]:self.extraction_size[1]], \
                self.dataset_embedding[self.extraction_size[1]:self.extraction_size[2]]]
        else:
            pred_emb_to_use = pred_img_embedding[0:self.extraction_size]
            data_emb_to_use = self.dataset_embedding[0:self.extraction_size]

        if recall_method == 'L1':
            if isinstance(pred_emb_to_use, list):
                recall_temp = []
                for i, pred_emb in enumerate(pred_emb_to_use):
                    recall_temp.append(self.L1_query(pred_emb, data_emb_to_use[i]))
                recall_tuple = tuple(np.mean(np.array(recall_temp), axis=0))
            else:
                recall_tuple = self.L1_query(pred_emb_to_use, data_emb_to_use)
        elif recall_method == 'cos_sim':
            if isinstance(pred_emb_to_use, list):
                recall_temp = []
                for i, pred_emb in enumerate(pred_emb_to_use):
                    recall_temp.append(self.cos_query(pred_emb, data_emb_to_use[i]))
                recall_tuple = tuple(np.mean(np.array(recall_temp), axis=0))
            else:
                recall_tuple = self.cos_query(pred_emb_to_use, data_emb_to_use)

        if get_ih_sim:
            ih_sim = self.cal_ih_sim(pred_img_embedding, self.dataset_embedding)
            return L2_dist, L1_dist, cos_sim, recall_tuple, ih_sim
        else:
            return L2_dist, L1_dist, cos_sim, recall_tuple

    def get_faces_from_different_segments(self, model, output_dir):
        #gfile.MkDir(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        temp_loader = deepcopy(self.loader)
        for i in prog_bar(
            range(len(self.loader.dataset)),
            title="[S2fEvaluator: " + \
                "Generating Faces from Different Speech Segments]",
            width=50):
            ######### unpack the data #########
            imgs = temp_loader.dataset.get_all_faces_of_id(i)
            log_mels = temp_loader.dataset.get_all_mel_segments_of_id(i)
            imgs = imgs.cuda()
            log_mels = log_mels.type(self.float_dtype) #cuda()
            #human_ids = human_ids.type(long_dtype)
            ###################################
            with torch.no_grad():
                model_out = model(log_mels)
            imgs_pred, _ = model_out
            if isinstance(imgs_pred, tuple):
                imgs_pred = imgs_pred[-1]
            imgs = imagenet_deprocess_batch(
                imgs, normalize_method=self.image_normalize_method)
            imgs_pred = imagenet_deprocess_batch(
                imgs_pred, normalize_method=self.image_normalize_method)
            #print(imgs.shape, imgs_pred.shape)
            identity_dir = os.path.join(output_dir, str(i))
            gfile.MkDir(identity_dir)

            for j in range(imgs.shape[0]):
                img_np = imgs[j].numpy().transpose(1, 2, 0)
                img_path = os.path.join(identity_dir, 'origin_%d.png' % j)
                imwrite(img_path, img_np)

            for k in range(imgs_pred.shape[0]):
                img_np = imgs_pred[k].numpy().transpose(1, 2, 0)
                img_path = os.path.join(identity_dir, 'pred_%d.png' % k)
                imwrite(img_path, img_np)

    def L2_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        if y is not None:
             differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances
