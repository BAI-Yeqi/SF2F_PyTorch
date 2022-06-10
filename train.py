import functools
import os
import json
import math
from collections import defaultdict
import random
import time
import pyprind
import glog as log
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from datasets import imagenet_deprocess_batch
import datasets
import models
import models.perceptual
from utils.losses import get_gan_losses
from utils import timeit, LossManager
from options.opts import args, options
from utils.logger import Logger
from utils import tensor2im
from utils.utils import load_my_state_dict
# losseds need to be modified
from utils.training_utils import add_loss, check_model, calculate_model_losses
from utils.training_utils import visualize_sample
from utils.evaluate import evaluate
from utils.evaluate_fid import evaluate_fid
from utils.s2f_evaluator import S2fEvaluator
torch.backends.cudnn.benchmark = True


def main():
    global args, options
    print(args)
    print(options['data'])
    #visualize_attn=options['eval'].get('visualize_attn', False)
    #print('########### visualize_attn: {} ###########'.format(visualize_attn))
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    log.info("Building loader...")
    train_loader, val_loader, test_loader = \
        datasets.build_loaders(options["data"])
    # Fuser Logic
    if args.train_fuser_only:
        train_loader.collate_fn = default_collate
        train_loader.dataset.return_mel_segments = True
        train_loader.dataset.mel_segments_rand_start = True
        val_loader.collate_fn = default_collate
        val_loader.dataset.return_mel_segments = True
        s2f_face_gen_mode = 'naive'
    else:
        s2f_face_gen_mode = 'average_facenet_embedding'
    # End of fuser logic
    #s2f_val_evaluator = S2fEvaluator(val_loader, options)
    s2f_val_evaluator = S2fEvaluator(
        val_loader,
        options,
        extraction_size=[100,200,300],
        hq_emb_dict=True,
        face_gen_mode=s2f_face_gen_mode)
    # Get total number of people in training set
    num_train_id = len(train_loader.dataset)
    # Initialize softmax neuron value
    for ac_net in ['identity', 'identity_low', 'identity_mid', 'identity_high']:
        if options['discriminator'].get(ac_net) is not None:
            options['discriminator'][ac_net]['num_id'] = num_train_id

    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],
        image_size=options["data"]["image_size"],
        checkpoint_start_from=args.checkpoint_start_from)
    model.type(float_dtype)
    # Fuser logic
    if args.train_fuser_only:
        # Hardcode to freeze batchnorm layers
        model.eval()
        if args.train_fuser_decoder:
            model.encoder.train_fuser_only()
        else:
            model.train_fuser_only()
    # End of fuser logic
    print(model)

    optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.learning_rate,
            betas=(args.beta1, 0.999),)

    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"]["d_img_weight"] < 0):
        # Ignore image discriminator
        img_discriminator = None
        d_img_kwargs = {}
        log.info("Ignoring Image Discriminator.")
    else:
        img_discriminator, d_img_kwargs = models.build_img_discriminator(
            options["discriminator"])
        log.info("Done Building Image Discriminator.")

    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"]["ac_loss_weight"] < 0):
        # Ignore AC discriminator
        ac_discriminator = None
        ac_img_kwargs = {}
        log.info("Ignoring Auxilary Classifier Discriminator.")
    else:
        ac_discriminator, ac_img_kwargs = models.build_ac_discriminator(
            options["discriminator"])
        #print('AC Discriminator:', ac_discriminator)
        log.info("Done Building Auxilary Classifier Discriminator.")

    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"].get("cond_loss_weight", -1) < 0):
        # Ignore Conditional discriminator
        cond_discriminator = None
        cond_d_kwargs = {}
        log.info("Ignoring Conditional Discriminator.")
    else:
        cond_discriminator, cond_d_kwargs = models.build_cond_discriminator(
            options["discriminator"])
        #print('Conditional Discriminator:', cond_discriminator)
        log.info("Done Building Conditional Discriminator.")

    perceptual_module = None
    if options["optim"].get("perceptual_loss_weight", -1) > 0:
        ploss_name = options.get("perceptual", {}).get("arch", "FaceNetLoss")
        ploss_cos_weight = options["optim"].get("cos_percept_loss_weight", -1)
        perceptual_module = getattr(
            models.perceptual,
            ploss_name)(cos_loss_weight=ploss_cos_weight)
        log.info("Done Building Perceptual {} Module.".format(ploss_name))
        if ploss_cos_weight > 0:
            log.info("Perceptual Cos Loss Weight: {}".format(ploss_cos_weight))
    else:
        log.info("Ignoring Perceptual Module.")

    gan_g_loss, gan_d_loss = get_gan_losses(options["optim"]["gan_loss_type"])

    optimizer_d_img = []
    if img_discriminator is not None:
        for i in range(len(img_discriminator)):
            img_discriminator[i].type(float_dtype)
            img_discriminator[i].train()
            print(img_discriminator[i])
            optimizer_d_img.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           img_discriminator[i].parameters()),
                    lr=args.learning_rate,
                    betas=(args.beta1, 0.999),))

    optimizer_d_ac = []
    if ac_discriminator is not None:
        for i in range(len(ac_discriminator)):
            ac_discriminator[i].type(float_dtype)
            ac_discriminator[i].train()
            print(ac_discriminator[i])
            optimizer_d_ac.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           ac_discriminator[i].parameters()),
                    lr=args.learning_rate,
                    betas=(args.beta1, 0.999),))

    optimizer_cond_d = []
    if cond_discriminator is not None:
        for i in range(len(cond_discriminator)):
            cond_discriminator[i].type(float_dtype)
            cond_discriminator[i].train()
            print(cond_discriminator[i])
            optimizer_cond_d.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           cond_discriminator[i].parameters()),
                    lr=args.learning_rate,
                    betas=(args.beta1, 0.999),))

    restore_path = None
    if args.resume is not None:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(
            options["logs"]["output_dir"], args.resume, restore_path)

    if restore_path is not None and os.path.isfile(restore_path):
        log.info('Restoring from checkpoint: {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])

        if img_discriminator is not None:
            for i in range(len(img_discriminator)):
                term_name = 'd_img_state_%d' % i
                img_discriminator[i].load_state_dict(checkpoint[term_name])
                term_name = 'd_img_optim_state_%d' % i
                optimizer_d_img[i].load_state_dict(checkpoint[term_name])

        if ac_discriminator is not None:
            for i in range(len(ac_discriminator)):
                term_name = 'd_ac_state_%d' % i
                ac_discriminator[i].load_state_dict(checkpoint[term_name])
                term_name = 'd_ac_optim_state_%d' % i
                optimizer_d_ac[i].load_state_dict(checkpoint[term_name])

        if cond_discriminator is not None:
            for i in range(len(cond_discriminator)):
                term_name = 'd_img_state_%d' % i
                cond_discriminator[i].load_state_dict(checkpoint[term_name])
                term_name = 'd_img_optim_state_%d' % i
                optimizer_cond_d[i].load_state_dict(checkpoint[term_name])

        t = checkpoint['counters']['t'] + 1
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()
        # Reset epoch here later
        start_epoch = checkpoint['counters']['epoch'] + 1
        log_path = os.path.join(options["logs"]["output_dir"], args.resume,)
        lr = checkpoint.get('learning_rate', args.learning_rate)
        best_inception = checkpoint["counters"].get("best_inception", (0., 0.))
        best_vfs = checkpoint["counters"].get("best_vfs", (0., 0.))
        best_recall_1 = checkpoint["counters"].get("best_recall_1", 0.)
        best_recall_5 = checkpoint["counters"].get("best_recall_5", 0.)
        best_recall_10 = checkpoint["counters"].get("best_recall_10", 0.)
        best_cos = checkpoint["counters"].get("best_cos", 0.)
        best_L1 = checkpoint["counters"].get("best_L1", 100000.0)
        options = checkpoint.get("options", options)
    else:
        t, start_epoch, best_inception, best_vfs = 0, 0, (0., 0.), (0., 0.)
        best_recall_1, best_recall_5, best_recall_10 = 0.0, 0.0, 0.0
        best_cos, best_L1 = 0.0, 100000.0
        lr = args.learning_rate
        checkpoint = {
            'args': args.__dict__,
            'options': options,
            'model_kwargs': model_kwargs,
            'd_img_kwargs': d_img_kwargs,
            'train_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'train_inception': [],
            'lr': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
                'best_inception': None,
                'best_vfs': None,
                'best_recall_1': None,
                'best_recall_5': None,
                'best_recall_10': None,
                'best_cos': None,
                'best_L1': None,
            },
            'model_state': None,
            'model_best_state': None,
            'optim_state': None,
            'd_img_state': None,
            'd_img_best_state': None,
            'd_img_optim_state': None,
            'd_ac_state': None,
            'd_ac_optim_state': None,
        }

        log_path = os.path.join(
            options["logs"]["output_dir"],
            options["logs"]["name"] + "-" + time.strftime("%Y%m%d-%H%M%S")
        )

    ### Fuser Logic
    if args.pretrained_path is not None and \
        os.path.isfile(args.pretrained_path):
        # Load
        log.info('Loading Pretrained Model: {}'.format(args.pretrained_path))
        pre_checkpoint = torch.load(args.pretrained_path)
        #model.load_state_dict(pre_checkpoint['model_state'])
        load_my_state_dict(model, pre_checkpoint['model_state'])
        #optimizer.load_state_dict(pre_checkpoint['optim_state'])

        if img_discriminator is not None:
            for i in range(len(img_discriminator)):
                term_name = 'd_img_state_%d' % i
                img_discriminator[i].load_state_dict(pre_checkpoint[term_name])
                #term_name = 'd_img_optim_state_%d' % i
                #optimizer_d_img[i].load_state_dict(pre_checkpoint[term_name])

        if ac_discriminator is not None:
            for i in range(len(ac_discriminator)):
                term_name = 'd_ac_state_%d' % i
                ac_discriminator[i].load_state_dict(pre_checkpoint[term_name])
                #term_name = 'd_ac_optim_state_%d' % i
                #optimizer_d_ac[i].load_state_dict(pre_checkpoint[term_name])
    ### End of fuser logic
    logger = Logger(log_path)
    log.info("Logging to: {}".format(log_path))
    # save the current config yaml
    copyfile(args.path_opts,
             os.path.join(log_path, options["logs"]["name"] + '.yaml'))

    model = nn.DataParallel(model.cuda())
    if ac_discriminator is not None:
        for i in range(len(ac_discriminator)):
            ac_discriminator[i] = nn.DataParallel(ac_discriminator[i].cuda())
    if img_discriminator is not None:
        for i in range(len(img_discriminator)):
            img_discriminator[i] = nn.DataParallel(img_discriminator[i].cuda())
    if cond_discriminator is not None:
        for i in range(len(cond_discriminator)):
            cond_discriminator[i] = nn.DataParallel(cond_discriminator[i].cuda())
    perceptual_module = nn.DataParallel(perceptual_module.cuda()) if \
        perceptual_module else None

    if args.evaluate:
        assert args.resume is not None
        if args.evaluate_train:
            log.info("Evaluting the training set.")
            train_mean, train_std, train_vfs_mean, train_vfs_std = \
                evaluate(model, train_loader, options)
            log.info("Inception score: {} ({})".format(train_mean, train_std))
            log.info("VggFace score: {} ({})".format(
                train_vfs_mean, train_vfs_std))
        log.info("Evaluting the validation set.")
        val_mean, val_std, vfs_mean, vfs_std = evaluate(
            model, val_loader, options)
        log.info("Inception score: {} ({})".format(val_mean, val_std))
        log.info("VggFace score: {} ({})".format(vfs_mean, vfs_std))
        fid_score = evaluate_fid(model, val_loader, options)
        log.info("FID score: {}".format(fid_score))
        return 0


    got_best_IS = True
    got_best_VFS = True
    got_best_R1 = True
    got_best_R5 = True
    got_best_R10 = True
    got_best_cos = True
    got_best_L1 = True
    others = None
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.eval_mode_after and model.training:
            log.info('[Epoch {}/{}] switching to eval mode'.format(
                epoch, args.epochs))
            model.eval()
            if epoch == args.eval_mode_after:
                optimizer = optim.Adam(
                        filter(lambda x: x.requires_grad, model.parameters()),
                        lr=lr,
                        betas=(args.beta1, 0.999),)
        if epoch >= args.disable_l1_loss_after and \
            options["optim"]["l1_pixel_loss_weight"] > 1e-10:
            #
            log.info('[Epoch {}/{}] Disable L1 Loss'.format(epoch, args.epochs))
            options["optim"]["l1_pixel_loss_weight"] = 0
        start_time = time.time()
        for iter, batch in enumerate(pyprind.prog_bar(
            train_loader,
            title="[Epoch {}/{}]".format(epoch, args.epochs),
            width=50)):
            # An iteration
            if args.timing:
                print("Loading Time: {} ms".format(
                    (time.time() - start_time) * 1000))
            t += 1
            ######### unpack the data #########
            imgs, log_mels, human_ids = batch
            imgs = imgs.cuda()
            log_mels = log_mels.type(float_dtype)
            human_ids = human_ids.type(long_dtype)
            ###################################
            with timeit('forward', args.timing):
                #model_out = model(imgs)
                model_out = model(log_mels)
                """
                imgs_pred: generated images
                others: placeholder for other output
                """
                imgs_pred, others = model_out

            if t % args.visualize_every == 0:
                #print('Performing Visualization...')
                training_status = model.training
                model.eval()
                samples = visualize_sample(
                    model,
                    imgs,
                    log_mels,
                    options["data"]["data_opts"]["image_normalize_method"],
                    visualize_attn=options['eval'].get('visualize_attn', False))
                model.train(mode=training_status)
                logger.image_summary(samples, t, tag="vis")

            with timeit('G_loss', args.timing):
                #model_boxes = None
                # Skip the pixel loss if not using GT boxes
                skip_pixel_loss = False

                # calculate L1 loss between imgs and imgs_self
                total_loss, losses = calculate_model_losses(
                    options["optim"], skip_pixel_loss, imgs, imgs_pred,)

                if img_discriminator is not None:
                    for i in range(len(img_discriminator)):
                        # TODO: Here we need to choose the two methods:
                        # 1. determine whether imgs_pred is tuple or not
                        # 2. change the return (imgs_pred) of model to be tuple
                        if isinstance(imgs_pred, tuple):
                            scores_fake = img_discriminator[i](imgs_pred[i])
                        else:
                            scores_fake = img_discriminator[i](imgs_pred)
                        weight = options["optim"]["d_loss_weight"] * \
                            options["optim"]["d_img_weight"]
                        loss_name = 'g_gan_img_loss_%d' % i
                        total_loss = add_loss(total_loss,
                            gan_g_loss(scores_fake), losses, loss_name, weight)

                if ac_discriminator is not None:
                    for i in range(len(ac_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            scores_fake, ac_loss = ac_discriminator[i](
                                imgs_pred[i], human_ids)
                        else:
                            scores_fake, ac_loss = ac_discriminator[i](
                                imgs_pred, human_ids)
                        weight = options["optim"]["d_loss_weight"] * \
                            options["optim"]["ac_loss_weight"]
                        loss_name = 'ac_loss_%d' % i
                        total_loss = add_loss(total_loss,
                                              ac_loss.mean(),
                                              losses,
                                              loss_name,
                                              weight)
                        loss_name = 'g_gan_ac_loss_%d' % i
                        total_loss = add_loss(total_loss,
                                              gan_g_loss(scores_fake),
                                              losses,
                                              loss_name,
                                              weight)

                if cond_discriminator is not None:
                    for i in range(len(cond_discriminator)):
                        # TODO: check whether condition is in others or not ?
                        cond_vecs = others['cond']
                        if isinstance(imgs_pred, tuple):
                            scores_fake = cond_discriminator[i](
                                imgs_pred[i], cond_vecs)
                        else:
                            scores_fake = cond_discriminator[i](
                                imgs_pred, cond_vecs)
                        weight = options["optim"]["d_loss_weight"] * \
                            options["optim"]["cond_loss_weight"]
                        loss_name = 'g_cond_loss_%d' % i
                        total_loss = add_loss(total_loss,
                            gan_g_loss(scores_fake), losses, loss_name, weight)

                if options["optim"].get("perceptual_loss_weight", -1) > 0:
                    if isinstance(imgs_pred, tuple):
                        # Multi-Resolution Pixel Loss
                        for i, img_pred in enumerate(imgs_pred):
                            loss_name = 'img_perceptual_loss_%d' % i
                            img = imgs
                            while img.size()[2] != img_pred.size()[2]:
                                img = F.interpolate(
                                    img, scale_factor=0.5, mode='nearest')
                            if s2f_val_evaluator.do_deprocess_and_preprocess:
                                img_pred_in = \
                                    s2f_val_evaluator.deprocess_and_preprocess(
                                        img_pred)
                                img_in = \
                                    s2f_val_evaluator.deprocess_and_preprocess(
                                        img)
                            if s2f_val_evaluator.crop_faces:
                                img_pred_in = \
                                    s2f_val_evaluator.crop_vgg_box(img_pred_in)
                                img_in = s2f_val_evaluator.crop_vgg_box(img_in)
                            perceptual_loss = perceptual_module(
                                img_pred_in, img_in)
                            perceptual_loss = perceptual_loss.mean()
                            weight = options["optim"]["perceptual_loss_weight"]
                            total_loss = add_loss(total_loss, perceptual_loss,
                                                  losses, loss_name,
                                                  weight)
                    else:
                        if s2f_val_evaluator.do_deprocess_and_preprocess:
                            imgs_pred_in = \
                                s2f_val_evaluator.deprocess_and_preprocess(
                                    imgs_pred)
                            imgs_in = \
                                s2f_val_evaluator.deprocess_and_preprocess(imgs)
                        if s2f_val_evaluator.crop_faces:
                            imgs_pred_in = \
                                s2f_val_evaluator.crop_vgg_box(imgs_pred_in)
                            imgs_in = s2f_val_evaluator.crop_vgg_box(imgs_in)
                        perceptual_loss = perceptual_module(
                            imgs_pred_in, imgs_in)
                        perceptual_loss = perceptual_loss.mean()
                        weight = options["optim"]["perceptual_loss_weight"]
                        total_loss = add_loss(total_loss, perceptual_loss,
                                              losses, "img_perceptual_loss",
                                              weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                log.warn('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()

            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}

            with timeit('D_loss', args.timing):
                if img_discriminator is not None:
                    d_img_losses = LossManager()
                    for i in range(len(img_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()
                        imgs_real = imgs.detach()
                        while imgs_real.size()[2] != imgs_fake.size()[2]:
                            imgs_real = F.interpolate(
                                imgs_real, scale_factor=0.5, mode='nearest')

                        scores_fake = img_discriminator[i](imgs_fake)
                        scores_real = img_discriminator[i](imgs_real)

                        d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
                        d_img_losses.add_loss(
                            d_img_gan_loss, 'd_img_gan_loss_%d' % i)

                    for i in range(len(img_discriminator)):
                        optimizer_d_img[i].zero_grad()
                    d_img_losses.total_loss.backward()
                    for i in range(len(img_discriminator)):
                        optimizer_d_img[i].step()

                if ac_discriminator is not None:
                    d_ac_losses = LossManager()
                    for i in range(len(ac_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()

                        imgs_real = imgs.detach()
                        while imgs_real.size()[2] != imgs_fake.size()[2]:
                            imgs_real = F.interpolate(
                                imgs_real, scale_factor=0.5, mode='nearest')

                        scores_real, ac_loss_real= ac_discriminator[i](
                            imgs_real, human_ids)
                        scores_fake, ac_loss_fake = ac_discriminator[i](
                            imgs_fake, human_ids)

                        d_ac_gan_loss = gan_d_loss(scores_real, scores_fake)
                        d_ac_losses.add_loss(
                            d_ac_gan_loss, 'd_ac_gan_loss_%d' % i)
                        d_ac_losses.add_loss(
                            ac_loss_real.mean(), 'd_ac_loss_real_%d' % i)

                    for i in range(len(ac_discriminator)):
                        optimizer_d_ac[i].zero_grad()
                    d_ac_losses.total_loss.backward()
                    for i in range(len(ac_discriminator)):
                        optimizer_d_ac[i].step()

                if cond_discriminator is not None:
                    cond_d_losses = LossManager()
                    for i in range(len(cond_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()
                        imgs_real = imgs.detach()
                        cond_vecs = others['cond'].detach()
                        while imgs_real.size()[2] != imgs_fake.size()[2]:
                            imgs_real = F.interpolate(
                                imgs_real, scale_factor=0.5, mode='nearest')

                        scores_fake = cond_discriminator[i](
                            imgs_fake, cond_vecs)
                        scores_real = cond_discriminator[i](
                            imgs_real, cond_vecs)

                        cond_d_gan_loss = gan_d_loss(scores_real, scores_fake)
                        cond_d_losses.add_loss(
                            cond_d_gan_loss, 'cond_d_gan_loss_%d' % i)

                    for i in range(len(cond_discriminator)):
                        optimizer_cond_d[i].zero_grad()
                    cond_d_losses.total_loss.backward()
                    for i in range(len(cond_discriminator)):
                        optimizer_cond_d[i].step()

            # Logging generative model losses
            for name, val in losses.items():
                logger.scalar_summary("loss/{}".format(name), val, t)
            if img_discriminator is not None:
                for name, val in d_img_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            if ac_discriminator is not None:
                for name, val in d_ac_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            if cond_discriminator is not None:
                for name, val in cond_d_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
            start_time = time.time()

        if epoch % args.eval_epochs == 0:
            log.info('[Epoch {}/{}] checking on val'.format(
                epoch, args.epochs)
            )
            val_results = check_model(
                args, options, epoch, val_loader, model)
            val_losses, val_samples, val_inception, val_vfs = val_results
            # call evaluate_s2f_metrics() here
            val_facenet_L2_dist, val_facenet_L1_dist, val_facenet_cos_sim, \
                val_recall_tuple, val_ih_sim  = \
                    s2f_val_evaluator.get_metrics(
                        model, recall_method='cos_sim', get_ih_sim=True)
            val_recall_at_1, val_recall_at_2, val_recall_at_5, \
                val_recall_at_10, val_recall_at_20, \
                    val_recall_at_50 = val_recall_tuple
            # Update the best of metrics
            if val_inception[0] > best_inception[0]:
                got_best_IS = True
                best_inception = val_inception
            if val_vfs[0] > best_vfs[0]:
                got_best_VFS = True
                best_vfs = val_vfs
            if val_recall_at_1 > best_recall_1:
                got_best_R1 = True
                best_recall_1 = val_recall_at_1
            if val_recall_at_5 > best_recall_5:
                got_best_R5 = True
                best_recall_5 = val_recall_at_5
            if val_recall_at_10 > best_recall_10:
                got_best_R10 = True
                best_recall_10 = val_recall_at_10
            if val_facenet_cos_sim > best_cos:
                got_best_cos = True
                best_cos = val_facenet_cos_sim
            if val_facenet_L1_dist < best_L1:
                got_best_L1 = True
                best_L1 = val_facenet_L1_dist
            checkpoint['counters']['best_inception'] = best_inception
            checkpoint['counters']['best_vfs'] = best_vfs
            checkpoint['val_samples'].append(val_samples)
            # checkpoint['val_batch_data'].append(val_batch_data)
            for k, v in val_losses.items():
                checkpoint['val_losses'][k].append(v)
                logger.scalar_summary("ckpt/val_{}".format(k), v, epoch)
            logger.scalar_summary("ckpt/val_inception", val_inception[0], epoch)
            logger.scalar_summary("ckpt/val_facenet_L2_dist",
                val_facenet_L2_dist, epoch)
            logger.scalar_summary("ckpt/val_facenet_L1_dist",
                val_facenet_L1_dist, epoch)
            logger.scalar_summary("ckpt/val_facenet_cos_sim",
                val_facenet_cos_sim, epoch)
            logger.scalar_summary("ckpt/val_recall_at_1",
                val_recall_at_1, epoch)
            logger.scalar_summary("ckpt/val_recall_at_2",
                val_recall_at_2, epoch)
            logger.scalar_summary("ckpt/val_recall_at_5",
                val_recall_at_5, epoch)
            logger.scalar_summary("ckpt/val_recall_at_10",
                val_recall_at_10, epoch)
            logger.scalar_summary("ckpt/val_recall_at_20",
                val_recall_at_20, epoch)
            logger.scalar_summary("ckpt/val_recall_at_50",
                val_recall_at_50, epoch)
            logger.scalar_summary("ckpt/val_ih_sim",
                val_ih_sim, epoch)
            logger.scalar_summary("ckpt/val_vfs",
                val_vfs[0], epoch)
            # Add speech2face metrics here..
            #
            logger.image_summary(val_samples, epoch, tag="ckpt_val")
            # log.info('[Epoch {}/{}] val iou: {}'.format(
            # epoch, args.epochs, val_avg_iou))
            log.info('[Epoch {}/{}] val inception score: {} ({})'.format(
                    epoch, args.epochs, val_inception[0], val_inception[1]))
            log.info('[Epoch {}/{}] best inception scores: {} ({})'.format(
                    epoch, args.epochs, best_inception[0], best_inception[1]))
            log.info('[Epoch {}/{}] val vfs scores: {} ({})'.format(
                    epoch, args.epochs, val_vfs[0], val_vfs[1]))
            log.info('[Epoch {}/{}] best vfs scores: {} ({})'.format(
                    epoch, args.epochs, best_vfs[0], best_vfs[1]))
            log.info('[Epoch {}/{}] val recall at 5: {}, '.format(
                     epoch, args.epochs, val_recall_at_5) + \
                        'best recall at 5: {}'.format(best_recall_5))
            log.info('[Epoch {}/{}] val recall at 10: {}, '.format(
                     epoch, args.epochs, val_recall_at_10) + \
                        'best recall at 10: {}'.format(best_recall_10))
            log.info('[Epoch {}/{}] val cosine similarity: {}, '.format(
                     epoch, args.epochs, val_facenet_cos_sim) + \
                        'best cosine similarity: {}'.format(best_cos))
            log.info('[Epoch {}/{}] val L1 distance: {}, '.format(
                     epoch, args.epochs, val_facenet_L1_dist) + \
                            'best L1 distance: {}'.format(best_L1))

            checkpoint['model_state'] = model.module.state_dict()

            if img_discriminator is not None:
                for i in range(len(img_discriminator)):
                    term_name = 'd_img_state_%d' % i
                    checkpoint[term_name] = \
                        img_discriminator[i].module.state_dict()
                    term_name = 'd_img_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_d_img[i].state_dict()

            if ac_discriminator is not None:
                for i in range(len(ac_discriminator)):
                    term_name = 'd_ac_state_%d' % i
                    checkpoint[term_name] = \
                        ac_discriminator[i].module.state_dict()
                    term_name = 'd_ac_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_d_ac[i].state_dict()

            if cond_discriminator is not None:
                for i in range(len(cond_discriminator)):
                    term_name = 'cond_d_state_%d' % i
                    checkpoint[term_name] = \
                        cond_discriminator[i].module.state_dict()
                    term_name = 'cond_d_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_cond_d[i].state_dict()

            checkpoint['optim_state'] = optimizer.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['counters']['t'] = t
            checkpoint['counters']['best_inception'] = best_inception
            checkpoint['counters']['best_vfs'] = best_vfs
            checkpoint['counters']['best_recall_1'] = best_recall_1
            checkpoint['counters']['best_recall_5'] = best_recall_5
            checkpoint['counters']['best_recall_10'] = best_recall_10
            checkpoint['counters']['best_cos'] = best_cos
            checkpoint['counters']['best_L1'] = best_L1
            checkpoint['lr'] = lr
            checkpoint_path = os.path.join(
                log_path,
                '%s_with_model.pt' % args.checkpoint_name)
            log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(
                epoch, args.epochs, checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            if got_best_IS:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_IS_with_model.pt'))
                got_best_IS = False
            if got_best_VFS:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_VFS_with_model.pt'))
                got_best_VFS = False
            if got_best_R1:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R1_with_model.pt'))
                got_best_R1 = False
            if got_best_R5:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R5_with_model.pt'))
                got_best_R5 = False
            if got_best_R10:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R10_with_model.pt'))
                got_best_R10 = False
            if got_best_L1:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_L1_with_model.pt'))
                got_best_L1 = False
            if got_best_cos:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_cos_with_model.pt'))
                got_best_cos = False

        if epoch > 0 and epoch % 1000 == 0:
            print('Saving checkpoint for Epoch {}.'.format(epoch))
            copyfile(
                checkpoint_path,
                os.path.join(log_path, 'epoch_{}_model.pt'.format(epoch)))
        # Fuser Logic
        elif args.train_fuser_only and epoch > 0 and epoch % 1 == 0:
            print('Saving checkpoint for Epoch {}.'.format(epoch))
            copyfile(
                checkpoint_path,
                os.path.join(log_path, 'epoch_{}_model.pt'.format(epoch)))
        # End of fuser logic

        if epoch >= args.decay_lr_epochs:
            lr_end = args.learning_rate * 1e-3
            decay_frac = (epoch - args.decay_lr_epochs + 1) / \
                (args.epochs - args.decay_lr_epochs + 1e-5)
            lr = args.learning_rate - decay_frac * (args.learning_rate - lr_end)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if img_discriminator is not None:
                for i in range(len(optimizer_d_img)):
                    for param_group in optimizer_d_img[i].param_groups:
                        param_group["lr"] = lr
                # for param_group in optimizer_d_img.param_groups:
                #     param_group["lr"] = lr
            log.info('[Epoch {}/{}] learning rate: {}'.format(
                epoch+1, args.epochs, lr))

        logger.scalar_summary("ckpt/learning_rate", lr, epoch)

    # Evaluating after the whole training process.
    log.info("Evaluting the validation set.")
    is_mean, is_std, vfs_mean, vfs_std = evaluate(model, val_loader, options)
    log.info("Inception score: {} ({})".format(is_mean, is_std))
    log.info("VggFace score: {} ({})".format(vfs_mean, vfs_std))


if __name__ == '__main__':
    main()
