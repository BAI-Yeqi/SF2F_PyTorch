import os
import pyprind
import glog as log

import torch
from torch.utils.data.dataloader import default_collate
#from scipy.misc import imresize
import pandas as pd

from datasets import imagenet_deprocess_batch
import datasets
import models
import models.perceptual
from options.opts import args, options
from utils.s2f_evaluator import S2fEvaluator

from scripts.compute_inception_score import get_inception_score
from scripts.compute_vggface_score import get_vggface_score
from scripts.compute_fid_score import calculate_activation_statistics, \
    calculate_frechet_distance
import scripts.compute_fid_score as inception_score


torch.backends.cudnn.benchmark = True


VGG_BOX = [0.235, 0.195, 0.765, 0.915]
def crop_vgg_box(imgs):
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

def load_model(options, checkpoint_start_from, checkpoint):
    output_model, output_name = [], []
    if os.path.isfile(checkpoint_start_from):
        model, _ = models.build_model(
            options["generator"],
            image_size=options["data"]["image_size"],
            checkpoint_start_from=checkpoint_start_from)
        output_model.append(model)
        output_name.append(checkpoint_start_from.split('/')[-1].split('_')[1])
    else:
        assert checkpoint
        for name in checkpoint:
            if 'epoch' in name:
                ckpt_name = '%s_model.pt' % name
            else:
                ckpt_name = 'best_%s_with_model.pt' % name
            checkpoint_path = os.path.join(checkpoint_start_from, ckpt_name)
            model, _ = models.build_model(
                options["generator"],
                image_size=options["data"]["image_size"],
                checkpoint_start_from=checkpoint_path)
            output_model.append(model)
            output_name.append(name)
    return output_model, output_name

def load_s2f(loader, options, extraction_size,
             hq_emb_dict, face_gen_mode, facenet_return_pooling):
    output, mode = [], []
    for fgm in face_gen_mode:
        evaluator = S2fEvaluator(
            loader,
            options,
            extraction_size=extraction_size,
            hq_emb_dict=hq_emb_dict,
            face_gen_mode=fgm,
            facenet_return_pooling=facenet_return_pooling)
        output.append(evaluator)
        mode.append(fgm)
    return output, mode

def main():
    global args, options
    print(args)

    device = torch.device('cuda')
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

    log.info("Building loader...")
    train_loader, val_loader, test_loader = \
        datasets.build_loaders(options["data"])
    if args.train_fuser_only:
        train_loader.collate_fn = default_collate
        train_loader.dataset.return_mel_segments = True
        val_loader.collate_fn = default_collate
        val_loader.dataset.return_mel_segments = True
        test_loader.collate_fn = default_collate
        test_loader.dataset.return_mel_segments = True

    normalize_method = options["data"]["data_opts"].get(
        'normalize_method', 'imagenet')

    log.info("Building Generative Model...")
    print(options["generator"])
    model, model_name = load_model(options,
                                   args.checkpoint_start_from,
                                   args.checkpoint)

    s2f_val_evaluator, face_gen_mode = load_s2f(test_loader,
        options,
        extraction_size=[100,200,300],
        hq_emb_dict=True,
        face_gen_mode=args.face_gen_mode,
        facenet_return_pooling=args.facenet_return_pooling)

    # init the output result
    result = {}
    result['Model'] = []
    result['Name'] = []
    result['Mode'] = []
    result['L2'] = []
    result['L1'] = []
    result['Cos'] = []
    result['Human'] = []
    result['R@1'] = []
    result['R@2'] = []
    result['R@5'] = []
    result['R@10'] = []
    result['R@20'] = []
    result['R@50'] = []
    result['IS mean'] = []
    result['IS std'] = []
    result['FID'] = []
    result['VFS mean'] = []
    result['VFS std'] = []
    exp_dir = args.checkpoint_start_from.split('/')[1]

    # Loop the face_gen_mode
    for i, mode in enumerate(face_gen_mode):
        # Loop the different model
        for j, name in enumerate(model_name):
            # print('segments_fusion', model[j].encoder.segments_fusion)
            model[j].type(float_dtype)
            # print(model[j])
            model[j].eval()
            model[j].to(device)

            if args.get_faces_from_different_segments:
                diff_seg_faces_dir = os.path.join(
                    './output', exp_dir, mode, name, 'diff_seg_faces_dir')
                s2f_val_evaluator[i].get_faces_from_different_segments(
                    model[j], diff_seg_faces_dir)
            val_facenet_L2_dist, val_facenet_L1_dist, val_facenet_cos_sim, \
                val_recall_tuple, ih_sim = \
                    s2f_val_evaluator[i].get_metrics(
                        model[j],
                        recall_method=args.recall_method,
                        get_ih_sim=True)
            val_recall_at_1, val_recall_at_2, val_recall_at_5, \
                val_recall_at_10, val_recall_at_20, \
                    val_recall_at_50 = val_recall_tuple

            print('-'*80)
            print('-'*30, 'Model Type: ', name, mode, '-'*30)
            print('val L2 Distance: ', val_facenet_L2_dist)
            print('val L1 Distance: ', val_facenet_L1_dist)
            print('val Cosine Similarity: ', val_facenet_cos_sim)
            print('val Inter-human (Cosine) Similarity: ', ih_sim)
            print('val Recall@1: ', val_recall_at_1)
            print('val Recall@2: ', val_recall_at_2)
            print('val Recall@5: ', val_recall_at_5)
            print('val Recall@10: ', val_recall_at_10)
            print('val Recall@20: ', val_recall_at_20)
            print('val Recall@50: ', val_recall_at_50)

            images = []
            images_gt = []
            for iter, batch in enumerate(pyprind.prog_bar(test_loader,
                                          title="[Generating Images]",
                                          width=50)):
                ######### unpack the data #########
                imgs, log_mels, human_ids = batch
                imgs = imgs.cuda()
                log_mels = log_mels.type(float_dtype) #cuda()
                human_ids = human_ids.type(long_dtype)
                ###################################
                with torch.no_grad():
                    model_out = model[j](log_mels)
                imgs_pred, _ = model_out

                if isinstance(imgs_pred, tuple):
                    imgs_pred = imgs_pred[-1]

                img = imagenet_deprocess_batch(
                    imgs_pred, normalize_method=normalize_method)
                # Crop the face
                #img = crop_vgg_box(img)
                for i in range(img.shape[0]):
                    img_np = img[i].numpy().transpose(1, 2, 0)
                    images.append(img_np)

                img = imagenet_deprocess_batch(
                    imgs, normalize_method=normalize_method)
                # Crop the face
                #img = crop_vgg_box(img)
                for i in range(img.shape[0]):
                    img_np = img[i].numpy().transpose(1, 2, 0)
                    images_gt.append(img_np)

            log.info("Start to compute Inception Score...")
            is_mean, is_std = get_inception_score(images)
            print('Inception Score mean: ', is_mean)
            print('Inception Score std: ', is_std)

            log.info("Start to compute FID Score...")
            acts_set = inception_score.get_fid_pred(images_gt)
            fake_acts_set = inception_score.get_fid_pred(images)
            real_mu, real_sigma = calculate_activation_statistics(acts_set)
            fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
            fid_score = calculate_frechet_distance(
                real_mu, real_sigma, fake_mu, fake_sigma)
            print('FID Score: ', fid_score)

            log.info("Start to compute VggFace Score...")
            vfs_mean, vfs_std = get_vggface_score(images)
            print('VggFace Score mean: ', vfs_mean)
            print('VggFace Score std: ', vfs_std)

            # log.info("Start to compute FVFD Score...")
            # acts_set = get_vggface_act(images_gt)
            # fake_acts_set = get_vggface_act(images)
            # real_mu, real_sigma = calculate_activation_statistics(acts_set)
            # fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
            # fvfd_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
            # print('FVFD Score: ', fvfd_score)

            print('-'*80)

            # Save the result
            result['Model'].append(exp_dir)
            result['Name'].append(name)
            result['Mode'].append(mode)
            result['L2'].append('%.3f' % val_facenet_L2_dist)
            result['L1'].append('%.3f' % val_facenet_L1_dist)
            result['Cos'].append('%.3f' % val_facenet_cos_sim)
            result['Human'].append('%.3f' % ih_sim)
            result['R@1'].append('%.3f' % val_recall_at_1)
            result['R@2'].append('%.3f' % val_recall_at_2)
            result['R@5'].append('%.3f' % val_recall_at_5)
            result['R@10'].append('%.3f' % val_recall_at_10)
            result['R@20'].append('%.3f' % val_recall_at_20)
            result['R@50'].append('%.3f' % val_recall_at_50)
            result['IS mean'].append('%.3f' % is_mean)
            result['IS std'].append('%.3f' % is_std)
            result['FID'].append('%.3f' % fid_score)
            result['VFS mean'].append('%.3f' % vfs_mean)
            result['VFS std'].append('%.3f' % vfs_std)

    # Save to Excel
    result_df = pd.DataFrame(result)
    sv_path = os.path.join('./output', exp_dir, '%s.xlsx' % exp_dir)
    result_df.to_excel(sv_path)


if __name__ == '__main__':
    main()
