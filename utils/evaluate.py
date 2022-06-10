import pyprind
import numpy as np
import torch
import torch.nn.functional as F
import glog as log

from datasets import imagenet_deprocess_batch
from scripts.compute_inception_score import get_inception_score
from scripts.compute_vggface_score import get_vggface_score


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

def evaluate(model, data_loader, options):
    '''
    Evaluate the current model...
    '''
    normalize_method = options["data"]["data_opts"].get(
        'normalize_method', 'imagenet')
    model.eval()
    log.info("Evaluating with Inception Scores.")
    images = []
    imgs_gt = []
    for iter, batch in enumerate(pyprind.prog_bar(data_loader,
                                  title="[Generating Images]",
                                  width=50)):
        ######### unpack the data #########
        imgs, log_mels, human_ids = batch
        imgs = imgs.cuda()
        imgs_gt.append(imgs[0])
        ###################################
        with torch.no_grad():
            float_dtype = torch.cuda.FloatTensor
            log_mels = log_mels.type(float_dtype)
            model_out = model(log_mels)
            """
            imgs_pred: generated images
            others: placeholder for other output
            """
        imgs_pred, others = model_out
        img = imagenet_deprocess_batch(
            imgs_pred, normalize_method=normalize_method)
        img = crop_vgg_box(img)
        for i in range(img.shape[0]):
            img_np = img[i].numpy().transpose(1, 2, 0)
            images.append(img_np)
    log.info("Computing inception scores...")
    is_mean, is_std = get_inception_score(images)
    log.info("Computing VF scores...")
    vfs_mean, vfs_std = get_vggface_score(images)

    return is_mean, is_std, vfs_mean, vfs_std


if __name__ == '__main__':
    evaluate()
