import torch
import models
from copy import deepcopy
from utils import update_values, load_model_state
import glog as log

def build_model(opts, image_size, checkpoint_start_from=None):
    if checkpoint_start_from is not None:
        log.info("Load checkpoint as initialization: {}".format(
            checkpoint_start_from))
        checkpoint = torch.load(checkpoint_start_from)
        # kwarg aka keyword arguments
        #kwargs = checkpoint['model_kwargs']
        kwargs = deepcopy(opts["options"])
        kwargs["image_size"] = image_size
        model = getattr(models, opts["arch"])(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        #print("state_dict: ", state_dict.keys())
        model = load_model_state(model, state_dict, strict=False)
    else:
        kwargs = deepcopy(opts["options"])
        kwargs["image_size"] = image_size
        model = getattr(models, opts["arch"])(**kwargs)
    return model, kwargs

def build_ac_discriminator(opts):
    d_kwargs = deepcopy(opts["generic"])
    discriminator = []
    if "identity" in opts.keys():
        d_kwargs = update_values(opts["identity"], d_kwargs)
        dis = models.AcDiscriminator(**d_kwargs)
        discriminator.append(dis)
    if "identity_low" in opts.keys():
        d_kwargs = update_values(opts["identity_low"], d_kwargs)
        dis_low = models.AcDiscriminator(**d_kwargs)
        discriminator.append(dis_low)
    if "identity_mid" in opts.keys():
        d_kwargs = update_values(opts["identity_mid"], d_kwargs)
        dis_mid = models.AcDiscriminator(**d_kwargs)
        discriminator.append(dis_mid)
    if "identity_high" in opts.keys():
        d_kwargs = update_values(opts["identity_high"], d_kwargs)
        dis_high = models.AcDiscriminator(**d_kwargs)
        discriminator.append(dis_high)
    return discriminator, d_kwargs

def build_img_discriminator(opts):
    discriminator = []
    if opts.get("pgan_dis") is not None:
        d_kwargs = deepcopy(opts["pgan_dis"])
        pgan_dis = models.PGAN_Discriminator(**d_kwargs)
        discriminator.append(pgan_dis)
    else:
        d_kwargs = deepcopy(opts["generic"])
        if "image" in opts.keys():
            d_kwargs = update_values(opts["image"], d_kwargs)
            dis = models.PatchDiscriminator(**d_kwargs)
            discriminator.append(dis)
        else:
            if "image_low" in opts.keys():
                d_kwargs = update_values(opts["image_low"], d_kwargs)
                discriminator_low = models.PatchDiscriminator(**d_kwargs)
                discriminator.append(discriminator_low)
            if "image_mid" in opts.keys():
                d_kwargs = update_values(opts["image_mid"], d_kwargs)
                discriminator_mid = models.PatchDiscriminator(**d_kwargs)
                discriminator.append(discriminator_mid)
            if "image_high" in opts.keys():
                d_kwargs = update_values(opts["image_high"], d_kwargs)
                discriminator_high = models.PatchDiscriminator(**d_kwargs)
                discriminator.append(discriminator_high)
    return discriminator, d_kwargs

def build_cond_discriminator(opts):
    discriminator = []
    d_kwargs = deepcopy(opts["generic"])
    if "condition" in opts.keys():
        d_kwargs = update_values(opts["condition"], d_kwargs)
        dis = models.CondPatchDiscriminator(**d_kwargs)
        discriminator.append(dis)
    else:
        if "cond_low" in opts.keys():
            d_kwargs = update_values(opts["cond_low"], d_kwargs)
            discriminator_low = models.CondPatchDiscriminator(**d_kwargs)
            discriminator.append(discriminator_low)
        if "cond_mid" in opts.keys():
            d_kwargs = update_values(opts["cond_mid"], d_kwargs)
            discriminator_mid = models.CondPatchDiscriminator(**d_kwargs)
            discriminator.append(discriminator_mid)
        if "cond_high" in opts.keys():
            d_kwargs = update_values(opts["cond_high"], d_kwargs)
            discriminator_high = models.CondPatchDiscriminator(**d_kwargs)
            discriminator.append(discriminator_high)
    return discriminator, d_kwargs
