# Code derived from https://github.com/openai/improved-gan/tree/master/inception_score
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
from scipy import linalg
import math
import sys
from pyprind import prog_bar


try:
    MODEL_DIR = '/mnt/.tensorflow/imagenet'#'/mnt/imagenet'
    os.listdir(MODEL_DIR)
except:
    MODEL_DIR = './.cache/.tensorflow/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
last_layer = None

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


def inception_forward(images, layer):
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    # bs = 100
    bs = 1
    # images = images.transpose(0, 2, 3, 1)
    #with tf.Session(config=config) as sess:
    preds = []
    n_batches = int(math.ceil(float(len(images)) / float(bs)))
    # for i in range(n_batches):
    for i in prog_bar(range(n_batches), title="Calculating FID Scores", width=50):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        inps = images[(i * bs):min((i + 1) * bs, len(images))]
        for inp in inps:
            pred = fw_sess.run(layer, {'ExpandDims:0': [inp]})
            preds.append(pred)
    preds = np.array(preds)
        # preds = np.concatenate(preds, 0)
    return preds


def get_mean_and_cov(images):
    before_preds = inception_forward(images, last_layer)
    m = np.mean(before_preds, 0)
    cov = np.cov(before_preds, rowvar=False)
    return m, cov


def get_fid(images, ref_stats=None, images_ref=None, splits=10):
    before_preds = inception_forward(images, last_layer)
    if ref_stats is None:
        if images_ref is None:
            raise ValueError('images_ref should be provided if ref_stats is None')
        m_ref, cov_ref = get_mean_and_cov(images_ref)
    fids = []
    for i in range(splits):
        part = before_preds[(i * before_preds.shape[0] // splits):((i + 1) * before_preds.shape[0] // splits), :]
        m_gen = np.mean(part, 0)
        cov_gen = np.cov(part, rowvar=False)
        fid = np.sum((m_ref - m_gen) ** 2) + np.trace(
            cov_ref + cov_gen - 2 * scipy.linalg.sqrtm(np.dot(cov_ref, cov_gen)))
        fids.append(fid)
    return np.mean(fids), np.std(fids)


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    preds = inception_forward(images, softmax)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_pred(images):
    preds = inception_forward(images, softmax)
    return preds


def get_fid_pred(images):
    preds = inception_forward(images, last_layer)
    return preds

def close_sess():
    fw_sess.close()


# This function is called automatically.
def _init_inception():
    global softmax
    global last_layer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                # o.set_shape(tf.TensorShape(new_shape))
                o.set_shape = tf.TensorShape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        last_layer = tf.squeeze(pool3)
        logits = tf.matmul(tf.expand_dims(last_layer, 0), w)
        softmax = tf.nn.softmax(logits[0])

    global fw_sess
    fw_sess = tf.Session(config=config)


if softmax is None:
    _init_inception()



def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
