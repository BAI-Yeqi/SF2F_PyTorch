from .model_setup import *

# Generator
# from .paste_gan import PasteGAN
from .face_gan import FaceGAN
from .face_gan_v2 import FaceGAN_v2
from .face_gan_crn import FaceGAN_CRN
from .face_gan_crn_v2 import FaceGAN_CRN_v2
from .face_recon import Face_Recon
from .encoder_decoder import EncoderDecoder

# Lower level encoder decoders
from .voice_encoders import V2F1DCNN
from .face_decoders import V2FDecoder

# Progressive GAN
from .pgan import PGan_Net
from .pgan_dis import PGAN_Discriminator

# Discriminator
from .discriminators import PatchDiscriminator
from .discriminators import AcDiscriminator
from .discriminators import AcCropDiscriminator
from .discriminators import CondPatchDiscriminator

# Facenet for Perceptual Loss and Evaluation
from .inception_resnet_v1 import InceptionResnetV1, fixed_image_standardization
