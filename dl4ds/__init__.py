__version__ = "1.3.0"

BACKBONE_BLOCKS = ['convnet',       # plain convolutional block w/o skip connections
                   'resnet',        # residual convolutional blocks
                   'densenet',      # dense convolutional blocks
                   'convnext']      # convnext style residual blocks
PREFIX_SAMPLE_TYPE = ['', 'rec']
NETS = [p + b for p in PREFIX_SAMPLE_TYPE for b in BACKBONE_BLOCKS]
UPSAMPLING_METHODS = ['spc',        # pixel shuffle or subpixel convolution in post-upscaling
                      'rc',         # resize convolution in post-upscaling
                      'dc',         # deconvolution or transposed convolution in post-upscaling
                      'pin']        # pre-upsampling via (bicubic) interpolation
POSTUPSAMPLING_METHODS = ['spc', 'rc', 'dc']
SPATIAL_MODELS = [p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
SPATIOTEMP_MODELS = ['rec' + p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
MODELS = [n + '_' + u for n in NETS for u in UPSAMPLING_METHODS]
MODELS.append('unet_pin')           # encoder decoder (unet) backbone in pre-upsampling
SPATIAL_MODELS.append('unet_pin') 

from .metrics import *
from .inference import *
from .utils import *
from .dataloader import *
from .models import *
from .training import *
from .preprocessing import *

