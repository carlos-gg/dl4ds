__version__ = "0.4.0"

BACKBONE_BLOCKS = ('convnet',  # plain convolutional block w/o skip connections
                   'resnet',  # residual convolutional blocks
                   'densenet')  # dense convolutional blocks
PREFIX_SAMPLE_TYPE = ('', 'rec')
NETS = [p + b for p in PREFIX_SAMPLE_TYPE for b in BACKBONE_BLOCKS]
UPSAMPLING_METHODS = ('spc',  # pixel shuffle or subpixel convolution in post-upscaling
                      'rc',  # resize convolution in post-upscaling
                      'dc',  # deconvolution or transposed convolution in post-upscaling
                      'pin')  # pre-upsampling via (bicubic) interpolation
POSTUPSAMPLING_METHODS = ('spc', 'rc', 'dc')
SPATIAL_MODELS = [p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
SPATIOTEMP_MODELS = ['rec' + p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
MODELS = [n + '_' + u for n in NETS for u in UPSAMPLING_METHODS]

from .metrics import *
from .training_logic import *
from .inference import *
from .utils import *
from .dataloader import *
from .models import *
from .cgan import *

