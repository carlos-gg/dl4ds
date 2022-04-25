__version__ = "1.6.0"

BACKBONE_BLOCKS = ['convnet',       # plain convolutional block w/o skip connections
                   'resnet',        # residual convolutional blocks
                   'densenet',      # dense convolutional blocks
                   'convnext',      # convnext style residual blocks
                   'unet']          # unet (encoder-decoder) backbone

UPSAMPLING_METHODS = ['spc',        # pixel shuffle or subpixel convolution in post-upscaling
                      'rc',         # resize convolution in post-upscaling
                      'dc',         # deconvolution or transposed convolution in post-upscaling
                      'pin']        # pre-upsampling via (bicubic) interpolation
POSTUPSAMPLING_METHODS = ['spc', 'rc', 'dc']

INTERPOLATION_METHODS = ['inter_area', 'nearest', 'bicubic', 'bilinear', 'lanczos']

LOSS_FUNCTIONS = ['mae', 'mse', 'dssim', 'dssim_mae', 'dssim_mse', 
                  'dssim_mae_mse', 'msdssim', 'msdssim_mae', 'msdssim_mae_mse']

from .metrics import *
from .inference import *
from .utils import *
from .dataloader import *
from .models import *
from .training import *
from .preprocessing import *

