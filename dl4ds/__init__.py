"""
.. include:: ../README.md
"""

__version__ = "1.8.0"

BACKBONE_BLOCKS = [
    'convnet',          # plain convolutional block w/o skip connections
    'resnet',           # residual convolutional blocks
    'densenet',         # dense convolutional blocks
    'convnext',         # convnext style residual blocks
    'unet']             # unet (encoder-decoder) backbone

UPSAMPLING_METHODS = [
    'spc',              # pixel shuffle or subpixel convolution in post-upscaling
    'rc',               # resize convolution in post-upscaling
    'dc',               # deconvolution or transposed convolution in post-upscaling
    'pin']              # pre-upsampling via (bicubic) interpolation
POSTUPSAMPLING_METHODS = ['spc', 'rc', 'dc']

INTERPOLATION_METHODS = [
    'inter_area',       # resampling using pixel area relation (from opencv)
    'nearest',          # nearest neightbors interpolation (from opencv)
    'bicubic',          # bicubic interpolation (from opencv)
    'bilinear',         # bilinear interpolation (from opencv)
    'lanczos']          # lanczos interpolation over 8x8 neighborhood (from opencv)

LOSS_FUNCTIONS = [
    'mae',              # mean absolute error  
    'mse',              # mean squarred error  
    'dssim',            # structural dissimilarity
    'dssim_mae',        # 0.8 * DSSIM + 0.2 * MAE
    'dssim_mse',        # 0.8 * DSSIM + 0.2 * MSE
    'dssim_mae_mse',    # 0.6 * DSSIM + 0.2 * MAE + 0.2 * MSE
    'msdssim',          # multiscale structural dissimilarity
    'msdssim_mae',      # 0.8 * MSDSSIM + 0.2 * MAE
    'msdssim_mae_mse']  # 0.6 * MSDSSIM + 0.2 * MAE + 0.2 * MSE

DROPOUT_VARIANTS = [
    'vanilla',          # vanilla dropout
    'gaussian',         # gaussian dropout
    'spatial',          # spatial dropout
    'mcdrop',           # monte carlo (vanilla) dropout
    'mcgaussiandrop',   # monte carlo gaussian dropout
    'mcspatialdrop']    # monte carlo spatial dropout

from .metrics import *
from .inference import *
from .utils import *
from .dataloader import *
from .models import *
from .training import *
from .preprocessing import *

