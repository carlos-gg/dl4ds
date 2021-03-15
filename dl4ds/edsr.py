import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model

from .blocks import residual_block, normalize, denormalize


def edsr(scale, n_channels, n_filters, n_res_blocks):
    """
    EDSR model with pixel shuffle upscaling
    """
    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, 3, padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters)
    b = Conv2D(n_filters, 3, padding='same')(b)
    x = Add()([x, b])
    
    x = upsample(x, scale, n_filters)
    x = Conv2D(1, 3, padding='same')(x)

    return Model(x_in, x, name="edsr")

# def edsr(scale, n_channels, n_filters, n_res_blocks, x_train_mean, x_train_std):
#     """
#     EDSR model with pixel shuffle upscaling
#     """
#     x_in = Input(shape=(None, None, n_channels))
#     x = Lambda(normalize)((x_in, x_train_mean, x_train_std))
#     x = b = Conv2D(n_filters, 3, padding='same')(x)
#     for i in range(n_res_blocks):
#         b = residual_block(b, n_filters)
#     b = Conv2D(n_filters, 3, padding='same')(b)
#     x = Add()([x, b])
    
#     x = upsample(x, scale, n_filters)
#     x = Conv2D(n_channels, 3, padding='same')(x)

#     x = Lambda(denormalize)((x, x_train_mean, x_train_std))
#     return Model(x_in, x, name="edsr")


def upsample(x, scale, n_filters):
    def upsample_conv(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(n_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_conv(x, 2, name='conv2d_scale_2')
    elif scale == 4:
        x = upsample_conv(x, 2, name='conv2d_1_scale_2')
        x = upsample_conv(x, 2, name='conv2d_2_scale_2')
    elif scale == 20:
        x = upsample_conv(x, 5, name='conv2d_1_scale_5')
        x = upsample_conv(x, 4, name='conv2d_2_scale_4')
    else:
        x = upsample_conv(x, scale, name='conv2d_scale_' + str(scale))

    return x


def pixel_shuffle(scale):
    """
    See https://arxiv.org/abs/1609.05158
    """
    return lambda x: tf.nn.depth_to_space(x, scale)

