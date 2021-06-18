from tensorflow.keras.layers import (Add, Conv2D, Lambda, ReLU, 
                                     BatchNormalization, LayerNormalization)
from .attention import ChannelAttention2D


def residual_block(x_in, filters, scaling=None, attention=False, normalization=None):
    """Create a residual block

    Standard residual block: Conv2D -> BN -> ReLU -> Conv2D -> BN
    EDSR-style block: Conv2D -> ReLU -> Conv2D
    """
    x = Conv2D(filters, (3, 3), padding='same')(x_in)
    if normalization is not None:
        if normalization == 'bn':
            x = BatchNormalization()(x)
        elif normalization == 'ln':
            x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if scaling is not None:
        x = Lambda(lambda t: t * scaling)(x)
    if attention:
        x = ChannelAttention2D(x.shape[-1])(x)
    x = Add()([x_in, x])
    return x


def normalize(params):
    x, x_train_mean, x_train_std = params
    return (x - x_train_mean) / x_train_std


def denormalize(params):
    x, x_train_mean, x_train_std = params
    return x * x_train_std + x_train_mean