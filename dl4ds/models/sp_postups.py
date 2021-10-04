import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, UpSampling2D, Dropout, 
                                     GaussianDropout, Concatenate)
from tensorflow.keras.models import Model

from .blocks import (ResidualBlock, ConvBlock, Deconvolution,
                     DenseBlock, TransitionBlock, SubpixelConvolution)
from ..utils import (checkarg_backbone, checkarg_upsampling, 
                    checkarg_dropout_variant)


def net_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_aux_channels,
    n_filters, 
    n_blocks, 
    n_channels_out=1, 
    normalization=None,
    dropout_rate=0.2,
    dropout_variant='spatial',
    attention=False,
    activation='relu',
    output_activation=None):
    """
    Deep neural network with different backbone architectures (according to the
    ``backbone_block``) and post-upsampling methods (according to 
    ``upsampling``).

    Parameters
    ----------
    normalization : str or None, optional
        Normalization method in the residual or dense block. Can be either 'bn'
        for BatchNormalization or 'ln' for LayerNormalization. If None, then no
        normalization is performed. For the 'resnet' backbone, it results in the
        EDSR-style residual block.
    dropout_rate : float, optional
        Float between 0 and 1. Fraction of the input units to drop. If 0 then no
        dropout is applied. 
    dropout_vaiant : str or None, optional
        Type of dropout: gaussian, block, spatial. 
    """
    backbone_block = checkarg_backbone(backbone_block)
    upsampling = checkarg_upsampling(upsampling)
    dropout_variant = checkarg_dropout_variant(dropout_variant)

    auxvar_arr = True if n_aux_channels > 0 else False
    if auxvar_arr:
        s_in = Input(shape=(None, None, n_aux_channels))

    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_blocks):
        if backbone_block == 'convnet':
            b = ConvBlock(
                n_filters, activation=activation, dropout_rate=dropout_rate, 
                dropout_variant=dropout_variant, normalization=normalization, 
                attention=attention)(b)
        elif backbone_block == 'resnet':
            b = ResidualBlock(
                n_filters, activation=activation, dropout_rate=dropout_rate, 
                dropout_variant=dropout_variant, normalization=normalization, 
                attention=attention)(b)
        elif backbone_block == 'densenet':
            b = DenseBlock(
                n_filters, activation=activation, dropout_rate=dropout_rate, 
                dropout_variant=dropout_variant, normalization=normalization, 
                attention=attention)(b)
            b = TransitionBlock(n_filters // 2)(b)  # another option: half of the DenseBlock channels
    b = Conv2D(n_filters, (3, 3), padding='same', activation=activation)(b)
    if dropout_rate > 0:
        if dropout_variant is None:
            b = Dropout(dropout_rate)(b)
        elif dropout_variant == 'gaussian':
            b = GaussianDropout(dropout_rate)(b)

    if backbone_block == 'convnet':
        x = b
    elif backbone_block == 'resnet':
        x = Add()([x, b])
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
    
    model_name = backbone_block + '_' + upsampling
    if auxvar_arr:
        ups_activation = activation
    else:
        ups_activation = output_activation

    if upsampling == 'spc':
        x = SubpixelConvolution(scale, n_filters)(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=ups_activation)(x)
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=ups_activation)(x)
    elif upsampling == 'dc':
        x = Deconvolution(scale, n_channels_out, ups_activation)(x)
    
    if auxvar_arr:
        # x = Concatenate()([x, s_in])
        s = ConvBlock(
            n_filters, 
            activation=activation, 
            dropout_rate=0, 
            normalization=normalization, 
            attention=False)(s_in)  
        x = Concatenate()([x, s])   

        x = ConvBlock(
            n_channels_out, 
            activation=output_activation, 
            dropout_rate=0, 
            normalization=normalization, 
            attention=False)(x)  # attention=True

    if auxvar_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)  
    else:
        return Model(inputs=x_in, outputs=x, name=model_name)  
