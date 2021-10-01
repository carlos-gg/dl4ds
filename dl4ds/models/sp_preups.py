import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, Concatenate, Dropout, 
                                     GaussianDropout)
from tensorflow.keras.models import Model

from ..blocks import ResidualBlock, ConvBlock, DenseBlock, TransitionBlock
from ..utils import checkarg_backbone, checkarg_dropout_variant
 

def net_pin(
    backbone_block,
    n_channels, 
    n_aux_channels,
    n_filters, 
    n_blocks, 
    n_channels_out=1, 
    activation='relu',
    dropout_rate=0.2,
    dropout_variant='spatial',
    normalization=None,
    attention=False,
    output_activation=None):
    """
    Deep neural network with different backbone architectures (according to the
    ``backbone_block``) and pre-upsampling via (bicubic) interpolation.

    The interpolation method depends on the ``interpolation`` argument used in
    the training procedure (which is passed to the DataGenerator).
    """
    backbone_block = checkarg_backbone(backbone_block)
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
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
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
    else:
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    
    model_name = backbone_block + '_pin'
    if auxvar_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)  
    else:
        return Model(inputs=x_in, outputs=x, name=model_name)