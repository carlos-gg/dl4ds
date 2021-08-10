import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, UpSampling2D, Dropout, 
                                     GaussianDropout, Concatenate, 
                                     TimeDistributed)
from tensorflow.keras.models import Model

from .blocks import (RecurrentConvBlock, ResidualBlock, ConvBlock, 
                     DenseBlock, TransitionBlock, SubpixelConvolution, 
                     Deconvolution)
from .utils import (checkarg_backbone, checkarg_upsampling, 
                    checkarg_dropout_variant)


def net_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
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
    if upsampling == 'spc':
        x = SubpixelConvolution(scale, n_filters)(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    elif upsampling == 'dc':
        x = Deconvolution(scale, n_channels_out, output_activation)(x)
    
    return Model(inputs=x_in, outputs=x, name=model_name)  


def recnet_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_filters, 
    n_blocks, 
    lr_size,
    time_window, 
    return_sequence=False,
    n_channels_out=1, 
    activation='relu',
    dropout_rate=0.2,
    dropout_variant='spatial',
    normalization=None,
    attention=False,
    output_activation=None):
    """
    Recurrent deep neural network with different backbone architectures 
    (according to the ``backbone_block``) and post-upsampling methods (according 
    to ``upsampling``). These models are capable of exploiting spatio-temporal
    samples.

    """
    backbone_block = checkarg_backbone(backbone_block)
    upsampling = checkarg_upsampling(upsampling)
    dropout_variant = checkarg_dropout_variant(dropout_variant)
        
    static_arr = True if isinstance(n_channels, tuple) else False
    if static_arr:
        x_n_channels = n_channels[0]
        static_n_channels = n_channels[1]
    else:
        x_n_channels = n_channels

    h_lr = lr_size[0]
    w_lr = lr_size[1]

    x_in = Input(shape=(None, None, None, x_n_channels))
    if backbone_block == 'convnet':
        skipcon = None
    elif backbone_block == 'resnet':
        skipcon = 'residual'
    elif backbone_block == 'densenet':
        skipcon = 'dense'
    
    x = b = RecurrentConvBlock(
        n_filters, 
        output_full_sequence=return_sequence, 
        skip_connection_type=skipcon,  
        activation=activation, 
        normalization=normalization)(x_in)

    if static_arr:
        s_in = Input(shape=(None, None, static_n_channels))
        s_in_lr = tf.image.resize(images=s_in, size=(h_lr, w_lr), method='bilinear')

        if return_sequence:
            s_in_lr = tf.expand_dims(s_in_lr, 1)
            s_in_lr = tf.repeat(s_in_lr, time_window, axis=1)
        
        x = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(x)
        x = Concatenate()([x, s_in_lr])
        b = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(b)
        b = Concatenate()([b, s_in_lr])

    if return_sequence:
        b = RecurrentConvBlock(
            n_filters, 
            output_full_sequence=return_sequence, 
            skip_connection_type=skipcon,  
            activation=activation, 
            normalization='ln',
            dropout_rate=dropout_rate,
            dropout_variant=dropout_variant)(b)
    else:
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
        n_filters_spc = n_filters
    elif backbone_block == 'resnet':
        x = Add()([x, b])
        n_filters_spc = n_filters
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
        n_filters_spc = x.get_shape()[-1]
    
    if return_sequence:
        if upsampling == 'spc':
            upsampling_layer = SubpixelConvolution(scale, n_filters_spc)
        elif upsampling == 'rc':
            upsampling_layer = UpSampling2D(scale, interpolation='bilinear')
        elif upsampling == 'dc':
            upsampling_layer = Deconvolution(scale, n_filters)
        x = TimeDistributed(upsampling_layer, name='upsampling_' + upsampling)(x)
    else:
        if upsampling == 'spc':
            x = SubpixelConvolution(scale, n_filters_spc)(x)
        elif upsampling == 'rc':
            x = UpSampling2D(scale, interpolation='bilinear')(x)
        elif upsampling == 'dc':
            x = Deconvolution(scale, n_filters)(x)  
            
    # concatenating the HR version of the static array
    if static_arr:
        # x = Concatenate()([x, s_in])
        s = ConvBlock(n_filters, activation=activation, dropout_rate=0, 
                        normalization=None, attention=False)(s_in)
        if return_sequence:
            s = tf.expand_dims(s, 1)
            s = tf.repeat(s, time_window, axis=1)
        x = Concatenate()([x, s])
        
    x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)

    model_name = 'rec' + backbone_block + '_' + upsampling
    if static_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)
    else:
        return Model(inputs=x_in, outputs=x, name=model_name)

