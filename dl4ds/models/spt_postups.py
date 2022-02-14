import tensorflow as tf
from tensorflow.keras.layers import (Add, Input, UpSampling2D, Dropout, 
                                     GaussianDropout, Concatenate, 
                                     TimeDistributed, SpatialDropout3D)
from tensorflow.keras.models import Model

from .blocks import (RecurrentConvBlock, ConvBlock, SubpixelConvolutionBlock, 
                     DeconvolutionBlock, LocalizedConvBlock, MCDropout, 
                     MCSpatialDropout3D, MCGaussianDropout)
from ..utils import (checkarg_backbone, checkarg_upsampling, 
                    checkarg_dropout_variant)


def recnet_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_aux_channels,
    n_filters, 
    n_blocks,   # not used, here for compatibility with the signature of other models
    lr_size,
    time_window, 
    n_channels_out=1, 
    activation='relu',
    dropout_rate=0.2,
    dropout_variant='spatial',
    normalization=None,
    attention=False,
    output_activation=None,
    localcon_layer=False):
    """
    Recurrent deep neural network with different backbone architectures 
    (according to the ``backbone_block``) and post-upsampling methods (according 
    to ``upsampling``). These models are capable of exploiting spatio-temporal
    samples.

    """
    backbone_block = checkarg_backbone(backbone_block)
    upsampling = checkarg_upsampling(upsampling)
    dropout_variant = checkarg_dropout_variant(dropout_variant)
        
    auxvar_array_is_given = True if n_aux_channels > 0 else False

    h_lr = lr_size[0]
    w_lr = lr_size[1]

    if not localcon_layer: 
        x_in = Input(shape=(None, None, None, n_channels))
    else:
        x_in = Input(shape=(None, h_lr, w_lr, n_channels))
    
    x = b = RecurrentConvBlock(
        n_filters, 
        activation=activation, 
        normalization=normalization)(x_in)

    b = RecurrentConvBlock(
        n_filters, 
        activation=activation, 
        normalization=normalization,
        dropout_rate=dropout_rate,
        dropout_variant=dropout_variant)(b)
    
    if dropout_rate > 0:
        if dropout_variant is None:
            b = Dropout(dropout_rate)(b)
        elif dropout_variant == 'gaussian':
            b = GaussianDropout(dropout_rate)(b)
        elif dropout_variant == 'spatial':
            b = SpatialDropout3D(dropout_rate)(b)
        elif dropout_variant == 'mcdrop':
            b = MCDropout(dropout_rate)(b)
        elif dropout_variant == 'mcgaussiandrop':
            b = MCGaussianDropout(dropout_rate)
        elif dropout_variant == 'mcspatialdrop':
            b = MCSpatialDropout3D(dropout_rate)
    
    if backbone_block == 'convnet':
        x = b
        n_filters_ = n_filters
    elif backbone_block == 'resnet':
        x = Add()([x, b])
        n_filters_ = n_filters
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
        n_filters_ = x.get_shape()[-1]
    
    if upsampling == 'spc':
        upsampling_layer = SubpixelConvolutionBlock(scale, n_filters_)
    elif upsampling == 'rc':
        upsampling_layer = UpSampling2D(scale, interpolation='bilinear')
    elif upsampling == 'dc':
        upsampling_layer = DeconvolutionBlock(scale, n_filters_)
    x = TimeDistributed(upsampling_layer, name='upsampling_' + upsampling)(x)

    #---------------------------------------------------------------------------
    # HR aux channels are processed
    if auxvar_array_is_given:
        s_in = Input(shape=(None, None, n_aux_channels))
        s = ConvBlock(n_filters, activation=activation, dropout_rate=0, 
                      normalization=None, attention=attention)(s_in)
        s = tf.expand_dims(s, 1)
        s = tf.repeat(s, time_window, axis=1)
        x = Concatenate()([x, s])
    
    #---------------------------------------------------------------------------
    # Localized convolutional layer
    if localcon_layer:
        lcb = LocalizedConvBlock(filters=2, use_bias=True)
        lws = TimeDistributed(lcb, name='localized_conv_block')(x)
        x = Concatenate()([x, lws])

    #---------------------------------------------------------------------------
    # Last conv layers
    x = ConvBlock(n_filters, activation=None, dropout_rate=dropout_rate, 
        normalization=normalization, attention=True)(x)  

    x = ConvBlock(n_channels_out, activation=output_activation, 
        dropout_rate=0, normalization=normalization, attention=False)(x) 

    model_name = 'rec' + backbone_block + '_' + upsampling
    if auxvar_array_is_given:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)
    else:
        return Model(inputs=[x_in], outputs=x, name=model_name)

