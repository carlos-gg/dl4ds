import tensorflow as tf
from tensorflow.keras.layers import (Add, Input, Concatenate, TimeDistributed)
from tensorflow.keras.models import Model

from .blocks import (RecurrentConvBlock, ConvBlock, SubpixelConvolutionBlock, 
                     DeconvolutionBlock, LocalizedConvBlock, 
                     get_dropout_layer, TransitionBlock, ResizeConvolutionBlock)
from ..utils import (checkarg_backbone, checkarg_upsampling, 
                    checkarg_dropout_variant)


def recnet_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_aux_channels,
    lr_size,
    time_window, 
    # ----- below are parameters that shall be tweaked by the user -----
    n_channels_out=1, 
    n_filters=8,
    n_blocks=4,
    dropout_rate=0,
    dropout_variant=None,
    normalization=None,
    attention=False,
    activation='relu',
    output_activation=None,
    rc_interpolation='bilinear',
    localcon_layer=False):
    """
    Recurrent deep neural network with different backbone architectures 
    (according to the ``backbone_block``) and post-upsampling methods (according 
    to ``upsampling``). These models are capable of exploiting spatio-temporal
    samples.

    Parameters
    ----------
    backbone_block : str
        Backbone type. One of dl4ds.BACKBONE_BLOCKS. WARNING: this parameter is
        not supposed to be set by the user. It's set internallly through 
        dl4ds.Trainers. 
    upsampling : str
        Upsampling method. One of dl4ds.UPSAMPLING_METHODS. WARNING: this 
        parameter is not supposed to be set by the user. It's set internallly 
        through dl4ds.Trainers. 
    scale : int
        Scaling factor, LR wrt HR grid. WARNING: this parameter is not supposed 
        to be set by the user. It's set internallly through dl4ds.Trainers. 
    n_channels : int
        Number of channels/variables in each sample. WARNING: this parameter is
        not supposed to be set by the user. It's set internallly through
        dl4ds.Trainers. 
    n_aux_channels : int
        Number of auxiliary channels. WARNING: this parameter is not supposed to 
        be set by the user. It's set internallly through dl4ds.Trainers. 
    lr_size : tuple
        Height and width of the LR grid. WARNING: this parameter is not supposed 
        to be set by the user. It's set internallly through dl4ds.Trainers.
    time_window : int
        Temporal window or number of time steps in each sample. WARNING: this 
        parameter is not supposed to be set by the user. It's set internallly  
        through dl4ds.Trainers.
    n_filters : int, optional
        Number of convolutional filters in RecurrentConvBlock. `n_filters` sets 
        the number of output filters in the convolution inside the ConvLSTM unit. 
    n_blocks : int, optional
        Number of recurrent convolutional blocks (RecurrentConvBlock). 
        Sets the depth of the network. 
    normalization : str or None, optional
        Normalization method in the residual or dense block. Can be either 'bn'
        for BatchNormalization or 'ln' for LayerNormalization. If None, then no
        normalization is performed (eg., for the 'resnet' backbone this results 
        in the EDSR-style residual block).
    dropout_rate : float, optional
        Float between 0 and 1. Fraction of the input units to drop. If 0 then no
        dropout is applied. 
    dropout_variant : str or None, optional
        Type of dropout. Defined in dl4ds.DROPOUT_VARIANTS variable. 
    attention : bool, optional
        If True, dl4ds.ChannelAttention2D is used in convolutional blocks. 
    activation : str, optional
        Activation function to use, as supported by tf.keras. E.g., 'relu' or 
        'gelu'.
    output_activation : str, optional
        Activation function to use in the last ConvBlock. Useful to constraint 
        the values distribution of the output grid.
    rc_interpolation : str, optional
        Interpolation used in the ResizeConvolutionBlock. Supported methods: 
        "bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", 
        "gaussian", "mitchellcubic". 
    localcon_layer : bool, optional
        If True, the LocalizedConvBlock is activated in the output module. 
    """
    backbone_block = checkarg_backbone(backbone_block)
    upsampling = checkarg_upsampling(upsampling)
    dropout_variant = checkarg_dropout_variant(dropout_variant)
        
    auxvar_array_is_given = True if n_aux_channels > 0 else False

    h_lr, w_lr = lr_size
    x_in = Input(shape=(None, h_lr, w_lr, n_channels))
    
    x = b = RecurrentConvBlock(n_filters, activation=activation, 
        normalization=normalization, name_suffix='1')(x_in)

    for i in range(n_blocks):
        b = RecurrentConvBlock(n_filters, activation=activation, 
            normalization=normalization, dropout_rate=dropout_rate,
            dropout_variant=dropout_variant, name_suffix=str(i + 2))(b)
    
    b = get_dropout_layer(dropout_rate, dropout_variant, dim=3)(b)
    
    if backbone_block == 'convnet':
        x = b
        n_filters_ups = n_filters
    elif backbone_block == 'resnet':
        x = Add()([x, b])
        n_filters_ups = n_filters
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
        n_filters_ups = x.get_shape()[-1]
    
    if upsampling == 'spc':
        upsampling_layer = SubpixelConvolutionBlock(scale, n_filters_ups)
    elif upsampling == 'rc':
        upsampling_layer = ResizeConvolutionBlock(scale, n_filters_ups, interpolation=rc_interpolation)
    elif upsampling == 'dc':
        upsampling_layer = DeconvolutionBlock(scale, n_filters_ups)
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
    x = TransitionBlock(x.get_shape()[-1] // 2, name='TransitionLast')(x)
    x = ConvBlock(n_filters, activation=None, dropout_rate=dropout_rate, 
        normalization=normalization, attention=True)(x)  

    x = ConvBlock(n_channels_out, activation=output_activation, 
        dropout_rate=0, normalization=normalization, attention=False)(x) 

    model_name = 'rec' + backbone_block + '_' + upsampling
    if auxvar_array_is_given:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)
    else:
        return Model(inputs=[x_in], outputs=x, name=model_name)

