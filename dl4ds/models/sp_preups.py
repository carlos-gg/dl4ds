import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, Concatenate,  
                                     UpSampling2D)
from tensorflow.keras.models import Model

from .blocks import (ResidualBlock, ConvBlock, DenseBlock, TransitionBlock,
                     LocalizedConvBlock, SubpixelConvolutionBlock, 
                     DeconvolutionBlock, EncoderBlock, PadConcat, 
                     get_dropout_layer, ConvNextBlock, ResizeConvolutionBlock)
from ..utils import checkarg_backbone, checkarg_dropout_variant
 

def net_pin(
    backbone_block,
    n_channels, 
    n_aux_channels,
    hr_size,
    # ----- below are parameters that shall be tweaked by the user -----
    n_channels_out=1, 
    n_filters=8, 
    n_blocks=6, 
    dropout_rate=0,
    dropout_variant=None,
    normalization=None,
    attention=False,
    activation='relu',
    output_activation=None,
    localcon_layer=False):
    """
    Deep neural network with different backbone architectures (according to the
    ``backbone_block``) and pre-upsampling via interpolation (the samples are 
    expected to be interpolated to the HR grid).

    The interpolation method depends on the ``interpolation`` argument used in
    the training procedure (which is passed to the DataGenerator).

    Parameters
    ----------
    backbone_block : str
        Backbone type. One of dl4ds.BACKBONE_BLOCKS. WARNING: this parameter is
        not supposed to be set by the user. It's set internallly through
        dl4ds.Trainers. 
    n_channels : int
        Number of channels/variables in each sample. WARNING: this parameter is
        not supposed to be set by the user. It's set internallly through
        dl4ds.Trainers. 
    n_aux_channels : int
        Number of auxiliary channels. WARNING: this parameter is not supposed to 
        be set by the user. It's set internallly through dl4ds.Trainers. 
    hr_size : tuple
        Height and width of the HR grid. WARNING: this parameter is not supposed 
        to be set by the user. It's set internallly through dl4ds.Trainers.
    n_filters : int, optional
        Number of convolutional filters in the first convolutional block. 
        `n_filters` sets the dimensionality of the output space of the first 
        convolutional block (i.e. the number of output filters in the 
        convolution). The number of filters grows in subsequent convolutional 
        blocks.
    n_blocks : int, optional
        Number of convolutional blocks (ConvBlock, ResidualBlock, DenseBlock or
        ConvNextBlock). Sets the depth of the network. 
    normalization : str or None, optional
        Normalization method in the residual or dense block. Can be either 'bn'
        for BatchNormalization or 'ln' for LayerNormalization. If None, then no
        normalization is performed. For the 'resnet' backbone, it results in the
        EDSR-style residual block.
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
    localcon_layer : bool, optional
        If True, the LocalizedConvBlock is activated in the output module. 
    """
    backbone_block = checkarg_backbone(backbone_block)
    dropout_variant = checkarg_dropout_variant(dropout_variant)

    h_hr = hr_size[0]
    w_hr = hr_size[1]

    auxvar_array_is_given = True if n_aux_channels > 0 else False
    if auxvar_array_is_given:
        if not localcon_layer:
            s_in = Input(shape=(None, None, n_aux_channels))
        else:
            s_in = Input(shape=(h_hr, w_hr, n_aux_channels))

    if not localcon_layer:  
        x_in = Input(shape=(None, None, n_channels))
    else:
        x_in = Input(shape=(h_hr, w_hr, n_channels))

    init_n_filters = n_filters
    #---------------------------------------------------------------------------
    # N conv blocks
    if backbone_block == 'convnext':  
        ks = (7, 7)     
        x = b = Conv2D(n_filters, ks, padding='same')(x_in)
        # N convnext blocks
        for i in range(n_blocks):
            n_filters = init_n_filters * (i + 1)
            b = ConvNextBlock(
                filters=n_filters, drop_path=0, normalization=normalization, 
                use_1x1conv=False if i == 0 else True, activation=activation,
                name='ConvNextBlock' + str(i+1))(b)
        x = TransitionBlock(n_filters, activation=activation)(x)
        x = Add()([x, b])
    else:
        ks = (3, 3)
        x = b = Conv2D(n_filters, ks, padding='same')(x_in)
        # N conv blocks
        for i in range(n_blocks):
            n_filters = init_n_filters * (i + 1)
            if backbone_block == 'convnet':
                b = ConvBlock(
                    n_filters, activation=activation, dropout_rate=dropout_rate, 
                    dropout_variant=dropout_variant, normalization=normalization, 
                    attention=attention, name='ConvBlock' + str(i+1))(b)
            elif backbone_block == 'resnet':
                b = ResidualBlock(
                    n_filters, activation=activation, dropout_rate=dropout_rate, 
                    dropout_variant=dropout_variant, normalization=normalization, 
                    use_1x1conv=False if i == 0 else True, attention=attention,
                    name='ResidualBlock' + str(i+1))(b)
            elif backbone_block == 'densenet':
                b = DenseBlock(
                    n_filters, activation=activation, dropout_rate=dropout_rate, 
                    dropout_variant=dropout_variant, normalization=normalization, 
                    attention=attention, name='DenseBlock' + str(i+1))(b)
                b = TransitionBlock(b.get_shape()[-1] // 2, 
                                    name='Transition' + str(i+1))(b)  
        b = Conv2D(n_filters, ks, padding='same', activation=activation)(b)
        
        b = get_dropout_layer(dropout_rate, dropout_variant)(b)

        if backbone_block == 'convnet':
            x = b
        elif backbone_block == 'resnet':
            x = TransitionBlock(n_filters, activation=activation)(x)
            x = Add()([x, b])
        elif backbone_block == 'densenet':
            x = Concatenate()([x, b])
            x = TransitionBlock(n_filters, activation=activation)(x)
    
    #---------------------------------------------------------------------------
    # Localized convolutional layer
    if localcon_layer:
        lws = LocalizedConvBlock(filters=2, use_bias=True)(x)
        x = Concatenate()([x, lws])

    #---------------------------------------------------------------------------
    # HR aux channels are processed
    if auxvar_array_is_given:
        if backbone_block == 'convnext':
            s = ConvNextBlock(
                filters=n_filters, drop_path=0, normalization=normalization, 
                use_1x1conv=True, activation=activation, 
                name='ConvNextBlock_aux')(s_in)
        else:
            s = ConvBlock(
                filters=n_filters, activation=activation, dropout_rate=0, 
                normalization=normalization, attention=False,
                name='ConvBlock_aux')(s_in) 
        x = Concatenate()([x, s])    

    #---------------------------------------------------------------------------
    # Last conv layers
    x = TransitionBlock(init_n_filters, name='TransitionLast')(x)
    x = ConvBlock(
        init_n_filters, ks_cl1=ks, ks_cl2=ks, activation=None, 
        dropout_rate=dropout_rate, normalization=normalization, attention=True)(x)  

    x = ConvBlock(
        n_channels_out, ks_cl1=ks, ks_cl2=ks, activation=output_activation, 
        dropout_rate=0, normalization=normalization, attention=False)(x)     
    
    model_name = backbone_block + '_pin'
    if auxvar_array_is_given:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)  
    else:
        return Model(inputs=[x_in], outputs=x, name=model_name)


def unet_pin(
    backbone_block,
    n_channels, 
    n_aux_channels,
    n_filters, 
    n_blocks, 
    hr_size,
    n_channels_out=1, 
    activation='relu',
    dropout_rate=0,
    dropout_variant=None,
    normalization=None,
    attention=False,
    decoder_upsampling='rc',
    rc_interpolation='bilinear',
    output_activation=None,
    width_cap=256,
    localcon_layer=False):
    """    
    Deep neural network with UNET (encoder-decoder) backbone and pre-upsampling 
    via interpolation.

    The interpolation method depends on the ``interpolation`` argument used in
    the training procedure (which is passed to the DataGenerator).

    Parameters
    ----------
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
    """
    backbone_block = checkarg_backbone(backbone_block)
    dropout_variant = checkarg_dropout_variant(dropout_variant)
    n_blocks = _check_nblocks(hr_size, n_blocks)
    h_hr = hr_size[0]
    w_hr = hr_size[1]

    auxvar_array_is_given = True if n_aux_channels > 0 else False
    if auxvar_array_is_given:
        if not localcon_layer and h_hr == w_hr:
            s_in = Input(shape=(None, None, n_aux_channels))
        else:
            s_in = Input(shape=(h_hr, w_hr, n_aux_channels))

    if not localcon_layer and h_hr == w_hr:  
        x_in = Input(shape=(None, None, n_channels))
    else:
        x_in = Input(shape=(h_hr, w_hr, n_channels))

    init_n_filters = n_filters
    #---------------------------------------------------------------------------
    # n enconding conv blocks
    x = x_in
    enconding_filters = []
    n_filters_list = []
    for i in range(n_blocks):
        droprate = dropout_rate if i == n_blocks else 0
        x, x_skipcon = EncoderBlock(
            n_filters=n_filters, activation=activation, 
            dropout_rate=droprate, dropout_variant=dropout_variant, 
            normalization=normalization, attention=attention, name_suffix=str(i+1))(x)
        enconding_filters.append(x_skipcon)
        n_filters_list.append(n_filters)
        n_filters = min(width_cap, n_filters * 2)   # doubling # of filters with each encoding layer, capping at 256

    # bottleneck layer
    x = ConvBlock(
        n_filters, activation=activation, dropout_rate=dropout_rate, 
        dropout_variant=dropout_variant, name='Bottleneck',
        normalization=None)(x)     # following Isola et al 2016

    # n decoding conv blocks
    n_filters_list = n_filters_list[::-1]
    for j, skip_connection in enumerate(reversed(enconding_filters)):        
        n_filters = n_filters_list[j]
        if decoder_upsampling == 'spc':
            x = SubpixelConvolutionBlock(2, n_filters, name_suffix=str(j+1))(x)
        elif decoder_upsampling == 'rc':
            x = ResizeConvolutionBlock(2, n_filters, interpolation=rc_interpolation, name_suffix=str(j+1))(x)
        elif decoder_upsampling == 'dc':
            x = DeconvolutionBlock(2, n_filters, activation, name_suffix=str(j+1))(x)

        x = PadConcat(name_suffix='_SkipConnection'+str(j+1))([x, skip_connection])        
        x = ConvBlock(
            n_filters, activation=activation, dropout_rate=0, 
            dropout_variant=dropout_variant, normalization=normalization, 
            attention=attention, name='DecoderConvBlock' + str(j+1))(x)

    x = get_dropout_layer(dropout_rate, dropout_variant)(x)

    #---------------------------------------------------------------------------
    # Localized convolutional layer
    if localcon_layer:
        lws = LocalizedConvBlock(filters=2, use_bias=True)(x)
        x = Concatenate()([x, lws])

    #---------------------------------------------------------------------------
    # HR aux channels are processed
    if auxvar_array_is_given:
        s = ConvBlock(n_filters, activation=activation, dropout_rate=0, 
            normalization=normalization, attention=False)(s_in)   
        x = Concatenate()([x, s])   

    #---------------------------------------------------------------------------
    # Last conv layers
    x = TransitionBlock(init_n_filters, name='TransitionLast')(x)
    x = ConvBlock(init_n_filters, activation=None, dropout_rate=dropout_rate, 
        normalization=normalization, attention=True)(x)  

    x = ConvBlock(n_channels_out, activation=output_activation, dropout_rate=0, 
        normalization=normalization, attention=False)(x)     
    
    model_name = backbone_block + '_pin'
    if auxvar_array_is_given:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)  
    else:
        return Model(inputs=[x_in], outputs=x, name=model_name)


def _check_nblocks(shape, power):  
    while shape[0] // 2**power < 2 or shape[1] // 2**power < 2:
        msg = f'`n_blocks` is too large, cannot downsample {power} times '
        msg += f'given the input grid size. Setting `n_blocks` to {power-1}'
        print(msg)
        power -= 1
    return power

