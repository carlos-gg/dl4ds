from tensorflow.keras.layers import Add, Conv2D, Input, Concatenate
from tensorflow.keras.models import Model

from .blocks import (RecurrentConvBlock, ResidualBlock, ConvBlock, 
                     DenseBlock, TransitionBlock)
from .utils import checkarg_backbone
 

def net_pin(
    backbone_block,
    n_channels, 
    n_filters, 
    n_blocks, 
    n_channels_out=1, 
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

    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_blocks):
        if backbone_block == 'convnet':
            b = ConvBlock(n_filters, normalization=normalization, attention=attention)(b)
        elif backbone_block == 'resnet':
            b = ResidualBlock(n_filters, normalization=normalization, attention=attention)(b)
        elif backbone_block == 'densenet':
            b = DenseBlock(n_filters, normalization=normalization, attention=attention)(b)
            b = TransitionBlock(n_filters // 2)(b)  # another option: half of the DenseBlock channels
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    
    if backbone_block == 'convnet':
        x = b
    elif backbone_block == 'resnet':
        x = Add()([x, b])
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
    
    x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    model_name = backbone_block + '_pin'
    return Model(inputs=x_in, outputs=x, name=model_name)


def recnet_pin(
    backbone_block,
    n_channels, 
    n_filters, 
    n_blocks, 
    n_channels_out=1, 
    time_window=None, 
    normalization=None,
    attention=False,
    output_activation=None):
    """
    Recurrent deep neural network with different backbone architectures 
    (according to the ``backbone_block``) and pre-upsampling via interpolation. 
    These models are capable of exploiting spatio-temporal samples.
    """
    backbone_block = checkarg_backbone(backbone_block)
    static_arr = True if isinstance(n_channels, tuple) else False
    if static_arr:
        x_n_channels = n_channels[0]
        static_n_channels = n_channels[1]
    else:
        x_n_channels = n_channels

    x_in = Input(shape=(time_window, None, None, x_n_channels))
    if backbone_block == 'convnet':
        skipcon = None
    elif backbone_block == 'resnet':
        skipcon = 'residual'
    elif backbone_block == 'densenet':
        skipcon = 'dense'
    x = b = RecurrentConvBlock(n_filters, output_full_sequence=False, skip_connection_type=skipcon,  
                               normalization=normalization)(x_in)

    if static_arr:
        s_in = Input(shape=(None, None, static_n_channels))
        x = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(x)
        x = Concatenate()([x, s_in])
        b = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(b)
        b = Concatenate()([b, s_in])

    for i in range(n_blocks):
        if backbone_block == 'convnet':
            b = ConvBlock(n_filters, normalization=normalization, attention=attention)(b)
        elif backbone_block == 'resnet':
            b = ResidualBlock(n_filters, normalization=normalization, attention=attention)(b)
        elif backbone_block == 'densenet':
            b = DenseBlock(n_filters, normalization=normalization, attention=attention)(b)
            b = TransitionBlock(n_filters // 2)(b)  # another option: half of the DenseBlock channels
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    
    if backbone_block == 'convnet':
        x = b
    elif backbone_block == 'resnet':
        x = Add()([x, b])
    elif backbone_block == 'densenet':
        x = Concatenate()([x, b])
    
    x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)

    model_name = 'rec' + backbone_block + '_pin' 
    if static_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)
    else:
        return Model(inputs=x_in, outputs=x, name=model_name)
