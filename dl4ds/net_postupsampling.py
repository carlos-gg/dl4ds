import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, Lambda, UpSampling2D, 
                                     Concatenate, Conv2DTranspose)
from tensorflow.keras.models import Model

from .blocks import (recurrent_residual_block, ResidualBlock, ConvBlock, 
                     DenseBlock, TransitionBlock)
from .utils import checkarg_backbone, checkarg_upsampling


def net_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    normalization=None,
    attention=False,
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
    """
    backbone_block = checkarg_backbone(backbone_block)
    upsampling = checkarg_upsampling(upsampling)

    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_res_blocks):
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
    
    model_name = backbone_block + '_' + upsampling
    if upsampling == 'spc':
        x = subpixel_convolution_layer(x, scale, n_filters)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', activation=output_activation)(x)
    elif upsampling == 'dc':
        x = deconvolution_layer(x, scale, output_activation)
    
    return Model(inputs=x_in, outputs=x, name=model_name)  


def recnet_postupsampling(
    backbone_block,
    upsampling,
    scale, 
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    time_window=None, 
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
        
    static_arr = True if isinstance(n_channels, tuple) else False
    if static_arr:
        x_n_channels = n_channels[0]
        static_n_channels = n_channels[1]
    else:
        x_n_channels = n_channels

    x_in = Input(shape=(time_window, None, None, x_n_channels))
    x = b = recurrent_residual_block(x_in, n_filters, full_sequence=False)

    if static_arr:
        s_in = Input(shape=(None, None, static_n_channels))
        x = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(x)
        x = Concatenate()([x, s_in])
        b = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(b)
        b = Concatenate()([b, s_in])

    for i in range(n_res_blocks):
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
    
    if upsampling == 'spc':
        x = subpixel_convolution_layer(x, scale, n_filters)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
    elif upsampling == 'dc':
        x = deconvolution_layer(x, scale, output_activation)

    model_name = 'rec' + backbone_block + '_' + upsampling
    if static_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name=model_name)
    else:
        return Model(inputs=x_in, outputs=x, name=model_name)


def subpixel_convolution_layer(x, scale, n_filters):
    def upsample_conv(x, factor, **kwargs):
        """Sub-pixel convolution
        """
        x = Conv2D(n_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_conv(x, 2, name='conv2d_scale_2')
    elif scale == 4:
        x = upsample_conv(x, 2, name='conv2d_1of2_scale_2')
        x = upsample_conv(x, 2, name='conv2d_2of2_scale_2')
    elif scale == 20:
        x = upsample_conv(x, 5, name='conv2d_1of2_scale_5')
        x = upsample_conv(x, 4, name='conv2d_2of2_scale_4')
    else:
        x = upsample_conv(x, scale, name='conv2d_scale_' + str(scale))
    return x


def pixel_shuffle(scale):
    """
    See: https://arxiv.org/abs/1609.05158
    """
    return lambda x: tf.nn.depth_to_space(x, scale)


def deconvolution_layer(x, scale, output_activation):
    """
    FSRCNN: https://arxiv.org/abs/1608.00367
    """
    if scale == 4:
        x = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same', 
                            name='deconv_1of2_scale_2')(x)
        x = Conv2DTranspose(1, (9, 9), strides=(2, 2), padding='same', 
                            name='deconv_2of2_scale_2', 
                            activation=output_activation)(x)
    elif scale == 20:
        x = Conv2DTranspose(1, (9, 9), strides=(4, 4), padding='same', 
                            name='deconv_1of2_scale_5')(x)
        x = Conv2DTranspose(1, (9, 9), strides=(5, 5), padding='same', 
                            name='deconv_2of2_scale_4', 
                            activation=output_activation)(x)
    else:
        x = Conv2DTranspose(1, (9, 9), strides=(scale, scale), padding='same', 
                            name='deconv_scale_' + str(scale, 
                            activation=output_activation))(x)
    return x