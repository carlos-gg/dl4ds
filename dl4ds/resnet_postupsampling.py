import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Input, Lambda, UpSampling2D, 
                                     concatenate, Conv2DTranspose)
from tensorflow.keras.models import Model

from .blocks import residual_block, recurrent_block


def resnet_postupsampling(
    upsampling,
    scale, 
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    attention=False,
    output_activation=None):
    """
    Residual network with different post-upsampling modules (depending on the 
    argument ``upsampling``):

    * ResNet-SPC. ResNet with EDSR-style residual blocks and pixel shuffle 
    post-upscaling
    * ResNet-RC. ResNet with EDSR residual blocks and resize convolution (via 
    bilinear interpolation) in post-upsampling
    * ResNet-DC. ResNet with EDSR residual blocks and transposed convolution (or 
    deconvolution) in post-upsampling
    """
    if not isinstance(upsampling, str) and upsampling in ['spc', 'rc', 'dc']:
        raise ValueError('Unknown upsampling module')

    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, attention=attention)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    
    if upsampling == 'spc':
        x = subpixel_convolution_layer(x, scale, n_filters)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
        model_name = 'resnet_spc'
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
        model_name = 'resnet_rc'
    elif upsampling == 'dc':
        x = deconvolution_layer(x, scale, output_activation)
        model_name = 'resnet_dc'      
    
    return Model(inputs=x_in, outputs=x, name=model_name)  

def recresnet_postupsampling(
    upsampling,
    scale, 
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    time_window=None, 
    attention=False,
    output_activation=None):
    """
    Recurrent residual network different post-upsampling modules (depending on the 
    argument ``upsampling``):

    * Recurrent ResNet-SPC. Recurrent Residual Network with EDSR-style residual 
    blocks and pixel shuffle post-upscaling
    * Recurrent ResNet-RC. Recurrent Residual Network with EDSR residual blocks 
    and resize convolution (via bilinear interpolation) in post-upsampling
    * Recurrent ResNet-RC. Recurrent Residual Network with EDSR residual blocks 
    and transposed convolution (or deconvolution) in post-upsampling
    """
    if not isinstance(upsampling, str) and upsampling in ['spc', 'rc', 'dc']:
        raise ValueError('Unknown upsampling module')
        
    static_arr = True if isinstance(n_channels, tuple) else False
    if static_arr:
        x_n_channels = n_channels[0]
        static_n_channels = n_channels[1]
    else:
        x_n_channels = n_channels

    x_in = Input(shape=(time_window, None, None, x_n_channels))
    x = b =recurrent_block(x_in, n_filters, full_sequence=False)

    if static_arr:
        s_in = Input(shape=(None, None, static_n_channels))
        x = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(x)
        x = concatenate([x, s_in])
        b = Conv2D(n_filters - static_n_channels, (1, 1), padding='same')(b)
        b = concatenate([b, s_in])

    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, attention=attention)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    
    if upsampling == 'spc':
        x = subpixel_convolution_layer(x, scale, n_filters)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
        model_name = 'recresnet_spc'
    elif upsampling == 'rc':
        x = UpSampling2D(scale, interpolation='bilinear')(x)
        x = Conv2D(n_channels_out, (3, 3), padding='same', 
                   activation=output_activation)(x)
        model_name = "recresnet_rc"
    elif upsampling == 'dc':
        x = deconvolution_layer(x, scale, output_activation)
        model_name = "recresnet_dc"

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