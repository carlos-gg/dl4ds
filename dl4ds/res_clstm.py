import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda, ConvLSTM2D, concatenate, LocallyConnected2D
from tensorflow.keras.models import Model

from .blocks import residual_block
from .resnet_spc import upsample


def rclstm_spc(scale, n_channels, n_filters, n_res_blocks, lr_height_width, n_channels_out=1, upsampling=True):
    """
    Residual ConvLSTM SPC. ConvLSTM layer followed by residual blocks (with 
    BatchNorm) and pixel shuffle post-upscaling
    """
    static_arr = True if isinstance(n_channels, tuple) else False
    
    if static_arr:
        x_n_channels = n_channels[0]
        static_n_channels = n_channels[1]
    else:
        x_n_channels = n_channels

    # no cropping
    if lr_height_width is not None:  
        h_lr = lr_height_width[0]
        w_lr = lr_height_width[1]
        h_hr = h_lr * scale
        w_hr = w_lr * scale
    # cropping square patches
    else:  
        h_lr = None
        w_lr = None
        h_hr = None
        w_hr = None

    x_in = Input(shape=(None, h_lr, w_lr, x_n_channels))
    if static_arr:
        s_in = Input(shape=(h_hr, w_hr, static_n_channels))
 
    x = b = ConvLSTM2D(n_filters, (5, 5), return_sequences=False, padding='same')(x_in) #  dropout=0.2

    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, batchnorm=True)
    x = Add()([x, b])

    if upsampling:
        x = upsample(x, scale, n_filters)

    if isinstance(n_channels, tuple):
        x = concatenate([x, s_in])
    
    x = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(n_channels_out, (3, 3), padding='same')(x)
    
    # x = LocallyConnected2D(n_channels_out, (1, 1), padding='valid', bias_initializer='zeros', use_bias=False)(x)

    if static_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name="rclstm_spc")
    else:
        return Model(inputs=x_in, outputs=x, name="rclstm_spc")