from tensorflow.keras.layers import Add, Conv2D, Input, concatenate
from tensorflow.keras.models import Model

from .blocks import residual_block, recurrent_block


def resnet_bi(
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    attention=False,
    output_activation=None):
    """
    ResNet-INT. ResNet with EDSR residual blocks and pre-upsampling via (bicubic)
    interpolation.

    The interpolation method depends on the ``interpolation`` argument used in
    the training procedure (which is passed to the DataGenerator).
    """
    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, attention=attention)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    
    x = Conv2D(n_channels_out, (3, 3), padding='same', 
               activation=output_activation)(x)

    return Model(inputs=x_in, outputs=x, name="resnet_bi")


def recresnet_bi(
    n_channels, 
    n_filters, 
    n_res_blocks, 
    n_channels_out=1, 
    time_window=None, 
    attention=False,
    output_activation=None):
    """
    Recurrent ResNet-INT. Recurrent Residual network with EDSR residual blocks 
    and pre-upsampling via interpolation.
    """
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
    
    x = Conv2D(n_channels_out, (3, 3), padding='same', 
               activation=output_activation)(x)

    if static_arr:
        return Model(inputs=[x_in, s_in], outputs=x, name="recresnet_bi")
    else:
        return Model(inputs=x_in, outputs=x, name="recresnet_bi")
