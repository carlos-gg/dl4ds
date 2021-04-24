from tensorflow.keras.layers import Add, Conv2D, Input
from tensorflow.keras.models import Model

from .blocks import residual_block


def resnet_int(n_channels, n_filters, n_res_blocks, n_channels_out=1, attention=False):
    """
    ResNet-INT. ResNet with EDSR residual blocks and pre-upsampling via interpolation.
    """
    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, attention=attention)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    
    x = Conv2D(n_channels_out, (3, 3), padding='same')(x)

    return Model(x_in, x, name="resnet_int")