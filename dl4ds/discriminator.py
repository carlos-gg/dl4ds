import tensorflow as tf
from tensorflow.keras.layers import (Input, Dropout, Dense, Conv2D, Add, 
                                     Concatenate, GlobalAveragePooling2D, 
                                     GlobalAveragePooling3D, Cropping2D)

from .blocks import ResidualBlock, RecurrentConvBlock
from . import POSTUPSAMPLING_METHODS


def residual_discriminator(
    n_channels, 
    model, 
    scale, 
    n_filters, 
    n_res_blocks, 
    normalization=None, 
    activation='relu',
    attention=False,
    return_sequence=False):
    """
    """
    upsampling = model.split('_')[-1]

    if return_sequence:
        x_in = Input(shape=(None, None, None, n_channels))    
    else:
        x_in = Input(shape=(None, None, n_channels))    

    backbone_block = model.split('_')[0]
    if backbone_block.startswith('rec'):
        backbone_block = backbone_block[3:]
    
    if return_sequence:
        if backbone_block == 'convnet':
            skipcon = None
            n_filters_recblock1 = n_filters
        elif backbone_block == 'resnet':
            skipcon = 'residual'
            n_filters_recblock1 = n_filters
        elif backbone_block == 'densenet':
            skipcon = 'dense'
            n_filters_recblock1 = n_filters // 2

        x_1 = b = RecurrentConvBlock(
            n_filters_recblock1, 
            output_full_sequence=True, 
            skip_connection_type=skipcon,  
            activation=activation, 
            normalization='ln',
            dropout_rate=0)(x_in)
    else:
        x_1 = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)

    for i in range(n_res_blocks):
        b = ResidualBlock(n_filters, normalization=normalization, attention=attention)(b)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x_1 = Add()([x_1, b])
    
    # Branch with the HR reference or HR generated array
    if return_sequence:
        x_ref = Input(shape=(None, None, None, 1))
    else:
        x_ref = Input(shape=(None, None, 1))    
    x_2 = c = Conv2D(n_filters, (3, 3), padding='same')(x_ref)
    for i in range(n_res_blocks):
        c = ResidualBlock(n_filters, normalization=normalization, attention=attention)(c)

    if upsampling in POSTUPSAMPLING_METHODS:  
        if scale == 5:      
            c = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Cropping2D(cropping=((0,1),(0,1)))(x_2)
        elif scale == 4:
            c = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
    elif model in ['resnet_bi', 'recresnet_bi']:
        c = Conv2D(n_filters, (3, 3), padding='same')(c)
        x_2 = Add()([x_2, c])

    x = Concatenate()([x_1, x_2])
    
    x = ResidualBlock(x.shape[-1], normalization=normalization, attention=attention)(x)
    
    # global average pooling operation for spatial data
    if return_sequence:
        x = GlobalAveragePooling3D()(x)
    else:
        x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='sigmoid')(x)
    output = Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([x_in, x_ref], output, name="discriminator")