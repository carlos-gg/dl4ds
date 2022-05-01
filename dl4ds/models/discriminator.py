import tensorflow as tf
from tensorflow.keras.layers import (Input, Dropout, Dense, Conv2D, Add, 
                                     Concatenate, GlobalAveragePooling2D, 
                                     GlobalAveragePooling3D, Cropping2D,
                                     Resizing)

from .blocks import ResidualBlock, RecurrentConvBlock
from .. import POSTUPSAMPLING_METHODS


def residual_discriminator(
    n_channels, 
    upsampling, 
    is_spatiotemporal,
    scale, 
    lr_size,
    # ----- below are parameters that shall be tweaked by the user -----
    n_filters=8, 
    n_res_blocks=4, 
    normalization=None, 
    activation='relu',
    attention=False):
    """
    """
    # Branch with the LR input array
    if is_spatiotemporal:
        x_in = Input(shape=(None, None, None, n_channels))    
    else:
        x_in = Input(shape=(None, None, n_channels))    
    
    if is_spatiotemporal:
        x_1 = b = RecurrentConvBlock(n_filters, activation=activation, 
            normalization='ln', dropout_rate=0)(x_in)
    else:
        x_1 = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)

    for i in range(n_res_blocks):
        b = ResidualBlock(n_filters, normalization=normalization, 
            attention=attention, name=f'ResidualBlock{str(i + 1)}_branch1')(b)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x_1 = Add()([x_1, b])
    
    # Branch with the HR reference or HR generated array
    if is_spatiotemporal:
        x_ref = Input(shape=(None, None, None, 1))
    else:
        x_ref = Input(shape=(None, None, 1))    
    x_2 = c = Conv2D(n_filters, (3, 3), padding='same')(x_ref)
    for i in range(n_res_blocks):
        c = ResidualBlock(n_filters, normalization=normalization, 
            attention=attention, name=f'ResidualBlock{str(i + 1)}_branch2')(c)

    if upsampling in POSTUPSAMPLING_METHODS:  
        if scale == 5:      
            c = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Cropping2D(cropping=((0,1),(0,1)))(x_2)
        elif scale == 4:
            c = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
        else:
            x_2 = Resizing(lr_size[0], lr_size[1], interpolation='bilinear', 
                           name='InterpolationDownsampling')(c)
    elif upsampling == 'pin':
        c = Conv2D(n_filters, (3, 3), padding='same')(c)
        x_2 = Add()([x_2, c])

    x = Concatenate(name='Concat2Branches')([x_1, x_2])
    
    x = ResidualBlock(x.shape[-1], normalization=normalization, attention=attention)(x)
    
    # global average pooling operation for spatial data
    if is_spatiotemporal:
        x = GlobalAveragePooling3D(name='GlobalAveragePooling')(x)
    else:
        x = GlobalAveragePooling2D(name='GlobalAveragePooling')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='sigmoid')(x)
    output = Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([x_in, x_ref], output, name="discriminator")