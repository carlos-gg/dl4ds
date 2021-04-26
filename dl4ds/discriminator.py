import tensorflow as tf
from tensorflow.keras.layers import (Input, Dropout, Dense, Conv2D, Add, concatenate, 
                                     GlobalAveragePooling2D, Cropping2D, AveragePooling2D)
from .blocks import residual_block


def residual_discriminator(n_channels, n_filters, n_res_blocks, model, scale, attention=False):
    """
    """
    x_in = Input(shape=(None, None, n_channels))    
    x_1 = b = Conv2D(n_filters, (3, 3), padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters, attention=attention)
    b = Conv2D(n_filters, (3, 3), padding='same')(b)
    x_1 = Add()([x_1, b])
    
    x_ref = Input(shape=(None, None, 1))    
    x_2 = c = Conv2D(n_filters, (3, 3), padding='same')(x_ref)
    for i in range(n_res_blocks):
        c = residual_block(c, n_filters, attention=attention)

    if model in ['resnet_spc', 'resnet_rec']:  
        # c = AveragePooling2D(scale)(c)
        if scale == 5:      
            c = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='valid', strides=(2,2))(c)
            x_2 = Cropping2D(cropping=((0,1),(0,1)))(x_2)
        elif scale == 4:
            c = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
            x_2 = Conv2D(n_filters, (3, 3), padding='same', strides=(2,2))(c)
    elif model == 'resnet_int':
        c = Conv2D(n_filters, (3, 3), padding='same')(c)
        x_2 = Add()([x_2, c])

    x = concatenate([x_1, x_2])
    
    x = residual_block(x, x.shape[-1], attention=attention)
    
    # global average pooling operation for spatial data
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='sigmoid')(x)
    output = Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([x_in, x_ref], output, name="discriminator")