import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, Input, Add, Lambda

from .blocks import residual_block


def metasr(n_channels, n_filters, n_res_blocks, n_channels_out=1, meta_ksize=(3,3)):
    """
    EDSR with MetaUpsample module
    """
    x_in = Input(shape=(None, None, n_channels))
    x = b = Conv2D(n_filters, (1, 1), padding='same')(x_in)
    for i in range(n_res_blocks):
        b = residual_block(b, n_filters)
    x = Add()([x, b])

    coord = Input((None, None, 3))
    meta_w = Dense(256, activation="relu")(coord)
    meta_w = Dense(meta_ksize[0] * meta_ksize[1] * n_filters * n_channels)(meta_w)
    x = MetaUpSample(n_channels_out, meta_ksize)([x, meta_w])

    model = Model([x_in, coord], [x], name='metasr')
    return model


class MetaUpSample(Layer):
    def __init__(self, filters, ksize, **kwargs):
        self.filters = filters
        self.ksize = ksize
        super(MetaUpSample, self).__init__(**kwargs)

    def get_config(self):
        """ 
        We override get_config method because it receives new arguments, so we
        can save a model with this layer
        """
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'ksize': self.ksize})
        return config
        
    def build(self, input_shape):
        super(MetaUpSample, self).build(input_shape)

    def call(self, inputs):
        x, meta_w = inputs
        w_shape = tf.shape(meta_w)
        x_shape = tf.shape(x)

        # Get projection positions 
        indices = tf.meshgrid(tf.range(w_shape[0]),tf.range(w_shape[1]),tf.range(w_shape[2]))
        indices = tf.reshape(tf.transpose(indices, [2,1,3,0]), [-1,3])
        indices = tf.cast(indices, "float32")
        b_idx, h_idx, w_idx = tf.split(indices, num_or_size_splits=3, axis=-1)
        h_idx = tf.cast(x_shape[1], "float32") * (h_idx / tf.cast(w_shape[1], "float32"))
        w_idx = tf.cast(x_shape[2], "float32") * (w_idx / tf.cast(w_shape[2], "float32"))
        indices = tf.concat([b_idx, h_idx, w_idx], axis=-1)
        indices = tf.cast(indices, "int32")

        meta_w = tf.reshape(meta_w,[w_shape[0],w_shape[1],w_shape[2],w_shape[3]//self.filters,self.filters])

        y = tf.image.extract_patches(x, sizes=[1, self.ksize[0], self.ksize[1], 1], 
                                     strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        y = tf.gather_nd(y, indices)
        y = tf.reshape(y, [w_shape[0], w_shape[1], w_shape[2], w_shape[3]//self.filters, 1])
        # Matrix multiplication
        y = tf.reduce_sum(y * meta_w, axis=-2)
        return y


def get_coords(hr_size, lr_size, scale):
    """ 
    """
    # scaling factor between the sizes of the LR and HR images
    scale_y = float(hr_size[0])/lr_size[0]
    scale_x = float(hr_size[1])/lr_size[1]
    # multi-dimensional grid of coordinates, with the size of the HR image
    coords = np.mgrid[0:hr_size[0], 0:hr_size[1]]    
    coords = coords.astype("float32")
    coords = np.transpose(coords, [1, 2, 0])  ## transposing   
    coords[:,:,0] = (coords[:,:,0]/scale_y) % 1
    coords[:,:,1] = (coords[:,:,1]/scale_x) % 1    
    coords = np.concatenate([coords, np.ones((hr_size[0], hr_size[1],1),"float32") / scale], 
                            axis=-1)   
    return coords