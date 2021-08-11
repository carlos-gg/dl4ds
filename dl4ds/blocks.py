import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, ConvLSTM2D, SeparableConv2D, 
                                     BatchNormalization, LayerNormalization, 
                                     Activation, SpatialDropout2D, Conv2DTranspose, 
                                     SpatialDropout3D, Concatenate)
from tensorflow_addons.layers import (WeightNormalization, SpectralNormalization,
                                      InstanceNormalization)

from .attention import ChannelAttention2D


class ConvBlock(tf.keras.layers.Layer): 
    """Convolutional block.
    """
    def __init__(self, filters, strides=1, ks_cl1=(3,3), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, depthwise_separable=False,
                 **conv_kwargs):
        super().__init__()
        self.normalization = normalization
        self.attention = attention
        self.dropout_variant = dropout_variant
        self.dropout_rate = dropout_rate
        self.depthwise_separable = depthwise_separable
        if self.depthwise_separable:
            self.conv1 = SeparableConv2D(filters, kernel_size=ks_cl1, padding='same', strides=strides, **conv_kwargs)
            self.conv2 = SeparableConv2D(filters, kernel_size=ks_cl2, padding='same', **conv_kwargs)
        else:
            self.conv1 = Conv2D(filters, kernel_size=ks_cl1, padding='same', strides=strides, **conv_kwargs)
            self.conv2 = Conv2D(filters, kernel_size=ks_cl2, padding='same', **conv_kwargs)

        if self.normalization == 'bn':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        elif self.normalization == 'ln':
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()
        elif self.normalization == 'wn':
            self.norm1 = WeightNormalization()
            self.norm2 = WeightNormalization()
        elif self.normalization == 'sn':
            self.norm1 = SpectralNormalization()
            self.norm2 = SpectralNormalization()
        elif self.normalization == 'in':
            self.norm1 = InstanceNormalization()
            self.norm2 = InstanceNormalization()

        if self.attention:
            self.att = ChannelAttention2D(filters)
        self.activation = Activation(activation)
        self.apply_dropout = False
        # Only spatial dropout is applied inside convolutional blocks
        if self.dropout_variant == 'spatial' and self.dropout_rate > 0:
            self.apply_dropout = True
            self.dropout1 = SpatialDropout2D(self.dropout_rate)
            self.dropout2 = SpatialDropout2D(self.dropout_rate)

    def call(self, X):
        """Model's forward pass.
        """
        Y = self.dropout1(X) if self.apply_dropout else X
        Y = self.conv1(Y)
        if self.normalization is not None:
            Y = self.norm1(Y)
        Y = self.activation(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y)
        Y = self.conv2(Y)
        if self.normalization is not None:
            Y = self.norm2(Y)
        Y = self.activation(Y)
        if self.attention:
            Y = self.att(Y)
        return Y


class ResidualBlock(ConvBlock): 
    """
    Residual block. Two examples:
    * Standard residual block [1]: Conv2D -> BN -> ReLU -> Conv2D -> BN -> Add _> ReLU
    * EDSR-style block: Conv2D -> ReLU -> Conv2D -> Add -> ReLU

    References
    ----------
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
    """
    def __init__(self, filters, strides=1, ks_cl1=(3,3), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, **conv_kwargs):
        super().__init__(filters, strides, ks_cl1, ks_cl2, activation, 
                         normalization, attention, dropout_rate, 
                         dropout_variant, **conv_kwargs)

    def call(self, X):
        Y = self.dropout1(X) if self.apply_dropout else X
        Y = self.conv1(Y)
        if self.normalization is not None:
            Y = self.norm1(Y)
        Y = self.activation(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y) 
        Y = self.conv2(Y)
        if self.normalization is not None:
            Y = self.norm2(Y)
        if self.attention:
            Y = self.att(Y)
        Y += X
        Y = self.activation(Y)
        return Y


class DenseBlock(ConvBlock): 
    """
    Dense block [2].

    References
    ----------
    [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Networks: https://arxiv.org/abs/1608.06993
    """
    def __init__(self, filters, strides=1, ks_cl1=(1,1), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, **conv_kwargs):
        super().__init__(filters, strides, ks_cl1, ks_cl2, activation, 
                         normalization, attention, dropout_rate, 
                         dropout_variant, **conv_kwargs)
        self.conv1 = Conv2D(4 * filters, padding='same', kernel_size=ks_cl1, strides=strides, **conv_kwargs)
        self.conv2 = Conv2D(filters, kernel_size=ks_cl2, padding='same', **conv_kwargs)
        self.concat = Concatenate()

    def call(self, X):
        Y = self.norm1(X) if self.normalization is not None else X
        Y = self.activation(Y)
        if self.apply_dropout:
            Y = self.dropout1(Y)
        Y = self.conv1(X)       
        if self.normalization is not None:
            Y = self.norm2(Y)
        Y = self.activation(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y)
        Y = self.conv2(Y)
        if self.attention:
            Y = self.att(Y)
        Y = self.concat([Y, X])
        return Y


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.batch_norm = BatchNormalization()
        self.activation = Activation(activation)
        self.conv = Conv2D(filters, kernel_size=1)

    def call(self, X):
        Y = self.batch_norm(X)
        Y = self.activation(Y)
        Y = self.conv(Y)
        return Y


class RecurrentConvBlock(tf.keras.layers.Layer): 
    """Recurrent convolutional block with a skip connection.
    """
    def __init__(self, filters, output_full_sequence, skip_connection_type=None, 
                 ks_cl1=(3,3), ks_cl2=(3,3), ks_cl3=(3,3), activation='relu', 
                 normalization=None, dropout_rate=0, dropout_variant=None, 
                 **conv_kwargs):
        super().__init__()
        self.normalization = normalization
        self.output_full_sequence = output_full_sequence
        self.skip_connection_type = skip_connection_type
        self.dropout_rate = dropout_rate
        self.dropout_variant = dropout_variant
        if self.skip_connection_type is not None:
            self.convlstm0 = ConvLSTM2D(filters, kernel_size=ks_cl1, return_sequences=True, padding='same', **conv_kwargs)
        self.convlstm1 = ConvLSTM2D(filters, kernel_size=ks_cl1, return_sequences=True, padding='same', **conv_kwargs)
        self.convlstm2 = ConvLSTM2D(filters, kernel_size=ks_cl2, return_sequences=True, padding='same', **conv_kwargs)
        if not self.output_full_sequence:
            self.convlstm3 = ConvLSTM2D(filters, kernel_size=ks_cl3, return_sequences=False, padding='same', **conv_kwargs)
 
        if self.normalization == 'bn':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        elif self.normalization == 'ln':
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()
        elif self.normalization == 'wn':
            self.norm1 = WeightNormalization()
            self.norm2 = WeightNormalization()
        elif self.normalization == 'sn':
            self.norm1 = SpectralNormalization()
            self.norm2 = SpectralNormalization()
        elif self.normalization == 'in':
            self.norm1 = InstanceNormalization()
            self.norm2 = InstanceNormalization()

        self.activation = Activation(activation)
        if self.skip_connection_type == 'residual':
            self.skipcon = Add()
        elif self.skip_connection_type == 'dense':
            self.skipcon = Concatenate()
        # Only spatial dropout is applied inside convolutional blocks
        if self.dropout_variant == 'spatial' and self.dropout_rate > 0:
            self.apply_dropout = True
            self.dropout1 = SpatialDropout3D(self.dropout_rate)
            self.dropout2 = SpatialDropout3D(self.dropout_rate)

    def call(self, X):
        """Model's forward pass. 
        """
        if self.skip_connection_type is not None:
            Y_c = self.convlstm0(X)
            Y = self.convlstm1(Y_c)
        else:
            Y = self.convlstm1(X)
        
        if self.normalization is not None:
            Y = self.norm1(Y)
        
        Y = self.activation(Y)
        
        Y = self.convlstm2(Y)
        
        if self.normalization is not None:
            Y = self.norm2(Y)
        
        if self.skip_connection_type is not None:
            Y = self.skipcon([Y, Y_c])
        
        Y = self.activation(Y)
        
        if not self.output_full_sequence:
            Y = self.convlstm3(Y)
        return Y


class SubpixelConvolution(tf.keras.layers.Layer):
    """
    """
    def __init__(self, scale, n_filters, **kwargs):
        """
        See: https://arxiv.org/abs/1609.05158
        """
        super().__init__()
        self.scale = scale
        self.n_filters = n_filters
        self.conv = Conv2D(self.n_filters * (self.scale ** 2), 3, padding='same', **kwargs)
        self.conv2x = Conv2D(self.n_filters * (2 ** 2), 3, padding='same', **kwargs)
        self.conv5x = Conv2D(self.n_filters * (5 ** 2), 3, padding='same', **kwargs)


    def upsample_conv(self, x, factor):
        """Sub-pixel convolution (pixel shuffle)
        """
        if factor == 2:
            x = self.conv2x(x)
        elif factor == 5:
            x = self.conv5x(x)
        else:
            x = self.conv(x)
        return tf.nn.depth_to_space(x, factor)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[1] * self.scale), 
                int(input_shape[2] * self.scale), input_shape[3])

    def call(self, x):
        """ """
        if self.scale == 2:
            x = self.upsample_conv(x, 2)
        elif self.scale == 4:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
        elif self.scale == 8:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
        elif self.scale == 10:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 5)
        elif self.scale == 20:
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 2)
            x = self.upsample_conv(x, 5)
        else:
            x = self.upsample_conv(x, self.scale)
        return x


class Deconvolution(tf.keras.layers.Layer):
    """
    FSRCNN: https://arxiv.org/abs/1608.00367
    """
    def __init__(self, scale, n_filters, output_activation=None):
        """
        """
        super().__init__()
        self.scale = scale
        self.output_activation = output_activation
        self.conv2dtranspose1 = Conv2DTranspose(n_filters, (9, 9), strides=(2, 2), 
            padding='same', name='deconv_1of2_scale_x2', use_bias=False)
        self.conv2dtranspose2 = Conv2DTranspose(n_filters, (9, 9), strides=(2, 2), 
            padding='same', name='deconv_2of2_scale_x2', 
            activation=output_activation, use_bias=False)
        self.conv2dtranspose = Conv2DTranspose(n_filters, (9, 9), 
            strides=(self.scale, self.scale), padding='same', 
            name='deconv_scale_x' + str(self.scale), 
            activation=output_activation, use_bias=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[1] * self.scale), 
                int(input_shape[2] * self.scale), input_shape[3])

    def call(self, x):
        """
        """
        if self.scale == 4:
            x = self.conv2dtranspose1(x)
            x = self.conv2dtranspose2(x)
        else:
            x = self.conv2dtranspose(x)
        return x
