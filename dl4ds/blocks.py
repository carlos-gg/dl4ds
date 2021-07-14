import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, ConvLSTM2D, BatchNormalization, 
                                     LayerNormalization, Activation, 
                                     SpatialDropout2D, Concatenate)
from .attention import ChannelAttention2D


class ConvBlock(tf.keras.Model): 
    """Convolutional block.
    """
    def __init__(self, filters, strides=1, ks_cl1=(3,3), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, **conv_kwargs):
        super().__init__()
        self.normalization = normalization
        self.attention = attention
        self.dropout_variant = dropout_variant
        self.dropout_rate = dropout_rate
        self.conv1 = Conv2D(filters, padding='same', kernel_size=ks_cl1, strides=strides, **conv_kwargs)
        self.conv2 = Conv2D(filters, kernel_size=ks_cl2, padding='same', **conv_kwargs)
        if self.normalization is not None:
            if self.normalization == 'bn':
                self.norm1 = BatchNormalization()
                self.norm2 = BatchNormalization()
            elif self.normalization == 'ln':
                self.norm1 = LayerNormalization()
                self.norm2 = LayerNormalization()
        if self.attention:
            self.att = ChannelAttention2D(filters)
        self.relu = Activation(activation)
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
        Y = self.relu(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y)
        Y = self.conv2(Y)
        if self.normalization is not None:
            Y = self.norm2(Y)
        Y = self.relu(Y)
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
        Y = self.relu(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y) 
        Y = self.conv2(Y)
        if self.normalization is not None:
            Y = self.norm2(Y)
        if self.attention:
            Y = self.att(Y)
        Y += X
        Y = self.relu(Y)
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
        Y = self.relu(Y)
        if self.apply_dropout:
            Y = self.dropout1(Y)
        Y = self.conv1(X)       
        if self.normalization is not None:
            Y = self.norm2(Y)
        Y = self.relu(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y)
        Y = self.conv2(Y)
        Y = self.concat([Y, X])
        if self.attention:
            Y = self.att(Y)
        return Y


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.batch_norm = BatchNormalization()
        self.relu = Activation(activation)
        self.conv = Conv2D(filters, kernel_size=1)

    def call(self, X):
        Y = self.batch_norm(X)
        Y = self.relu(Y)
        Y = self.conv(Y)
        return Y


class RecurrentConvBlock(tf.keras.Model): 
    """Recurrent convolutional block with a skip connection.
    """
    def __init__(self, filters, output_full_sequence, skip_connection_type=None, 
                 ks_cl1=(3,3), ks_cl2=(3,3), ks_cl3=(3,3), activation='relu', 
                 normalization=None, **conv_kwargs):
        super().__init__()
        self.normalization = normalization
        self.output_full_sequence = output_full_sequence
        self.skip_connection_type = skip_connection_type
        self.convlstm1 = ConvLSTM2D(filters, kernel_size=ks_cl1, return_sequences=True, padding='same', **conv_kwargs)
        self.convlstm2 = ConvLSTM2D(filters, kernel_size=ks_cl2, return_sequences=True, padding='same', **conv_kwargs)
        self.convlstm3 = ConvLSTM2D(filters, kernel_size=ks_cl3, return_sequences=True, padding='same', **conv_kwargs)
        if not self.output_full_sequence:
            self.convlstm4 = ConvLSTM2D(filters, kernel_size=ks_cl3, return_sequences=False, padding='same', **conv_kwargs)
        if self.normalization is not None:
            if self.normalization == 'bn':
                self.norm = BatchNormalization()
            elif self.normalization == 'ln':
                self.norm = LayerNormalization()
        self.relu = Activation(activation)
        if self.skip_connection_type == 'residual':
            self.skipcon = Add()
        elif self.skip_connection_type == 'dense':
            self.skipcon = Concatenate()

    def call(self, X):
        """Model's forward pass. Closer to the structre of the residual block.
        """
        Y_c = self.convlstm1(X)
        Y = self.convlstm2(Y_c)
        if self.normalization is not None:
            Y = self.norm(Y)
        Y = self.relu(Y)
        Y = self.convlstm3(Y)
        if self.normalization is not None:
            Y = self.norm(Y)
        if self.skip_connection_type is not None:
            Y = self.skipcon([Y, Y_c])
        Y = self.relu(Y)
        if not self.output_full_sequence:
            Y = self.convlstm4(Y)
        return Y

