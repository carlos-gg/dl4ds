import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, ConvLSTM2D, Concatenate,
                                     SeparableConv2D, BatchNormalization, 
                                     LayerNormalization, Activation, 
                                     Dropout, GaussianDropout,
                                     SpatialDropout2D, Conv2DTranspose, 
                                     SpatialDropout3D, LocallyConnected2D,
                                     ZeroPadding2D, MaxPooling2D, UpSampling2D)


class ConvBlock(tf.keras.layers.Layer): 
    """
    Convolutional block.

    References
    ----------
    [1] Effective and Efficient Dropout for Deep Convolutional Neural Networks: 
    https://arxiv.org/abs/1904.03392
    [2] Rethinking the Usage of Batch Normalization and Dropout: 
    https://arxiv.org/abs/1905.05928
    """
    def __init__(self, filters, strides=1, ks_cl1=(3,3), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, 
                 depthwise_separable=False, name=None, **conv_kwargs):
        super().__init__(name=name)
        self.normalization = normalization
        self.attention = attention
        self.dropout_variant = dropout_variant
        self.dropout_rate = dropout_rate
        self.depthwise_separable = depthwise_separable
        if self.depthwise_separable:
            self.conv1 = SeparableConv2D(
                filters, 
                kernel_size=ks_cl1, 
                padding='same', 
                strides=strides, 
                use_bias=True if self.normalization is None else False,
                **conv_kwargs)
            self.conv2 = SeparableConv2D(
                filters, 
                kernel_size=ks_cl2, 
                padding='same', 
                use_bias=True if self.normalization is None else False,
                **conv_kwargs)
        else:
            self.conv1 = Conv2D(
                filters, 
                kernel_size=ks_cl1, 
                padding='same', 
                strides=strides, 
                use_bias=True if self.normalization is None else False,
                **conv_kwargs)
            self.conv2 = Conv2D(
                filters, 
                kernel_size=ks_cl2, 
                padding='same', 
                use_bias=True if self.normalization is None else False,
                **conv_kwargs)

        if self.normalization is not None:
            if self.normalization not in ['bn', 'ln']:
                raise ValueError(f'Normalization not supported, got {self.normalization}')
        if self.normalization == 'bn':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        elif self.normalization == 'ln':
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()

        if self.attention:
            self.att = ChannelAttention2D(filters)
        self.activation = Activation(activation)
        
        self.apply_dropout = False
        if self.dropout_rate > 0:
            if self.dropout_variant not in [None, 'spatial' 'gaussian']:
                raise ValueError(f'Dropout not supported, got {self.dropout_variant}')
            else:
                self.apply_dropout = True

            if self.dropout_variant == 'spatial':
                self.dropout1 = SpatialDropout2D(self.dropout_rate)
                self.dropout2 = SpatialDropout2D(self.dropout_rate)
            elif self.dropout_variant == 'gaussian':
                self.dropout1 = GaussianDropout(self.dropout_rate)
                self.dropout2 = GaussianDropout(self.dropout_rate)
            elif self.dropout_variant is None:
                self.dropout1 = Dropout(self.dropout_rate)
                self.dropout2 = Dropout(self.dropout_rate)
        else:
            self.apply_dropout = False

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
    Residual block. 
    
    Two examples:
    * Standard residual block [1]: Conv2D -> BN -> ReLU -> Conv2D -> BN -> Add _> ReLU
    * EDSR-style block: Conv2D -> ReLU -> Conv2D -> Add -> ReLU

    References
    ----------
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition: 
        https://arxiv.org/abs/1512.03385
    """
    def __init__(self, filters, strides=1, ks_cl1=(3,3), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, name=None, **conv_kwargs):
        super().__init__(filters, strides, ks_cl1, ks_cl2, activation, 
                         normalization, attention, dropout_rate, 
                         dropout_variant, name=name, **conv_kwargs)

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
    Dense block.

    References
    ----------
    [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Networks: 
        https://arxiv.org/abs/1608.06993
    """
    def __init__(self, filters, strides=1, ks_cl1=(1,1), ks_cl2=(3,3), 
                 activation='relu', normalization=None, attention=False, 
                 dropout_rate=0, dropout_variant=None, name=None, **conv_kwargs):
        super().__init__(filters, strides, ks_cl1, ks_cl2, activation, 
                         normalization, attention, dropout_rate, 
                         dropout_variant, name=name, **conv_kwargs)
        self.conv1 = Conv2D(
            4 * filters, 
            padding='same', 
            kernel_size=ks_cl1, 
            strides=strides, 
            **conv_kwargs)
        self.conv2 = Conv2D(
            filters, 
            kernel_size=ks_cl2, 
            padding='same', 
            **conv_kwargs)
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
    """ 
    Transition layer to control the complexity of the model by using 1x1 
    convolutions. Used in architectures, such as the Densenet.

    References
    ----------
    [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Networks: 
        https://arxiv.org/abs/1608.06993
    """
    def __init__(self, filters, activation='relu', normalization=None, 
                 name_suffix='', **kwargs):
        super().__init__(name='Transition' + name_suffix, **kwargs)
        if normalization is not None and normalization == 'bn':
            self.batch_norm = BatchNormalization()
        else:
            self.batch_norm = None
        self.activation = Activation(activation)
        self.conv = Conv2D(filters, kernel_size=1)

    def call(self, X):
        if self.batch_norm is not None:
            Y = self.batch_norm(X)
            Y = self.activation(Y)
            Y = self.conv(Y)
        else:
            Y = self.conv(X)
            Y = self.activation(Y)
        return Y


class LocalizedConvBlock(tf.keras.layers.Layer):
    """ 
    Localized convolutional block through a locally connected layer (1x1 kernel) 
    with biases.
    """
    def __init__(self, filters=2, activation=None, use_bias=True, 
                 name_sufix='', **kwargs):
        super().__init__(name='LocalizedConvBlock' + name_sufix, **kwargs)
        self.filters = filters
        self.transition = TransitionBlock(filters=filters)
        self.localconv = LocallyConnected2D(
            filters=filters,
            kernel_size=(1, 1),
            implementation=3,
            bias_initializer='zeros',
            use_bias=use_bias,
            activation=activation)

    def call(self, X):
        Y = self.transition(X)
        Y = self.localconv(Y)
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)


class RecurrentConvBlock(tf.keras.layers.Layer): 
    """
    Recurrent convolutional block.
    """
    def __init__(self, filters, ks_cl1=(5,5), ks_cl2=(3,3), activation='relu', 
                 normalization=None, dropout_rate=0, dropout_variant=None, 
                 name_suffix='', **conv_kwargs):
        super().__init__(name='RecurrentConvBlock' + name_suffix)
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.dropout_variant = dropout_variant
        self.convlstm1 = ConvLSTM2D(
            filters, kernel_size=ks_cl1, return_sequences=True, padding='same', 
            recurrent_dropout=0, **conv_kwargs)
        self.convlstm2 = ConvLSTM2D(
            filters, kernel_size=ks_cl2, return_sequences=True, padding='same', 
            recurrent_dropout=0, **conv_kwargs)

        if self.normalization is not None:
            if self.normalization not in ['bn', 'ln']:
                raise ValueError(f'Normalization not supported, got {self.normalization}')
        if self.normalization == 'bn':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        elif self.normalization == 'ln':
            self.norm1 = LayerNormalization()
            self.norm2 = LayerNormalization()

        self.activation = Activation(activation)
        self.skipconnection = Add()

        self.apply_dropout = False
        if self.dropout_rate > 0:
            if self.dropout_variant not in [None, 'spatial' 'gaussian']:
                raise ValueError(f'Dropout not supported, got {self.dropout_variant}')
            else:
                self.apply_dropout = True

            if self.dropout_variant == 'spatial':
                self.dropout1 = SpatialDropout3D(self.dropout_rate)
                self.dropout2 = SpatialDropout3D(self.dropout_rate)
            elif self.dropout_variant == 'gaussian':
                self.dropout1 = GaussianDropout(self.dropout_rate)
                self.dropout2 = GaussianDropout(self.dropout_rate)
            elif self.dropout_variant is None:
                self.dropout1 = Dropout(self.dropout_rate)
                self.dropout2 = Dropout(self.dropout_rate)
        else:
            self.apply_dropout = False

    def call(self, X):
        """
        Forward pass. 
        """
        Y = self.dropout1(X) if self.apply_dropout else X
        Y = self.convlstm1(Y)
        if self.normalization is not None:
            Y = self.norm1(Y)
        Y = self.activation(Y)
        if self.apply_dropout:
            Y = self.dropout2(Y)
        Y = self.convlstm2(Y)
        if self.normalization is not None:
            Y = self.norm2(Y)
        Y = self.activation(Y)
        return Y


class SubpixelConvolutionBlock(tf.keras.layers.Layer):
    """
    Subpixel convolution (pixel shuffle) block.

    References
    ----------
    [1] Real-Time Single Image and Video Super-Resolution Using an Efficient 
    Sub-Pixel Convolutional Neural Network: https://arxiv.org/abs/1609.05158
    """
    def __init__(self, scale, n_filters, name_suffix='', **kwargs):
        super().__init__(name='SubpixelConvolution' + name_suffix)
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
        """  Forward pass.
        """
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


class DeconvolutionBlock(tf.keras.layers.Layer):
    """
    Deconvolution or transposed convolution block.

    References
    ----------
    [1] FSRCNN - Accelerating the Super-Resolution Convolutional Neural Network: 
    https://arxiv.org/abs/1608.00367
    """
    def __init__(self, scale, n_filters, output_activation=None, 
                 name_suffix=''):
        super().__init__(name='Deconvolution' + name_suffix)
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


class ChannelAttention2D(tf.keras.layers.Layer):
    """
    Channel Attention for CNNs. Inputs need to be Conv2D feature maps.
    The layer implements the following:
        1. Average Pooling to create `[1,1,C]` vectors
        2. Conv2D with k=1 for fully connected features and relu ac
        3. Sigmoid activation to create attention maps
    
    Parameters
    ----------
    nf [int]: number of filters or channels
    r[int] : Reduction factor
    
    Input
    -----
    Feature maps : Conv2D feature maps of the shape `[batch,W,H,C]`.
    
    Output
    ------
    Attention activated Conv2D features of shape `[batch,W,H,C]`.
    
    Reference
    ---------
    CBAM: Convolutional Block Attention Module (Sanghyun Woo et al 2018): 
    https://arxiv.org/abs/1807.06521

    Notes
    -----
    Here is a code example for using `ChannelAttention2D` in a CNN:
    ```python
    inp = Input(shape=(1920,1080,3))
    cnn_layer = Conv2D(32,3,,activation='relu', padding='same')(inp)
    # Using the .shape[-1] to simplify network modifications. Can directly input number of channels as well
    attention_cnn = ChannelAttention2D(cnn_layer.shape[-1],cnn_layer.shape[1:-1])(cnn_layer)
    #ADD DNN layers .....
    ```
    """

    def __init__(self, nf, r=4, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.r = r
        self.conv1 = Conv2D(filters=int(nf/r), kernel_size=1, use_bias=True)
        self.conv2 = Conv2D(filters=nf, kernel_size=1, use_bias=True)

    @tf.function
    def call(self, x):
        y = tf.reduce_mean(x,axis=[1,2],keepdims=True)
        y = self.conv1(y)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = tf.nn.sigmoid(y)
        y = tf.multiply(x, y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        config.update({"Red_factor": self.r})
        return config


class EncoderBlock(tf.keras.layers.Layer):
    """
    """
    def __init__(self, n_filters, activation=None, dropout_rate=0,
                 dropout_variant=None, normalization=None, attention=False,
                 name_suffix=''):
        super().__init__(name='EncoderBlock' + name_suffix)
        self.conv = ConvBlock(
            n_filters, activation=activation, dropout_rate=dropout_rate, 
            dropout_variant=dropout_variant, normalization=normalization, 
            attention=attention)
        self.maxpool = MaxPooling2D(pool_size=(2, 2))

    def call(self, X):
        Y = self.conv(X)
        Y_downsampled = self.maxpool(Y)

        return [Y_downsampled, Y]


class PadConcat(tf.keras.layers.Layer):
    """ 
    """
    def __init__(self, debug=False, name_suffix=''):
        super().__init__(name='Concatenate' + name_suffix)
        self.debug = debug

    def call(self, X):
        (t1, t2) = X
        y1 = t1.get_shape().as_list()[1]
        x1 = t1.get_shape().as_list()[2]
        y2 = t2.get_shape().as_list()[1]
        x2 = t2.get_shape().as_list()[2]

        if self.debug:
            print(f'inp t1 ({y1},{x1}) t2 ({y2},{x2})')

        if y2 < y1:
            t2 = ZeroPadding2D(padding=((0, y1 - y2), (0, 0)))(t2)
        elif y2 > y1:
            t1 = ZeroPadding2D(padding=((0, y2 - y1), (0, 0)))(t1)

        if x2 < x1:
            t2 = ZeroPadding2D(padding=((0, 0), (0, x1 - x2)))(t2)
        elif x2 > x1:
            t1 = ZeroPadding2D(padding=((0, 0), (0, x2 - x1)))(t1)

        if self.debug:
            y1 = t1.get_shape().as_list()[1]
            x1 = t1.get_shape().as_list()[2]
            y2 = t2.get_shape().as_list()[1]
            x2 = t2.get_shape().as_list()[2]
            print(f'out t1 ({y1},{x1}) t2 ({y2},{x2})')

        return Concatenate()([t1, t2])