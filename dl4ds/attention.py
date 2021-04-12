import tensorflow as tf
from tensorflow.keras.layers import Conv2D


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

