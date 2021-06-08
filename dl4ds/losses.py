import tensorflow as tf
import tensorflow.keras.backend as tfk
from tensorflow.keras.losses import mean_absolute_error


def dssim(y_true, y_pred):
    """
    Structural Dissimilarity (DSSIM). DSSIM is derived from the structural 
    similarity index measure (Wang, Z. et al. 2004).

    Notes
    -----
    https://www.tensorflow.org/api_docs/python/tf/image/ssim
    tf.image.ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    
    https://github.com/keras-team/keras-contrib/issues/464
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
 
    """
    maxv = tfk.maximum(tfk.max(y_true), tfk.max(y_pred))
    minv = tfk.minimum(tfk.min(y_true), tfk.min(y_pred))
    drange = maxv - minv
    if len(y_true.get_shape()) == 5:
        ssim = tf.image.ssim(y_true[:,:,:,:,0], y_pred[:,:,:,:,0], drange) 
    elif len(y_true.get_shape()) == 4:
        ssim = tf.image.ssim(y_true, y_pred, drange)
    dssim = tf.reduce_mean(1 - ssim / 2.0)
    return dssim


def dssim_mae(y_true, y_pred):
    """
    DSSIM + MAE (L1)
    """
    mae_loss = mean_absolute_error(y_true, y_pred)  
    dssim_loss = dssim(y_true, y_pred)
    return  0.8 * dssim_loss + 0.2 * mae_loss
    