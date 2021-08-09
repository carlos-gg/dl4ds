import tensorflow as tf
import tensorflow.keras.backend as tfk


def mae(y_true, y_pred):
    """Mean absolute erro, L1 pixel loss
    """
    maef = tf.keras.losses.MeanAbsoluteError()
    mae_loss = maef(y_true, y_pred)  
    return mae_loss


def mse(y_true, y_pred):
    """Mean squared error, L2 pixel loss
    """
    msef = tf.keras.losses.MeanSquaredError()
    mse_loss = msef(y_true, y_pred)  
    return mse_loss


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
    ssim = tf.image.ssim(y_true, y_pred, drange)
    dssim = tf.reduce_mean(1 - ssim / 2.0)
    return dssim


def dssim_mae(y_true, y_pred):
    """
    DSSIM + MAE (L1)
    """
    mae_loss = mae(y_true, y_pred)  
    dssim_loss = dssim(y_true, y_pred)
    return  0.8 * dssim_loss + 0.2 * mae_loss


def dssim_mae_mse(y_true, y_pred):
    """
    DSSIM + MAE (L1) + MSE (L2)
    
    See: https://www.mdpi.com/2073-4433/10/5/244/htm
    """
    mae_loss = mae(y_true, y_pred)  
    mse_loss = mse(y_true, y_pred)  
    dssim_loss = dssim(y_true, y_pred)
    return  0.6 * dssim_loss + 0.2 * mae_loss + 0.2 * mse_loss


def dssim_mse(y_true, y_pred):
    """
    DSSIM + MSE (L2)
    """
    mse_loss = mse(y_true, y_pred)  
    dssim_loss = dssim(y_true, y_pred)
    return  0.8 * dssim_loss + 0.2 * mse_loss