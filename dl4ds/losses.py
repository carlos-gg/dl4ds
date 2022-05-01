import tensorflow as tf
import tensorflow.keras.backend as tfk


def mae(y_true, y_pred):
    """
    Mean absolute error, L1 pixel loss
    """
    maef = tf.keras.losses.MeanAbsoluteError()
    mae_loss = maef(y_true, y_pred)  
    return mae_loss


def mse(y_true, y_pred):
    """
    Mean squared error, L2 pixel loss
    """
    msef = tf.keras.losses.MeanSquaredError()
    mse_loss = msef(y_true, y_pred)  
    return mse_loss


def dssim(y_true, y_pred):
    """
    Structural Dissimilarity (DSSIM). DSSIM is derived from the structural 
    similarity index measure.
    
    References
    ----------
    Wang, Z. et al. 2004, Image quality assessment: from error visibility to 
    structural similarity: https://ieeexplore.ieee.org/document/1284395

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
    if tfk.min(y_true) < 0:
        y_true_pos = y_true - tfk.min(y_true)
    else:
        y_true_pos = y_true
    if tfk.min(y_pred) < 0:
        y_pred_pos = y_pred - tfk.min(y_pred)
    else:
        y_pred_pos = y_pred
    ssim = tf.image.ssim(y_true_pos, y_pred_pos, max_val=drange, filter_size=11,
        filter_sigma=1.5, k1=0.01, k2=0.03)
    dssim = tf.reduce_mean((1 - ssim) / 2.0)
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
    
    References
    ----------
    Computer Vision in Precipitation Nowcasting: Applying Image Quality 
    Assessment Metrics for Training Deep Neural Networks: 
    https://www.mdpi.com/2073-4433/10/5/244/htm
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


def msdssim(y_true, y_pred):
    """
    Multiscale Structural Dissimilarity (MSDSSIM). 
    
    References
    ----------
    Wang, Z. 2003, "Multiscale structural similarity for image quality 
    assessment": https://ieeexplore.ieee.org/document/1292216

    Notes
    -----
    https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale

    power_factors: Iterable of weights for each of the scales. The number of 
    scales used is the length of the list. Index 0 is the unscaled resolution's 
    weight and each increasing scale corresponds to the image being downsampled 
    by 2. Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333), which are the 
    values obtained in the original paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    """
    maxv = tfk.maximum(tfk.max(y_true), tfk.max(y_pred))
    minv = tfk.minimum(tfk.min(y_true), tfk.min(y_pred))
    drange = maxv - minv
    if tfk.min(y_true) < 0:
        y_true_pos = y_true - tfk.min(y_true)
    else:
        y_true_pos = y_true
    if tfk.min(y_pred) < 0:
        y_pred_pos = y_pred - tfk.min(y_pred)
    else:
        y_pred_pos = y_pred
    msssim = tf.image.ssim_multiscale(y_true_pos, y_pred_pos, max_val=drange, 
        filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03,
        power_factors=(0.0448, 0.2856, 0.3001, 0.2363))
    msssim = tf.reduce_mean((1 - msssim) / 2.0)
    return msssim


def msdssim_mae(y_true, y_pred):
    """
    MSDSSIM + MAE (L1)
    """
    mae_loss = mae(y_true, y_pred)  
    msdssim_loss = msdssim(y_true, y_pred)
    return  0.8 * msdssim_loss + 0.2 * mae_loss


def msdssim_mae_mse(y_true, y_pred):
    """
    MSDSSIM + MAE (L1) + MSE (L2)
    """
    mae_loss = mae(y_true, y_pred)  
    mse_loss = mse(y_true, y_pred)  
    msdssim_loss = msdssim(y_true, y_pred)
    return  0.6 * msdssim_loss + 0.2 * mae_loss + 0.2 * mse_loss