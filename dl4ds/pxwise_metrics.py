import cv2
import tensorflow as tf
import numpy as np
from matplotlib import interactive, pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr
import os
import seaborn as sns
sns.set(style="white")

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .metasr import get_coords



def compute_corr(y_t, y_that, mode='spearman'):
    """
    https://scipy.github.io/devdocs/generated/scipy.stats.spearmanr.html

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    """
    corrmap = np.zeros_like(y_t[0,:,:,0]) 
    lenx = y_t[0,:,:,0].shape[1]
    leny = y_t[0,:,:,0].shape[0]

    if mode == 'spearman':
        f = spearmanr
    elif mode == 'pearson':
        f = pearsonr

    for x in range(lenx):
        for y in range(leny):
            corrmap[y, x] = f(y_t[:,y,x,0], y_that[:,y,x,0])[0]

    return corrmap


def pxwise_metrics(y_test, y_test_hat, dpi=100, savepath=None):
    """ 
    MSE
    https://keras.io/api/losses/regression_losses/#mean_squared_error-function

    MSLE
    https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error-(msle)

    """
    if y_test.ndim == 5:
        y_test = np.squeeze(y_test, -1)
        y_test_hat = np.squeeze(y_test_hat, -1)

    ### Computing metrics
    drange = max(y_test.max(), y_test_hat.max()) - min(y_test.min(), y_test_hat.min())
    with tf.device("cpu:0"):
        psnr = tf.image.psnr(y_test, y_test_hat, drange)
    mean_psnr = np.mean(psnr)
    std_psnr = np.std(psnr)

    with tf.device("cpu:0"):
        ssim = tf.image.ssim(tf.convert_to_tensor(y_test, dtype=tf.float32), 
                        tf.convert_to_tensor(y_test_hat, dtype=tf.float32), 
                        drange)
    mean_ssim = np.mean(ssim)
    std_ssim = np.std(ssim)

    with tf.device("cpu:0"):
        maes = tf.keras.metrics.mean_absolute_error(y_test, y_test_hat)
    maes_pairs = np.mean(maes, axis=(1,2))
    mean_mae = np.mean(maes_pairs)
    std_mae = np.std(maes_pairs)

    with tf.device("cpu:0"):
        mses = tf.keras.metrics.mean_squared_error(y_test, y_test_hat)
    pxwise_mse = np.mean(mses, axis=0)
    global_mse = np.nanmean(pxwise_mse)
    mses_pairs = np.mean(mses, axis=(1,2))
    tmean_mse = np.mean(mses, axis=0)
    mean_mse = np.mean(mses_pairs)
    std_mse = np.std(mses_pairs)

    ### Plotting the MSE map
    subpti = f'MSE map ($\mu$ = {global_mse:.6f})'
    ecv.plot_ndarray(pxwise_mse, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     interactive=False, save=os.path.join(savepath, 'mse.png'))

    ### Plotting the Spearman correlation coefficient
    spearman_corrmap = compute_corr(y_test, y_test_hat)
    mean_spcorr = np.mean(spearman_corrmap)
    subpti = f'Spearman correlation map ($\mu$ = {mean_spcorr:.6f})'
    ecv.plot_ndarray(spearman_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     interactive=False, save=os.path.join(savepath, 'corr_spear.png'))

    # ### Plotting the Pearson correlation coefficient
    pearson_corrmap = compute_corr(y_test, y_test_hat, mode='pearson')
    mean_pecorr = np.mean(pearson_corrmap)
    subpti = f'Pearson correlation map ($\mu$ = {mean_pecorr:.6f})'
    ecv.plot_ndarray(pearson_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     interactive=False, save=os.path.join(savepath, 'corr_pears.png'))

    if savepath is None:      
        ### Plotting violin plots
        f, ax = plt.subplots(1, 5, figsize=(12, 4))

        ax_ = sns.violinplot(x=psnr, ax=ax[0], orient='v')
        plt.setp(ax_.collections, alpha=0.5)
        ax_.set_title('PSNR')
        ax_.set_xlabel(f'$\mu$ = {mean_psnr:.6f} \n$\sigma$ = {std_psnr:.6f}')

        ax_ = sns.violinplot(x=ssim, ax=ax[1], orient='v')
        plt.setp(ax_.collections, alpha=0.5)
        ax_.set_title('SSIM')
        ax_.set_xlabel(f'$\mu$ = {mean_ssim:.6f} \n$\sigma$ = {std_ssim:.6f}')

        ax_ = sns.violinplot(x=maes_pairs, ax=ax[2], orient='v')
        plt.setp(ax_.collections, alpha=0.5)
        ax_.set_title('MAE')
        ax_.set_xlabel(f'$\mu$ = {mean_mae:.6f} \n$\sigma$ = {std_mae:.6f}')

        ax_ = sns.violinplot(x=mses_pairs, ax=ax[3], orient='v')
        plt.setp(ax_.collections, alpha=0.5)
        ax_.set_title('MSE')
        ax_.set_xlabel(f'$\mu$ = {mean_mse:.6f} \n$\sigma$ = {std_mse:.6f}')

        f.tight_layout()
        plt.show()

    print('Metrics on y_test and y_test_hat:\n')
    print(f'PSNR \tmu = {mean_psnr:.6f} \tsigma = {std_psnr:.8f}')
    print(f'SSIM \tmu = {mean_ssim:.6f} \tsigma = {std_ssim:.8f}')
    print(f'MAE \tmu = {mean_mae:.6f} \tsigma = {std_mae:.8f}')
    print(f'MSE \tmu = {mean_mse:.6f} \tsigma = {std_mse:.8f}')
    print(f'Spearman correlation \tmu = {mean_spcorr}')
    print(f'Pearson correlation \tmu = {mean_spcorr}')

    return pxwise_mse, spearman_corrmap, pearson_corrmap


def plot_sample(model, lr_image, topography=None, landocean=None, 
                predictors=None, dpi=150, scale=None, savepath=None, plot=True):
    """
    """
    def check_image_dims_for_inference(image):
        """ Output is a 4d array, where first and last are unitary """
        image = np.squeeze(image)
        image = np.expand_dims(np.asarray(image, "float32"), axis=-1)
        image = np.expand_dims(image, 0)
        return image
    
    model_architecture = model.name
    if model_architecture in ('rspc', 'rmup'):
        input_image = check_image_dims_for_inference(lr_image)
        
        # expecting a 3d ndarray, [lat, lon, variables]
        if predictors is not None:
            predictors = np.expand_dims(predictors, 0)
            input_image = np.concatenate([input_image, predictors], axis=3)
        if topography is not None: 
            topography = cv2.resize(topography, (input_image.shape[2], 
                                    input_image.shape[1]), 
                                    interpolation=cv2.INTER_CUBIC)
            topography = check_image_dims_for_inference(topography)
            input_image = np.concatenate([input_image, topography], axis=3)
        if landocean is not None: 
            landocean = cv2.resize(landocean, (input_image.shape[2], 
                                  input_image.shape[1]), 
                                  interpolation=cv2.INTER_NEAREST)
            landocean = check_image_dims_for_inference(landocean)
            input_image = np.concatenate([input_image, landocean], axis=3)
        
        if model_architecture == 'rspc':
            pred_image = model.predict(input_image)
        elif model_architecture == 'rmup':
            if scale is None:
                raise ValueError('`scale` must be set for `metasr` model')
            lr_y, lr_x = np.squeeze(lr_image).shape
            hr_y, hr_x = int(lr_y * scale), int(lr_x * scale)
            coords = get_coords((hr_y, hr_x), (lr_y, lr_x), scale)
            coords = np.asarray([coords])
            pred_image = model.predict((input_image, coords))
    
    elif model_architecture == 'rint':
        lr_y, lr_x = np.squeeze(lr_image).shape    
        hr_x = int(lr_x * scale)
        hr_y = int(lr_y * scale) 
        input_image = cv2.resize(np.squeeze(lr_image), (hr_x, hr_y), interpolation=cv2.INTER_CUBIC)
        input_image = check_image_dims_for_inference(input_image)
        if topography is not None: 
            topography = check_image_dims_for_inference(topography)
            input_image = np.concatenate([input_image, topography], axis=3)
        if landocean is not None:
            landocean = check_image_dims_for_inference(landocean)
            input_image = np.concatenate([input_image, landocean], axis=3)
        pred_image = model.predict(input_image)

    if plot:
        if savepath is not None:
            savepath = os.path.join(savepath, 'sample_nogt.png')
        else:
            savepath = None

        ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image)), interactive=False, 
                        subplot_titles=('LR image', 'SR image'), dpi=dpi,
                        save=savepath)
    return pred_image


def plot_sample_with_gt(model, hr_image, scale, topography=None, landocean=None,
                        predictors=None, dpi=150, interpolation='nearest', 
                        savepath=None):
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR 

    hr_image = np.squeeze(hr_image)
    hr_y, hr_x = hr_image.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    lr_image = cv2.resize(hr_image, (lr_x, lr_y), interpolation=interp)
    pred_image = plot_sample(model, lr_image, topography=topography, 
                            predictors=predictors, landocean=landocean, 
                            scale=scale, plot=False)
    residuals = hr_image - np.squeeze(pred_image)
    half_range = (hr_image.max() - hr_image.min()) / 2

    if savepath is not None:
        savepath = os.path.join(savepath, 'sample_gt.png')
    else:
        savepath = None
    ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image), hr_image, residuals), 
                     interactive=False, dpi=dpi, show_axis=False, 
                     subplot_titles=('LR image', 'SR image (Yhat)', 
                                     'Ground truth (GT)', 'Residuals (GT - Yhat)'), 
                     save=savepath, horizontal_padding=0.1)
    return pred_image



