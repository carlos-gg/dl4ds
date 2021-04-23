import cv2
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
from matplotlib import interactive, pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import os
import seaborn as sns
sns.set(style="white")

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .utils import resize_array


def compute_correlation(y, y_hat, over='time', mode='spearman', n_jobs=40):
    """
    https://scipy.github.io/devdocs/generated/scipy.stats.spearmanr.html

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    """
    def corr_per_px(tupyx):
        if mode == 'spearman':
            f = spearmanr
        elif mode == 'pearson':
            f = pearsonr
        y_coord, x_coord = tupyx
        return y_coord, x_coord, f(y[:,y_coord,x_coord,0], y_hat[:,y_coord,x_coord,0])[0]

    def corr_per_gridpair(index):
        if mode == 'spearman':
            f = spearmanr
        elif mode == 'pearson':
            f = pearsonr
        return index, f(y[index].ravel(), y_hat[index].ravel())[0]

    #---------------------------------------------------------------------------
    if over == 'time':
        corrmap = np.zeros_like(y[0,:,:,0]) 
        yy, xx = np.where(y[0,:,:,0])
        coords = zip(yy, xx)
        out = Parallel(n_jobs=n_jobs, verbose=False)(delayed(corr_per_px)(i) for i in coords) 

        for i in range(len(out)):
            y_coord, x_coord, val = out[i]
            corrmap[y_coord, x_coord] = val

        return corrmap

    elif over == 'space':
        n_timesteps = np.arange(y.shape[0])
        out = Parallel(n_jobs=n_jobs, verbose=False)(delayed(corr_per_gridpair)(i) for i in n_timesteps)

        list_corrs = []
        list_inds = []
        for i in range(len(out)):
            index, value = out[i]
            list_inds.append(index)
            list_corrs.append(value)

        return list_corrs


def compute_metrics(y_test, y_test_hat, dpi=150, n_jobs=40, save_path=None):
    """ 
    MSE
    https://keras.io/api/losses/regression_losses/#mean_squared_error-function

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
    if save_path is not None:
        np.save(os.path.join(save_path, 'mse_pergridpair.npy'), mses_pairs)
    tmean_mse = np.mean(mses, axis=0)
    mean_mse = np.mean(mses_pairs)
    std_mse = np.std(mses_pairs)

    ### Plotting the MSE map
    subpti = f'MSE map ($\mu$ = {global_mse:.6f})'
    ecv.plot_ndarray(pxwise_mse, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     plot_size_px=800, interactive=False, 
                     save=os.path.join(save_path, 'mse.png'))

    ### Spearman correlation coefficient
    spearman_corr_perpairs = compute_correlation(y_test, y_test_hat, n_jobs=n_jobs, over='space')
    if save_path is not None:
        np.save(os.path.join(save_path, 'spearcorr_pergridpair.npy'), spearman_corr_perpairs)
    spearman_corrmap = compute_correlation(y_test, y_test_hat, n_jobs=n_jobs)
    mean_spcorr = np.mean(spearman_corrmap)
    subpti = f'Spearman correlation map ($\mu$ = {mean_spcorr:.6f})'
    ecv.plot_ndarray(spearman_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     plot_size_px=800, interactive=False, 
                     save=os.path.join(save_path, 'corr_spear.png'))

    ### Pearson correlation coefficient
    pearson_corr_perpairs = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs, over='space')
    if save_path is not None:
        np.save(os.path.join(save_path, 'pearcorr_pergridpair.npy'), pearson_corr_perpairs)
    pearson_corrmap = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs)
    mean_pecorr = np.mean(pearson_corrmap)
    subpti = f'Pearson correlation map ($\mu$ = {mean_pecorr:.6f})'
    ecv.plot_ndarray(pearson_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     plot_size_px=800, interactive=False, 
                     save=os.path.join(save_path, 'corr_pears.png'))
     
    ### Plotting violin plots
    f, ax = plt.subplots(1, 4, figsize=(12, 4))

    ax_ = sns.violinplot(x=np.array(psnr), ax=ax[0], orient='h')
    plt.setp(ax_.collections, alpha=0.5)
    ax_.set_title('PSNR')
    ax_.set_xlabel(f'$\mu$ = {mean_psnr:.6f} \n$\sigma$ = {std_psnr:.6f}')

    ax_ = sns.violinplot(x=np.array(ssim), ax=ax[1], orient='h')
    plt.setp(ax_.collections, alpha=0.5)
    ax_.set_title('SSIM')
    ax_.set_xlabel(f'$\mu$ = {mean_ssim:.6f} \n$\sigma$ = {std_ssim:.6f}')

    ax_ = sns.violinplot(x=maes_pairs, ax=ax[2], orient='h')
    plt.setp(ax_.collections, alpha=0.5)
    ax_.set_title('MAE')
    ax_.set_xlabel(f'$\mu$ = {mean_mae:.6f} \n$\sigma$ = {std_mae:.6f}')

    ax_ = sns.violinplot(x=mses_pairs, ax=ax[3], orient='h')
    plt.setp(ax_.collections, alpha=0.5)
    ax_.set_title('MSE')
    ax_.set_xlabel(f'$\mu$ = {mean_mse:.6f} \n$\sigma$ = {std_mse:.6f}')

    f.tight_layout()
    if save_path is not None: 
        plt.savefig(os.path.join(save_path, 'violin_plots.png'))
    else:
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
                predictors=None, dpi=150, scale=None, save_path=None, plot=True):
    """
    Check the model prediction for a single LR image/grid. 
    """
    def check_image_dims_for_inference(image):
        """ Output is a 4d array, where first and last are unitary """
        image = np.squeeze(image)
        image = np.expand_dims(np.asarray(image, "float32"), axis=-1)
        image = np.expand_dims(image, 0)
        return image
    
    model_architecture = model.name
    if model_architecture  in ['resnet_spc', 'resnet_rec']:
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
        
        pred_image = model.predict(input_image)
    
    elif model_architecture == 'resnet_int':
        if scale is None:
            raise ValueError('`scale` must be set for `rint` model')
        lr_y, lr_x = np.squeeze(lr_image).shape    
        hr_x = int(lr_x * scale)
        hr_y = int(lr_y * scale) 
        # upscaling the lr image via interpolation
        input_image = cv2.resize(np.squeeze(lr_image), (hr_x, hr_y), interpolation=cv2.INTER_CUBIC)
        input_image = check_image_dims_for_inference(input_image)
        if predictors is not None:
            predictors = np.expand_dims(predictors, 0)
            predictors = cv2.resize(np.squeeze(predictors), (hr_x, hr_y), interpolation=cv2.INTER_CUBIC)
            predictors = np.expand_dims(predictors, 0)
            input_image = np.concatenate([input_image, predictors], axis=3)
        if topography is not None: 
            topography = check_image_dims_for_inference(topography)
            input_image = np.concatenate([input_image, topography], axis=3)
        if landocean is not None:
            landocean = check_image_dims_for_inference(landocean)
            input_image = np.concatenate([input_image, landocean], axis=3)
        pred_image = model.predict(input_image)

    if plot:
        if save_path is not None:
            savepath = os.path.join(save_path, 'sample_nogt.png')
        else:
            savepath = None

        ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image)), interactive=False, 
                        subplot_titles=('LR image', 'SR image'), dpi=dpi, plot_size_px=800,
                        horizontal_padding=0.2, save=savepath)
    return pred_image


def plot_sample_with_gt(model, hr_image, scale, topography=None, landocean=None,
                        predictors=None, dpi=150, interpolation='bicubic', 
                        save_path=None):
    """ """
    hr_image = np.squeeze(hr_image)
    hr_y, hr_x = hr_image.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    lr_image = resize_array(hr_image, (lr_x, lr_y), interpolation)
    pred_image = plot_sample(model, lr_image, topography=topography, 
                            predictors=predictors, landocean=landocean, 
                            scale=scale, plot=False)
    residuals = hr_image - np.squeeze(pred_image)
    half_range = (hr_image.max() - hr_image.min()) / 2

    if save_path is not None:
        savepath = os.path.join(save_path, 'sample_gt.png')
    else:
        savepath = None
    ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image), hr_image, residuals), 
                     interactive=False, dpi=dpi, show_axis=False, 
                     subplot_titles=('LR image', 'SR image (Yhat)', 
                                     'Ground truth (GT)', 'Residuals (GT - Yhat)'), 
                     save=savepath, horizontal_padding=0.1, plot_size_px=800)
    return pred_image



