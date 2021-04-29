import cv2
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
from matplotlib import interactive, pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import os
import seaborn as sns

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .utils import resize_array


def compute_rmse(y, y_hat, over='time', squared=False, n_jobs=40):
    """

    Parameters
    ----------
    squared : bool
        If True returns MSE value, if False returns RMSE value.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    def rmse_per_px(tupyx):
        y_coord, x_coord = tupyx
        return y_coord, x_coord, mean_squared_error(y[:,y_coord,x_coord,0], y_hat[:,y_coord,x_coord,0])

    def rmse_gridpair(index):
        return mean_squared_error(y[index].flatten(), y_hat[index].flatten(), squared=squared)

    #---------------------------------------------------------------------------
    if over == 'time':
        rmse_map = np.zeros_like(y[0,:,:,0]) 
        yy, xx = np.where(y[0,:,:,0])
        coords = zip(yy, xx)
        out = Parallel(n_jobs=n_jobs, verbose=False)(delayed(rmse_per_px)(i) for i in coords) 

        for i in range(len(out)):
            y_coord, x_coord, val = out[i]
            rmse_map[y_coord, x_coord] = val
        return rmse_map

    elif over == 'space':
        n_timesteps = np.arange(y.shape[0])
        out = Parallel(n_jobs=n_jobs, verbose=False)(delayed(rmse_gridpair)(i) for i in n_timesteps)
        return out
    

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

    ### RMSE 
    temp_rmse_map = compute_rmse(y_test, y_test_hat, n_jobs=n_jobs, over='time')
    spatial_rmse = compute_rmse(y_test, y_test_hat, n_jobs=n_jobs, over='space')
    if save_path is not None:
        np.save(os.path.join(save_path, 'mse_pergridpair.npy'), spatial_rmse)
    mean_spatial_rmse = np.mean(spatial_rmse)
    std_spatial_rmse = np.std(spatial_rmse)
    mean_temp_rmse = np.mean(temp_rmse_map)
    std_temp_rmse = np.std(temp_rmse_map)

    subpti = f'MSE map ($\mu$ = {mean_temp_rmse:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'mse.png')
    else:
        savepath = None
    ecv.plot_ndarray(temp_rmse_map, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     plot_size_px=800, interactive=False, save=savepath)

    ### Spearman correlation coefficient
    spatial_spearman_corr = compute_correlation(y_test, y_test_hat, n_jobs=n_jobs, over='space')
    mean_spatial_spearman_corr = np.mean(spatial_spearman_corr)
    std_spatial_spearman_corr = np.std(spatial_spearman_corr)
    if save_path is not None:
        np.save(os.path.join(save_path, 'spearcorr_pergridpair.npy'), spatial_spearman_corr)
    temp_spearman_corrmap = compute_correlation(y_test, y_test_hat, n_jobs=n_jobs)
    mean_temp_spcorr = np.mean(temp_spearman_corrmap)
    subpti = f'Spearman correlation map ($\mu$ = {mean_temp_spcorr:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'corr_spear.png')
    else:
        savepath = None
    #ecv.plot_ndarray(temp_spearman_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
    #                 plot_size_px=800, interactive=False, save=savepath)

    ### Pearson correlation coefficient
    spatial_pearson_corr = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs, over='space')
    mean_spatial_pearson_corr = np.mean(spatial_pearson_corr)
    std_spatial_pearson_corr = np.std(spatial_pearson_corr)
    if save_path is not None:
        np.save(os.path.join(save_path, 'pearcorr_pergridpair.npy'), spatial_pearson_corr)
    pearson_corrmap = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs)
    mean_pecorr = np.mean(pearson_corrmap)
    subpti = f'Pearson correlation map ($\mu$ = {mean_pecorr:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'corr_pears.png')
    else:
        savepath = None
    ecv.plot_ndarray(pearson_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     plot_size_px=800, interactive=False, save=savepath)
     
    ### Plotting violin plots
    # http://seaborn.pydata.org/tutorial/aesthetics.html
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_context("notebook")
    f, ax = plt.subplots(1, 6, figsize=(15, 5), dpi=dpi)
    for axis in f.axes:
        axis.tick_params(labelrotation=40)

    ax_ = sns.violinplot(x=np.array(psnr), ax=ax[0], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('PSNR')
    ax_.set_xlabel(f'$\mu$ = {mean_psnr:.4f} \n$\sigma$ = {std_psnr:.4f}')

    ax_ = sns.violinplot(x=np.array(ssim), ax=ax[1], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('SSIM')
    ax_.set_xlabel(f'$\mu$ = {mean_ssim:.4f} \n$\sigma$ = {std_ssim:.4f}')

    ax_ = sns.violinplot(x=maes_pairs, ax=ax[2], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('MAE')
    ax_.set_xlabel(f'$\mu$ = {mean_mae:.4f} \n$\sigma$ = {std_mae:.4f}')

    ax_ = sns.violinplot(x=spatial_rmse, ax=ax[3], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('RMSE')
    ax_.set_xlabel(f'$\mu$ = {mean_spatial_rmse:.4f} \n$\sigma$ = {std_spatial_rmse:.4f}')

    ax_ = sns.violinplot(x=spatial_pearson_corr, ax=ax[4], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('Pearson correlation')
    ax_.set_xlabel(f'$\mu$ = {mean_spatial_pearson_corr:.4f} \n$\sigma$ = {std_spatial_pearson_corr:.4f}')

    ax_ = sns.violinplot(x=spatial_spearman_corr, ax=ax[5], orient='h', color="skyblue", saturation=1, linewidth=0.8)
    ax_.set_title('Spearman correlation')
    ax_.set_xlabel(f'$\mu$ = {mean_spatial_spearman_corr:.4f} \n$\sigma$ = {std_spatial_spearman_corr:.4f}')

    f.tight_layout()
    if save_path is not None: 
        plt.savefig(os.path.join(save_path, 'violin_plots.png'))
    else:
        plt.show()
    
    sns.set_style("white")
    
    print('Metrics on y_test and y_test_hat:\n')
    print(f'PSNR \tmu = {mean_psnr} \tsigma = {std_psnr}')
    print(f'SSIM \tmu = {mean_ssim} \tsigma = {std_ssim}')
    print(f'MAE \tmu = {mean_mae} \tsigma = {std_mae}')
    print(f'Temporal RMSE \tmu = {mean_temp_rmse} \tsigma = {std_temp_rmse}')
    print(f'Temporal Spearman correlation \tmu = {mean_spatial_spearman_corr}')
    print(f'Temporal Pearson correlation \tmu = {mean_spatial_pearson_corr}')
    print()
    print(f'Spatial MSE \tmu = {mean_spatial_rmse} \tsigma = {std_spatial_rmse}')
    print(f'Spatial Spearman correlation \tmu = {mean_spatial_spearman_corr} \tsigma = {std_spatial_spearman_corr}')
    print(f'Spatial Pearson correlation \tmu = {mean_spatial_pearson_corr} \tsigma = {std_spatial_pearson_corr}')

    return spatial_rmse, spatial_pearson_corr


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



