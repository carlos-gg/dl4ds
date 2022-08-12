import tensorflow as tf
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import os
import seaborn as sns
import ecubevis as ecv

from .utils import checkarray_ndim, Timing


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
        rmse_map *= np.nan
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
        corrmap *= np.nan
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


def compute_metrics(
    y_test, 
    y_test_hat, 
    dpi=150, 
    plot_size_px=1000,
    n_jobs=-1, 
    scaler=None, 
    mask=None,
    save_path=None):
    """ Compute temporal and spatial-wise metrics, e.g., RMSE and CORRELATION, 
    based on the groundtruth and prediction ndarrays.

    Parameters
    ----------
    y_test : np.ndarray
        Groundtruth.
    y_test_hat : np.ndarray
        Prediction.
    dpi : int, optional
        DPI of the plots.  
    n_jobs : int, optional
        Number of cores for the computation of metrics (parallelizing over
        grid points). Passed to joblib.Parallel. If -1 all CPUs are used. If 1 
        or None is given, no parallel computing code is used at all, which is 
        useful for debugging.
    scaler : scaler object
        Scaler object from preprocessing module. 
    mask : np.ndarray or None
        Binary mask with valid (ones) and non-valid (zeroes) grid points.
    save_path : str or None, optional
        Path to save results to disk. 
        
    """
    timing = Timing()

    if y_test.ndim == 5:
        y_test = np.squeeze(y_test, -1)
        y_test_hat = np.squeeze(y_test_hat, -1)

    y_test = checkarray_ndim(y_test, 4, -1)
    y_test_hat = checkarray_ndim(y_test_hat, 4, -1)

    # backward transformation with the provided scaler
    if scaler is not None:
        if hasattr(scaler, 'inverse_transform'):
            y_test = scaler.inverse_transform(y_test)
            y_test_hat = scaler.inverse_transform(y_test_hat)        

    # applying valid grid points mask
    if mask is not None:
        if isinstance(mask, xr.DataArray):
            mask = mask.values.copy()
        elif isinstance(mask, np.ndarray):
            mask = mask.copy()

        if mask.ndim == 2:
            mask = np.expand_dims(mask, -1)
        y_test = y_test.copy()
        y_test_hat = y_test_hat.copy()
        for i in range(y_test.shape[0]):
            y_test[i] *= mask
        for i in range(y_test_hat.shape[0]):
            y_test_hat[i] *= mask
        mask_nan = mask.astype('float').copy()
        mask_nan[mask == 0] = np.nan
        mask = np.squeeze(mask)

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
        np.save(os.path.join(save_path, 'metrics_mse_pergridpair.npy'), spatial_rmse)
    mean_spatial_rmse = np.mean(spatial_rmse)
    std_spatial_rmse = np.std(spatial_rmse)
    mean_temp_rmse = np.nanmean(temp_rmse_map)
    std_temp_rmse = np.nanstd(temp_rmse_map)
    if mask is not None:
        temp_rmse_map[np.where(mask == 0)] = 0
    subpti = f'RMSE map ($\mu$ = {mean_temp_rmse:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'metrics_pergridpoint_rmse_map.png')
        np.save(os.path.join(save_path, 'metrics_pergridpoint_rmse_map.npy'), temp_rmse_map)
    else:
        savepath = None
    ecv.plot_ndarray(temp_rmse_map, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     plot_size_px=plot_size_px, interactive=False, save=savepath)

    ### Normalized per grid point RMSE 
    norm_temp_rmse_map = temp_rmse_map / (np.mean(y_test) * 100)
    norm_mean_temp_rmse = np.nanmean(norm_temp_rmse_map)
    norm_std_temp_rmse = np.nanstd(norm_temp_rmse_map)
    if mask is not None:
        norm_temp_rmse_map[np.where(mask == 0)] = 0
    subpti = f'nRMSE map ($\mu$ = {norm_mean_temp_rmse:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'metrics_pergridpoint_nrmse_map.png')
        np.save(os.path.join(save_path, 'metrics_pergridpoint_nrmse_map.npy'), norm_temp_rmse_map)
    else:
        savepath = None
    ecv.plot_ndarray(norm_temp_rmse_map, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     plot_size_px=plot_size_px, interactive=False, save=savepath)

    # Normalized mean bias
    nmeanbias = np.mean(y_test_hat - y_test, axis=0)
    nmeanbias /= np.mean(y_test) * 100
    if mask is not None:
        nmeanbias *= mask_nan
    mean_nmeanbias = np.nanmean(nmeanbias)
    nmeanbias[np.where(mask == 0)] = 0
    subpti = f'NMBias map ($\mu$ = {mean_nmeanbias:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'metrics_nmeanbias_map.png')
        np.save(os.path.join(save_path, 'metrics_nmeanbias_map.npy'), nmeanbias)
    else:
        savepath = None
    ecv.plot_ndarray(nmeanbias, dpi=dpi, subplot_titles=(subpti), cmap='viridis', 
                     plot_size_px=plot_size_px, interactive=False, save=savepath)

    ### Spearman correlation coefficient
    spatial_spearman_corr = compute_correlation(y_test, y_test_hat, n_jobs=n_jobs, over='space')
    mean_spatial_spearman_corr = np.mean(spatial_spearman_corr)
    std_spatial_spearman_corr = np.std(spatial_spearman_corr)
    if save_path is not None:
        np.save(os.path.join(save_path, 'metrics_spearcorr_pergridpair.npy'), spatial_spearman_corr)

    ### Pearson correlation coefficient
    spatial_pearson_corr = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs, over='space')
    mean_spatial_pearson_corr = np.mean(spatial_pearson_corr)
    std_spatial_pearson_corr = np.std(spatial_pearson_corr)
    if save_path is not None:
        np.save(os.path.join(save_path, 'metrics_pearcorr_pergridpair.npy'), spatial_pearson_corr)
    temp_pearson_corrmap = compute_correlation(y_test, y_test_hat, mode='pearson', n_jobs=n_jobs)
    mean_temp_pearson_corr = np.nanmean(temp_pearson_corrmap)
    std_temp_pearson_corr = np.nanstd(temp_pearson_corrmap)
    temp_pearson_corrmap[np.where(mask == 0)] = 0
    subpti = f'Pearson correlation map ($\mu$ = {mean_temp_pearson_corr:.6f})'
    if save_path is not None:
        savepath = os.path.join(save_path, 'metrics_pergridpoint_corrpears_map.png')
        np.save(os.path.join(save_path, 'metrics_pergridpoint_corrpears_map.npy'), temp_pearson_corrmap)
    else:
        savepath = None
    ecv.plot_ndarray(temp_pearson_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', 
                     plot_size_px=plot_size_px, interactive=False, save=savepath)
    
    ### Plotting violin plots: http://seaborn.pydata.org/tutorial/aesthetics.html
    sns.set_style("whitegrid") #{"axes.facecolor": ".9"}
    sns.despine(left=True)
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
        plt.savefig(os.path.join(save_path, 'metrics_violin_plots.png'))
        plt.close()
    else:
        plt.show()
    
    sns.set_style("white")

    if save_path is not None: 
        f = open(os.path.join(save_path, 'metrics_summary.txt'), "a")
    else:
        f = None

    print('Metrics on y_test and y_test_hat:\n', file=f)
    print(f'PSNR \tmu = {mean_psnr} \tsigma = {std_psnr}', file=f)
    print(f'SSIM \tmu = {mean_ssim} \tsigma = {std_ssim}', file=f)
    print(f'MAE \tmu = {mean_mae} \tsigma = {std_mae}', file=f)
    print(f'Per-grid-point RMSE \tmu = {mean_temp_rmse} \tsigma = {std_temp_rmse}', file=f)
    print(f'Per-grid-point nRMSE \tmu = {norm_mean_temp_rmse} \tsigma = {norm_std_temp_rmse}', file=f)
    print(f'Per-grid-point Spearman correlation \tmu = {mean_spatial_spearman_corr} \tsigma = {std_spatial_spearman_corr}', file=f)
    print(f'Per-grid-point Pearson correlation \tmu = {mean_temp_pearson_corr} \tsigma = {std_temp_pearson_corr}', file=f)
    print(file=f)
    print(f'Spatial MSE \tmu = {mean_spatial_rmse} \tsigma = {std_spatial_rmse}', file=f)
    print(f'Spatial Spearman correlation \tmu = {mean_spatial_spearman_corr} \tsigma = {std_spatial_spearman_corr}', file=f)
    print(f'Spatial Pearson correlation \tmu = {mean_spatial_pearson_corr} \tsigma = {std_spatial_pearson_corr}', file=f)

    if save_path is not None:
        f.close()

    timing.runtime()
    return temp_rmse_map, temp_pearson_corrmap, nmeanbias

