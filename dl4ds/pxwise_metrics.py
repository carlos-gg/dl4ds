import cv2
import tensorflow as tf
import numpy as np
from matplotlib import interactive, pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
sns.set(style="white")

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv


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


def pxwise_metrics(y_test, y_test_hat, dpi=100):
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
    ecv.plot_ndarray(pxwise_mse, dpi=dpi, subplot_titles=(subpti), cmap='viridis', interactive=False)

    ### Plotting the Spearman correlation coefficient
    spearman_corrmap = compute_corr(y_test, y_test_hat)
    mean_spcorr = np.mean(spearman_corrmap)
    subpti = f'Spearman correlation map ($\mu$ = {mean_spcorr:.6f})'
    ecv.plot_ndarray(spearman_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', interactive=False)

    # ### Plotting the Pearson correlation coefficient
    pearson_corrmap = compute_corr(y_test, y_test_hat, mode='pearson')
    mean_pecorr = np.mean(pearson_corrmap)
    subpti = f'Pearson correlation map ($\mu$ = {mean_pecorr:.6f})'
    ecv.plot_ndarray(pearson_corrmap, dpi=dpi, subplot_titles=(subpti), cmap='magma', interactive=False)
        
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


def plot_sample(model, lr_image, dpi=150):
    input_image = np.expand_dims(np.asarray(lr_image, "float32"), axis=0)
    pred_image = model.predict(input_image, batch_size=1)
    ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image)), interactive=False, 
                     subplot_titles=('LR image', 'SR image'), dpi=dpi)
    return pred_image

def plot_sample_with_gt(model, hr_image, scale, dpi=150): 
    lr_image = hr_image.copy()
    hr_y, hr_x, _ = lr_image.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale) 
    lr_image = cv2.resize(lr_image, (lr_x, lr_y), interpolation=cv2.INTER_NEAREST)
    lr_image = np.expand_dims(lr_image, -1)
    lr_image = tf.convert_to_tensor(lr_image, np.float32)
    
    pred_image = model.predict(np.expand_dims(lr_image, 0), batch_size=1)
    
    residuals = np.squeeze(hr_image)- np.squeeze(pred_image)
    half_range = (hr_image.max() - hr_image.min()) / 2

    ecv.plot_ndarray((np.squeeze(lr_image), np.squeeze(pred_image), np.squeeze(hr_image), residuals), interactive=False,
                     subplot_titles=('LR image', 'SR image (Yhat)', 'Ground truth (GT)', 'Residuals (GT - Yhat)'), 
                     dpi=dpi, axis=False)

    return pred_image


# def plot_sample(x_test, y_test, y_test_hat, index=None, xvar=['Geopotential 100'], 
#                 yvar='Precipitation', dynamicrange='yhat', cmap='viridis', 
#                 vmin=None, vmax=None):
#     """ 
#     """     
#     if y_test.ndim == 5:
#         y_test = np.squeeze(y_test, -1)
#         y_test_hat = np.squeeze(y_test_hat, -1)
#         x_test = np.squeeze(x_test, -1)

#     ### Grabbing the test sample and the model prediction
#     if index is None:
#         index = np.random.randint(y_test.shape[0])
#     print(f'\nShowing test sample with index: {index} \n')
#     x = x_test[index]
#     yhat = y_test_hat[index,:,:,0].copy()
#     y = y_test[index,:,:,0].copy()

#     f, ax = plt.subplots(1, len(xvar), figsize=(4 * len(xvar), 3), dpi=100,
#                          sharey=True)
#     for i, label in enumerate(xvar):
#         ax[i].imshow(x[:,:,i], origin='lower', cmap=cmap)
#         ax[i].set_title(f'{label}, test sample', fontsize=8)
#         ax[i].set_xticks([])
#         ax[i].set_yticks([])
#         # if i == 0:
#             # ax[i].set_ylabel("$\it{lat}$", fontsize=10)
#         # ax[i].set_xlabel("$\it{lon}$", fontsize=10)
#     f.subplots_adjust(wspace=0.003)

#     if vmin is None:
#         vmin = dict()
#     if vmax is None:
#         vmax = dict()

#     f, ax = plt.subplots(1, 3, figsize=(20, 8), dpi=100, sharey=True)

#     if dynamicrange == 'yhat':
#         vminy = vminyhat = vmindif = yhat.min()
#         vmaxy = vmaxyhat = vmaxdif = yhat.max() 
#     elif dynamicrange == 'y':
#         vminy = vminyhat = vmindif = y.min()
#         vmaxy = vmaxyhat = vmaxdif = y.max()
#     elif dynamicrange is None:
#         if isinstance(vmin, dict):
#             vminy = vmin.get('y')
#             vminyhat = vmin.get('yhat')
#             vmindif = vmin.get('dif')
#         else:
#             vminy = vminyhat = vmindif = None  
#         if isinstance(vmax, dict):
#             vmaxy = vmax.get('y')
#             vmaxyhat = vmax.get('yhat')
#             vmaxdif = vmax.get('dif')
#         else:
#             vmaxy = vmaxyhat = vmaxdif = None

#     imprlr0 = ax[0].imshow(y, origin='ĺower', vmin=vminy, vmax=vmaxy, cmap=cmap)
#     # ax[0].set_title(f'{yvar}, ground truth', fontsize=10)
#     ax[0].set_title(f'Ground truth', fontsize=12)
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     # ax[0].set_ylabel("$\it{lat}$", fontsize=10)
#     # ax[0].set_xlabel("$\it{lon}$", fontsize=10)
#     plot_cbar(f, ax[0], imprlr0)

#     imprlr1 = ax[1].imshow(yhat, origin='ĺower', vmin=vminyhat, vmax=vmaxyhat, 
#                            cmap=cmap)
#     ax[1].set_title(f'Model output', fontsize=12)
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     # ax[1].set_xlabel("$\it{lon}$", fontsize=10)
#     plot_cbar(f, ax[1], imprlr1)

#     imprlr1 = ax[2].imshow(y - yhat, origin='ĺower', cmap=cmap, 
#                            vmin=vmindif, vmax=vmaxdif)
#     ax[2].set_title(f'Residuals', fontsize=12)
#     ax[2].set_xticks([])
#     ax[2].set_yticks([])
#     # ax[2].set_xlabel("$\it{lon}$", fontsize=10)
#     plot_cbar(f, ax[2], imprlr1)
