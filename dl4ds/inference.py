import cv2
import os
import numpy as np

from .metasr import get_coords


def predict_with_gt(model, x_test, scale, topography=None, 
                    landocean=None,interpolation='nearest', savepath=None):
    """
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR
    model_architecture = model.name
    _, hr_y, hr_x, _ = x_test.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    n_channels = 1
    if topography is not None:
        n_channels += 1
        topo_interp = cv2.resize(topography, (lr_x, lr_y), interpolation=interp)
    if landocean is not None:
        n_channels += 1
        lando_interp = cv2.resize(landocean, (lr_x, lr_y), interpolation=interp)
    x_test_lr = np.zeros((x_test.shape[0], lr_y, lr_x, n_channels))
    for i in range(x_test.shape[0]):
        x_test_lr[i, :, :, 0] = cv2.resize(x_test[i], (lr_x, lr_y), interpolation=interp)
        if topography is not None:
            x_test_lr[:, :, :, 1] = topo_interp
        if landocean is not None:
            x_test_lr[:, :, :, 2] = lando_interp
    
    print('Downsampled x_test shape: ', x_test_lr.shape)

    if model_architecture == 'edsr':
        x_test_pred = model.predict(x_test_lr)
    elif model_architecture == 'metasr':
        hr_y, hr_x = np.squeeze(x_test[0]).shape
        lr_x = int(hr_x / scale)
        lr_y = int(hr_y / scale)
        coords = np.asarray(len(x_test) * [get_coords((hr_y, hr_x), (lr_y, lr_x), scale)])
        x_test_pred = model.predict((x_test, coords))

    if savepath is not None:
        name = os.path.join(savepath, 'x_test_pred.npy')
        np.save(name, x_test_pred.astype('float32'))
    return x_test_pred