import os
import numpy as np

from .utils import resize_array, spatial_to_temporal_samples


def predict_with_gt(
    model, 
    data_test, 
    scale, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    time_window=None,
    interpolation='bicubic', 
    save_path=None,
    save_fname='x_test_pred.npy',
    return_lr=False):
    """Predict with a groudtruth. The HR gridded input is downsampled, then 
    super-resolved or downscaled using the trained super-resolution model. 

    Parameters
    ----------
    model : tf.keras model
        Trained model.
    data_test : numpy.ndarray
        Batch of HR grids. 
    scale : int
        Scaling factor. 
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    predictors : tuple of 4D ndarrays 
        Predictor variables, with dimensions [nsamples, lat, lon, 1].
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    save_path : str or None, optional
        If not None, the prediction (gridded variable at HR) is saved to disk.
    save_fname : str, optional
        Filename to complete the path were the prediciton is saved. 
    """ 
    model_architecture = model.name

    if time_window is not None:
        data_test = spatial_to_temporal_samples(data_test, time_window)

    if model_architecture in ['resnet_spc', 'resnet_rc', 'resnet_dc', 'resnet_bi']:
        _, hr_y, hr_x, _ = data_test.shape
        lr_x = int(hr_x / scale)
        lr_y = int(hr_y / scale)
        
        n_channels = 1
        pos = {'pred':1, 'topo':1,'laoc':1}
        if predictors is not None:
            n_predictors = len(predictors)
            n_channels += n_predictors   
            pos['topo'] += n_predictors  
            pos['laoc'] += n_predictors 
        if topography is not None:
            n_channels += 1
            pos['laoc'] += 1
        if landocean is not None:
            n_channels += 1
        
        if model_architecture in ['resnet_spc', 'resnet_rc', 'resnet_dc']:
            if topography is not None:
                topo_interp = resize_array(topography, (lr_x, lr_y), interpolation)
            if landocean is not None:
                # integer array can only be interpolated with nearest method
                lando_interp = resize_array(landocean, (lr_x, lr_y), interpolation='nearest')
            x_test_lr = np.zeros((data_test.shape[0], lr_y, lr_x, n_channels))
        
            for i in range(data_test.shape[0]):
                x_test_lr[i, :, :, 0] = resize_array(data_test[i], (lr_x, lr_y), interpolation)
                if predictors is not None:
                    # we create a tuple of 3D ndarrays [lat, lon, 1]
                    tuple_predictors = tuple([var[i] for var in predictors])
                    # turned into a 3d ndarray, [lat, lon, variables]
                    array_predictors = np.asarray(tuple_predictors)
                    array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                    x_test_lr[i, :, :, pos['pred']:n_predictors+1] = array_predictors
            if topography is not None:                                                          
                x_test_lr[:, :, :, pos['topo']] = topo_interp
            if landocean is not None:
                x_test_lr[:, :, :, pos['laoc']] = lando_interp
        
            print('Downsampled x_test shape: ', x_test_lr.shape)

            x_test_pred = model.predict(x_test_lr)

        elif model_architecture == 'resnet_bi':
            x_test_lr = np.zeros((data_test.shape[0], hr_y, hr_x, n_channels))

            for i in range(data_test.shape[0]):
                # downsampling and upsampling via interpolation
                x_test_resized = resize_array(data_test[i], (lr_x, lr_y), interpolation)
                x_test_resized = resize_array(x_test_resized, (hr_x, hr_y), interpolation)
                x_test_lr[i, :, :, 0] = x_test_resized
                if predictors is not None:
                    # we create a tuple of 3D ndarrays [lat, lon, 1]
                    tuple_predictors = tuple([var[i] for var in predictors])
                    # turned into a 3d ndarray, [lat, lon, variables]
                    array_predictors = np.asarray(tuple_predictors)
                    array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                    array_predictors = resize_array(array_predictors, (hr_x, hr_y), interpolation)
                    x_test_lr[i, :, :, pos['pred']:n_predictors+1] = array_predictors
            if topography is not None:                                                         
                x_test_lr[:, :, :, pos['topo']] = topography
            if landocean is not None:
                x_test_lr[:, :, :, pos['laoc']] = landocean
        
            print('Downsampled x_test shape: ', x_test_lr.shape)
            x_test_pred = model.predict(x_test_lr)
    
    elif model_architecture in ['recresnet_spc', 'recresnet_rc', 'recresnet_dc']:
        n_samples, n_t, hr_y, hr_x, n_channels = data_test.shape
        lr_x = int(hr_x / scale)
        lr_y = int(hr_y / scale)
        x_test_lr = np.zeros((n_samples, n_t, lr_y, lr_x, n_channels))
        for i in range(n_samples):
            x_test_lr[i] = resize_array(data_test[i], (lr_x, lr_y), interpolation, squeezed=False)

        print('Downsampled x_test shape: ', x_test_lr.shape)
        if topography is not None or landocean is not None:
            topography = resize_array(topography, (lr_x, lr_y), interpolation, squeezed=False)
            landocean = resize_array(landocean, (lr_x, lr_y), 'nearest', squeezed=False)
            topography = np.expand_dims(topography, -1)
            landocean = np.expand_dims(landocean, -1)
            static_array = np.concatenate([topography, landocean], axis=-1)
            static_array = np.expand_dims(static_array, 0)
            static_array = np.repeat(static_array, n_samples, 0)
            x_test_pred = model.predict([x_test_lr, static_array])
        else:
            x_test_pred = model.predict(x_test_lr)

    elif model_architecture == 'recresnet_bi':
        n_samples, n_t, hr_y, hr_x, n_channels = data_test.shape
        lr_x = int(hr_x / scale)
        lr_y = int(hr_y / scale)
        x_test_lr = np.zeros((n_samples, n_t, hr_y, hr_x, n_channels))
        for i in range(n_samples):
            temp = resize_array(data_test[i], (lr_x, lr_y), interpolation, squeezed=False)
            x_test_lr[i] = resize_array(temp, (hr_x, hr_y), interpolation, squeezed=False)
        
        print('Downsampled x_test shape: ', x_test_lr.shape)
        if topography is not None or landocean is not None:
            topography = np.expand_dims(topography, -1)
            landocean = np.expand_dims(landocean, -1)
            static_array = np.concatenate([topography, landocean], axis=-1)
            static_array = np.expand_dims(static_array, 0)
            static_array = np.repeat(static_array, n_samples, 0)
            x_test_pred = model.predict([x_test_lr, static_array])
        else:
            x_test_pred = model.predict(x_test_lr)

    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, x_test_pred.astype('float32'))
    
    if return_lr:
        return x_test_pred, x_test_lr
    else:
        return x_test_pred


def predict(
    model, 
    input_array, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    interpolation='bicubic', 
    save_path=None,
    save_fname='x_test_pred.npy'):
    """Predict without a groudtruth. Super-resolve or downscale a LR gridded 
    variable using the trained super-resolution model. 

    Parameters
    ----------
    model : tf.keras model
        Trained model.
    input_array : numpy.ndarray
        Gridded data in LR.
    predictors : tuple of 4D ndarrays 
        Predictor variables, with dimensions [nsamples, lat, lon, 1].
    topography : None or 2D ndarray, optional
        Elevation data. Assumed in HR for rint and in LR (resizing happens 
        anyway) for rspc.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    predictors : tuple of 4D ndarrays 
        Predictor variables, with dimensions [nsamples, lat, lon, 1].
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    save_path : str or None, optional
        If not None, the prediction (gridded variable at HR) is saved to disk.
    save_fname : str, optional
        Filename to complete the path were the prediciton is saved. 
    """ 
    model_architecture = model.name
    
    n_channels = 1
    pos = {'pred':1, 'topo':1,'laoc':1}
    if predictors is not None:
        n_predictors = len(predictors)
        n_channels += n_predictors   
        pos['topo'] += n_predictors  
        pos['laoc'] += n_predictors 
    if topography is not None:
        n_channels += 1
        pos['laoc'] += 1
    if landocean is not None:
        n_channels += 1
    
    if model_architecture in ['resnet_spc', 'resnet_rc']:
        _, size_y, size_x, _ = input_array.shape
        # resizing both topography and land-ocean to match x_test
        if topography is not None:
            topo_interp = resize_array(topography, (size_x, size_y), interpolation)
        if landocean is not None:
            # integer array can only be interpolated with nearest method
            lando_interp = resize_array(landocean, (size_x, size_y), interpolation='nearest')
        array = np.zeros((input_array.shape[0], size_y, size_x, n_channels))
    
        for i in range(input_array.shape[0]):
            # assuming input_array has shape (lat, lon, 1)
            array[i, :, :, :] = input_array[i]
            if predictors is not None:
                # we create a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in predictors])
                # turned into a 3d ndarray, [lat, lon, variables]
                array_predictors = np.asarray(tuple_predictors)
                array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                array[i, :, :, pos['pred']:n_predictors+1] = array_predictors
        if topography is not None:
            array[:, :, :, pos['topo']] = topo_interp
        if landocean is not None:
            array[:, :, :, pos['laoc']] = lando_interp
    
        #print('Downsampled x_test shape: ', array.shape)
        array_pred = model.predict(array)

    elif model_architecture == 'resnet_bi':
        # upsampling via interpolation to match the HR topography
        hr_y, hr_x = topography.shape
        array = np.zeros((input_array.shape[0], hr_y, hr_x, n_channels))

        for i in range(input_array.shape[0]):
            input_array_i = resize_array(input_array[i], (hr_x, hr_y), interpolation)
            array[i, :, :, 0] = input_array_i
            if predictors is not None:
                # we create a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in predictors])
                # turned into a 3d ndarray, [lat, lon, variables]
                array_predictors = np.asarray(tuple_predictors)
                array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                array_predictors = resize_array(array_predictors, (hr_x, hr_y), interpolation)
                array[i, :, :, pos['pred']:n_predictors+1] = array_predictors
        if topography is not None:
            array[:, :, :, pos['topo']] = topography
        if landocean is not None:
            array[:, :, :, pos['laoc']] = landocean
    
        #print('Downsampled x_test shape: ', array.shape)
        array_pred = model.predict(array)

    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, array_pred.astype('float32'))
    return array_pred
