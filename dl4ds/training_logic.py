import os
import livelossplot
import numpy as np
import tensorflow as tf
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_absolute_error
from matplotlib.pyplot import show
from .utils import Timing, list_devices
from .dataloader import data_loader


def training(
    model_function, 
    x_train, 
    x_val, 
    x_test,  
    predictors_train=None,
    predictors_val=None,
    predictors_test=None,
    topography=None, 
    landocean=None,
    scale=5, 
    interpolation='nearest', 
    patch_size=40, 
    batch_size=64, 
    epochs=60, 
    steps_per_epoch=1000, 
    validation_steps=100, 
    test_steps=1000,
    learning_rate=1e-4, 
    lr_decay_after=1e5,
    early_stopping=False, 
    patience=6, 
    min_delta=0, 
    savetoh5_name=None, 
    savetoh5_dir='./models/', 
    device='GPU', 
    plot='plt', 
    show_plot=True, 
    verbosity='max', 
    **architecture_params):
    """  

    Parameters
    ----------
    predictors_train : tuple of 4D ndarray, optional
        Predictor variables for trianing. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    predictors_val : tuple of 4D ndarray, optional
        Predictor variables for validation. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    predictors_test : tuple of 4D ndarray, optional
        Predictor variables for testing. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    steps_per_epoch : int, optional
        batch_size * steps_per_epoch samples are passed per epoch.

    TO-DO:
    * add other losses (SSIM, SSIM+MAE)
    * Chosing GPUs: strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
    
    """
    timing = Timing()
    
    ### vervosity
    if verbosity == 'max':
        verbose = 1
    elif verbosity == 'min':
        verbose = 2
    else:
        verbose = 0
    
    ### devices
    if verbose in [1 ,2]:
        print('List of devices:')
    if device == 'GPU':
        devices = list_devices('logical', gpu=True, verbose=verbose)
    elif device == 'CPU':
        devices = list_devices('logical', gpu=False, verbose=verbose)
    else:
        raise ValueError('device not recognized')
    
    ### strategy for distribution
    strategy = tf.distribute.MirroredStrategy(devices)  
    if verbose in [1 ,2]:
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    batch_size_per_replica = batch_size
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    if verbose in [1 ,2]:
        print(f'Global batch size: {global_batch_size}, per replica: {batch_size_per_replica}')

    ### checking model name from function name (should be equal to keras model name)
    model_architecture = model_function.__name__
    if model_architecture not in ['rspc', 'rint', 'rmup']:
        raise ValueError('`model_function` not recognized. Must be one of the '
                         'following: rspc, rint, rmup')

    ### data loader
    ds_train = data_loader(
        x_train, 
        scale=scale, 
        batch_size=global_batch_size,
        predictors=predictors_train,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model_architecture, 
        interpolation=interpolation)
    ds_val = data_loader(
        x_val, 
        scale=scale, 
        batch_size=global_batch_size, 
        predictors=predictors_val,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model_architecture, 
        interpolation=interpolation)
    ds_test = data_loader(
        x_test, 
        scale=scale, 
        batch_size=global_batch_size, 
        predictors=predictors_test,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model_architecture, 
        interpolation=interpolation)

    ### number of channels
    n_channels = x_train.shape[-1]
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1
    if predictors_train is not None:
        n_channels += len(predictors_train)
    
    ### callbacks: early stopping and loss plotting
    callbacks = []
    if early_stopping:
        earlystop = EarlyStopping(monitor='val_loss', mode='min', 
                                  patience=patience, min_delta=min_delta, 
                                  verbose=verbose)
        callbacks.append(earlystop)
    if plot == 'llp':
        plotlosses = livelossplot.PlotLossesKerasTF()
        callbacks.append(plotlosses) 

    ### instantiating and fitting the model
    with strategy.scope():
        if model_architecture == 'rspc':
            model = model_function(scale=scale, n_channels=n_channels, **architecture_params)
        elif model_architecture in ('rmup', 'rint'):
            model = model_function(n_channels=n_channels, **architecture_params)
        if verbose == 1:
            model.summary(line_length=150)

        if isinstance(learning_rate, tuple):
            ### Adam optimizer with a scheduler 
            learning_rate = PiecewiseConstantDecay(boundaries=[lr_decay_after], 
                                                   values=[learning_rate[0], learning_rate[1]])
        optimizer = Adam(learning_rate=learning_rate)

        ### compiling and training the model with L1 pixel loss
        model.compile(optimizer=optimizer, loss=mean_absolute_error)
        fithist = model.fit(ds_train, 
                            epochs=epochs, 
                            steps_per_epoch=steps_per_epoch,
                            validation_data=ds_val, 
                            validation_steps=validation_steps, 
                            verbose=verbose, 
                            callbacks=callbacks)
        score = model.evaluate(ds_test, steps=test_steps, verbose=verbose)
        print(f'\nScore on the test set: {score}')
        
        if plot == 'plt':
            if savetoh5_name is not None and savetoh5_dir is not None:
                learning_curve_name = os.path.join(savetoh5_dir, savetoh5_name) + '_learncurve.png'
            else:
                learning_curve_name = None
            plot_history(fithist.history, path=learning_curve_name)
            if show_plot:
                show()
        
        if savetoh5_name is not None and savetoh5_dir is not None:
            os.makedirs(savetoh5_dir, exist_ok=True)
            model.save(os.path.join(savetoh5_dir, savetoh5_name) + '.h5')

        timing.runtime()
        
        return model