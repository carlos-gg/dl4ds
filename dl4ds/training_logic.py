from builtins import ValueError
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
from .data_load import data_loader, data_loader_metasr


def training(model_function, 
             x_train, x_val, x_test,  
             array_predictors=None,
             topography=None, 
             landocean=None,
             scale=20, 
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
        print(f'Gloabl batch size: {global_batch_size}, per replica: {batch_size_per_replica}')

    ### checking model
    model_architecture = model_function.__name__
    if model_architecture not in ['edsr', 'metasr']:
        raise ValueError('`model_function` must be `edsr` or `metasr`')

    ### data loaders
    if model_architecture == 'edsr':
        ds_train = data_loader(x_train, scale=scale, batch_size=global_batch_size,
                            array_predictors=array_predictors, 
                            topography=topography, landocean=landocean,
                            patch_size=patch_size, model=model_architecture, 
                            interpolation=interpolation)
        ds_val = data_loader(x_val, scale=scale, batch_size=global_batch_size, 
                            array_predictors=array_predictors, 
                            topography=topography, landocean=landocean,
                            patch_size=patch_size, model=model_architecture, 
                            interpolation=interpolation)
        ds_test = data_loader(x_test, scale=scale, batch_size=global_batch_size, 
                            array_predictors=array_predictors, 
                            topography=topography, landocean=landocean,
                            patch_size=patch_size, model=model_architecture, 
                            interpolation=interpolation)
    elif model_architecture == 'metasr':
        ds_train = data_loader_metasr(x_train, scale, global_batch_size, patch_size)
        ds_val = data_loader_metasr(x_val, scale, global_batch_size, patch_size)
        ds_test = data_loader_metasr(x_test, scale, global_batch_size, patch_size)

    ### number of channels
    n_channels = x_train.shape[-1]
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1
    if array_predictors is not None:
        n_channels += array_predictors.shape[-1]
    
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
        if model_architecture == 'edsr':
            model = model_function(scale=scale, n_channels=n_channels, **architecture_params)
        elif model_architecture == 'metasr':
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