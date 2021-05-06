"""


"""

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
import horovod.tensorflow.keras as hvd

from .utils import Timing, list_devices
from .dataloader import DataGenerator
from .resnet_int import resnet_int
from .resnet_rec import resnet_rec
from .resnet_spc import resnet_spc


def training(
    model, 
    x_train, 
    x_val, 
    x_test,  
    predictors_train=None,
    predictors_val=None,
    predictors_test=None,
    topography=None, 
    landocean=None,
    scale=5, 
    interpolation='bicubic', 
    patch_size=50, 
    batch_size=64, 
    epochs=60, 
    steps_per_epoch=None, 
    validation_steps=None, 
    test_steps=None,
    learning_rate=1e-4, 
    lr_decay_after=1e5,
    early_stopping=False, 
    patience=6, 
    min_delta=0, 
    save=False,
    save_path=None, 
    savecheckpoint_path='./checkpoints/',
    device='GPU', 
    gpu_memory_growth=True,
    plot='plt', 
    show_plot=True, 
    save_plot=False,
    verbose=1, 
    **architecture_params):
    """  

    Parameters
    ----------
    batch_size : int, optional
        Batch size per replica.
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
    verbose : bool, optional
        Verbosity mode. False or 0 = silent. True or 1, max amount of 
        information is printed out. When equal 2, then less info is shown.
    
    """
    timing = Timing()
       
    # Initialize Horovod
    hvd.init()

    ### devices
    if verbose in [1 ,2]:
        print('List of devices:')
    if device == 'GPU':
        devices = list_devices('physical', gpu=True, verbose=verbose)
        tf.config.experimental.set_visible_devices(devices[hvd.local_rank()], 'GPU')
        if gpu_memory_growth:
            for gpu in devices:
                tf.config.experimental.set_memory_growth(gpu, True)
    elif device == 'CPU':
        devices = list_devices('physical', gpu=False, verbose=verbose)
    else:
        raise ValueError('device not recognized')
    
    n_devices = len(devices)
    if verbose in [1 ,2]:
        print ('Number of devices: {}'.format(n_devices))
    batch_size_per_replica = batch_size
    global_batch_size = batch_size_per_replica * n_devices
    if verbose in [1 ,2]:
        print(f'Global batch size: {global_batch_size}, per replica: {batch_size_per_replica}')

    if model not in ['resnet_spc', 'resnet_int', 'resnet_rec']:
        raise ValueError('`model` not recognized. Must be one of the following: resnet_spc, resnet_int, resnet_rec')

    if patch_size % scale != 0:
        raise ValueError('`patch_size` must be divisible by `scale` (remainder must be zero)')
    
    ### data loader
    datagen_params = dict(
        scale=scale, 
        batch_size=global_batch_size,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model, 
        interpolation=interpolation)
    
    ds_train = DataGenerator(x_train, predictors=predictors_train, **datagen_params)
    ds_val = DataGenerator(x_val, predictors=predictors_val, **datagen_params)
    ds_test = DataGenerator(x_test, predictors=predictors_test, **datagen_params)

    ### number of channels
    n_channels = x_train.shape[-1]
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1
    if predictors_train is not None:
        n_channels += len(predictors_train)
    
    ### instantiating and fitting the model
    if model == 'resnet_spc':
        model = resnet_spc(scale=scale, n_channels=n_channels, **architecture_params)
    elif model == 'resnet_rec':
        model = resnet_rec(scale=scale, n_channels=n_channels, **architecture_params)
    elif model == 'resnet_int':
        model = resnet_int(n_channels=n_channels, **architecture_params)
    
    if verbose == 1:
        if (device == 'GPU' and hvd.rank() == 0) or device == 'CPU':
            model.summary(line_length=150)

    if isinstance(learning_rate, tuple):
        ### Adam optimizer with a scheduler 
        learning_rate = PiecewiseConstantDecay(boundaries=[lr_decay_after], 
                                                values=[learning_rate[0], learning_rate[1]])
    optimizer = Adam(learning_rate=learning_rate)

    ### Callbacks
    # early stopping
    callbacks = []
    if early_stopping:
        earlystop = EarlyStopping(monitor='val_loss', mode='min', 
                                  patience=patience, min_delta=min_delta, 
                                  verbose=verbose)
        callbacks.append(earlystop)
    # loss plotting
    if plot == 'llp':
        plotlosses = livelossplot.PlotLossesKerasTF()
        callbacks.append(plotlosses) 

    # Horovod: add Horovod DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    
    if verbose in [1 ,2]:
        verbose=1 if hvd.rank() == 0 else 0

    # Model checkopoints are saved at the end of every epoch, if it's the best seen so far.
    if savecheckpoint_path is not None:
        os.makedirs(savecheckpoint_path, exist_ok=True)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(savecheckpoint_path, './checkpoint_epoch-{epoch:02d}_val-loss-{val_loss:.6f}.h5'),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if (device=='GPU' and hvd.rank() == 0) or device=='CPU':
            callbacks.append(model_checkpoint_callback)

    ### compiling and training the model with L1 pixel loss
    if steps_per_epoch is not None:
        steps_per_epoch = steps_per_epoch // hvd.size()
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
    
    timing.runtime()
    
    if plot == 'plt':
        if save_plot:
            learning_curve_fname = 'learning_curve.png'
        else:
            learning_curve_fname = None
        plot_history(fithist.history, path=learning_curve_fname)
        if show_plot:
            show()

    if save:
        if save_path is None:
            save_path = './saved_model/'
    
        if (device=='GPU' and hvd.rank() == 0) or device=='CPU':
            os.makedirs(save_path, exist_ok=True)
            model.save(save_path, save_format='tf')
    
    return model

