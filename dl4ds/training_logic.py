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
    interpolation='bicubic', 
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
    savetoh5_name='', 
    savetoh5_dir='./', 
    savecheckpoint_dir='./checkpoints/',
    device='GPU', 
    gpu_memory_growth=False,
    plot='plt', 
    show_plot=True, 
    verbosity='max', 
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

    TO-DO
    -----
    add other losses (SSIM, SSIM+MAE)
    
    """
    timing = Timing()
    
    ### vervosity
    if verbosity == 'max':
        verbose = 1
    elif verbosity == 'min':
        verbose = 2
    else:
        verbose = 0
    
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

    ### checking model name from function name (should be equal to keras model name)
    model_architecture = model_function.__name__
    if model_architecture not in ['rspc', 'rint']:
        raise ValueError('`model_function` not recognized. Must be one of the '
                         'following: rspc, rint')

    ### data loader
    datagen_params = dict(
        scale=scale, 
        batch_size=global_batch_size,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model_architecture, 
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
    if model_architecture == 'rspc':
        model = model_function(scale=scale, n_channels=n_channels, **architecture_params)
    elif model_architecture == 'rint':
        model = model_function(n_channels=n_channels, **architecture_params)
    if verbose == 1:
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
    if savecheckpoint_dir is not None:
        os.makedirs(savecheckpoint_dir, exist_ok=True)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(savecheckpoint_dir, './checkpoint_epoch-{epoch:02d}_val-loss-{val_loss:.2f}.h5'),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if device=='GPU' and hvd.rank() == 0:
            callbacks.append(model_checkpoint_callback)
        elif device=='CPU':
            callbacks.append(model_checkpoint_callback)

    ### compiling and training the model with L1 pixel loss
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
    
    if savetoh5_name == '':
        savetoh5_name = f'scale{str(scale)}'
    savetoh5_name = model_architecture + '_' + savetoh5_name
    savetoh5_path = os.path.join(savetoh5_dir, savetoh5_name)
    if plot == 'plt':
        if savetoh5_dir is not None:
            learning_curve_fname = savetoh5_path + '_learning_curve.png'
        else:
            learning_curve_fname = None
        plot_history(fithist.history, path=learning_curve_fname)
        if show_plot:
            show()
    
    if savetoh5_dir is not None:
        os.makedirs(savetoh5_dir, exist_ok=True)
        model.save(savetoh5_path + '.h5')
    
    timing.runtime()
    
    return model

