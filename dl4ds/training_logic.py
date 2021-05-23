"""
Training procedure for supervised models
"""

import os
import livelossplot
import tensorflow as tf
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_absolute_error
from matplotlib.pyplot import show
import horovod.tensorflow.keras as hvd

from .utils import Timing, list_devices, set_gpu_memory_growth, set_visible_gpus, checkarg_model
from .dataloader import DataGenerator
from .resnet_int import resnet_int
from .resnet_rec import resnet_rec
from .resnet_spc import resnet_spc, rclstm_spc


def training(
    model, 
    x_train, 
    x_val, 
    x_test,  
    y_train=None,
    y_val=None,
    y_test=None,
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
    use_multiprocessing=False,
    plot='plt', 
    show_plot=True, 
    save_plot=False,
    verbose=True, 
    **architecture_params):
    """  

    Parameters
    ----------
    model : str
        String with the name of the model architecture, either 'resnet_spc', 
        'resnet_int' or 'resnet_rec'.
    x_train : 4D ndarray
        Training dataset with dims [nsamples, lat, lon, 1].
    x_val : 4D ndarray
        Validation dataset with dims [nsamples, lat, lon, 1]. This holdout 
        dataset is used at the end of each epoch to check the losses and prevent 
        overfitting.
    x_test : 4D ndarray
        Testing dataset with dims [nsamples, lat, lon, 1]. Holdout not used
        during training. 
    predictors_train : tuple of 4D ndarray, optional
        Predictor variables for trianing. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    predictors_val : tuple of 4D ndarray, optional
        Predictor variables for validation. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    predictors_test : tuple of 4D ndarray, optional
        Predictor variables for testing. Given as tuple of 4D ndarray with dims 
        [nsamples, lat, lon, 1]. 
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    scale : int, optional
        Scaling factor. 
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    patch_size : int, optional
        Size of the square patches used to grab training samples.
    batch_size : int, optional
        Batch size per replica.
    epochs : int, optional
        Number of epochs or passes through the whole training dataset. 
    steps_per_epoch : int, optional
        ``batch_size * steps_per_epoch`` samples are passed per epoch.
    validation_steps : int, optional
        Steps using at the end of each epoch for drawing validation samples. 
    test_steps : int, optional
        Steps using after training for drawing testing samples.
    learning_rate : float or tuple of floats, optional
        Learning rate. If a tuple is given, it corresponds to the min and max
        LR used for a PiecewiseConstantDecay scheduler.
    lr_decay_after : float or None, optional
        Used for the PiecewiseConstantDecay scheduler.
    early_stopping : bool, optional
        Whether to use early stopping.
    patience : int, optional
        Patience for early stopping. 
    min_delta : float, otional 
        Min delta for early stopping.
    save : bool, optional
        Whether to save the final model. 
    save_path : None or str
        Path for saving the final model. If None, then ``'./saved_model/'`` is
        used. The SavedModel format is a directory containing a protobuf binary 
        and a TensorFlow checkpoint.
    savecheckpoint_path : None or str
        Path for saving the training checkpoints. If None, then no checkpoints
        are saved during training. 
    device : str
        Choice of 'GPU' or 'CPU' for the training of the Tensorflow models. 
    gpu_memory_growth : bool, optional
        By default, TensorFlow maps nearly all of the GPU memory of all GPUs.
        If True, we request to only grow the memory usage as is needed by the 
        process.
    plot : str, optional
        Either 'plt' for static plot of the learning curves or 'llp' for 
        interactive plotting (useful on jupyterlab as an alternative to 
        Tensorboard).
    show_plot : bool, optional
        If True the static plot is shown after training. 
    save_plot : bool, optional
        If True the static plot is saved to disk after training. 
    verbose : bool, optional
        Verbosity mode. False or 0 = silent. True or 1, max amount of 
        information is printed out. When equal 2, then less info is shown.
    **architecture_params : dict
        Dictionary with additional parameters passed to the neural network model.

    """
    timing = Timing()
       
    # initialize Horovod
    hvd.init()

    ### devices
    if verbose in [1 ,2]:
        print('List of devices:')
    if device == 'GPU':
        if gpu_memory_growth:
            set_gpu_memory_growth(verbose=False)
        # pin GPU to be used to process local rank (one GPU per process)       
        set_visible_gpus(hvd.local_rank())
        devices = list_devices('physical', gpu=True, verbose=verbose) 
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

   # identifying the first Horovod worker (for distributed training with GPUs), or CPU training
    if (device == 'GPU' and hvd.rank() == 0) or device == 'CPU':
        running_on_first_worker = True
    else:
        running_on_first_worker = False

    model = checkarg_model(model)

    if patch_size % scale != 0:
        raise ValueError('`patch_size` must be divisible by `scale` (remainder must be zero)')
    
    ### data loader
    datagen_params = dict(scale=scale, 
        batch_size=global_batch_size,
        topography=topography, 
        landocean=landocean, 
        patch_size=patch_size, 
        model=model, 
        interpolation=interpolation)
    
    ds_train = DataGenerator(x_train, y_train, predictors=predictors_train, **datagen_params)
    ds_val = DataGenerator(x_val, y_val, predictors=predictors_val, **datagen_params)
    ds_test = DataGenerator(x_test, y_test, predictors=predictors_test, **datagen_params)

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
    elif model == 'rclstm_spc':
        model = rclstm_spc(scale=scale, n_channels=n_channels, **architecture_params)

    if verbose == 1 and running_on_first_worker:
        model.summary(line_length=150)

    if isinstance(learning_rate, tuple):
        ### Adam optimizer with a scheduler 
        learning_rate = PiecewiseConstantDecay(boundaries=[lr_decay_after], 
                                               values=[learning_rate[0], learning_rate[1]])
    elif isinstance(learning_rate, float):
        # as in Goyan et al 2018 (https://arxiv.org/abs/1706.02677)
        learning_rate *= hvd.size()
    optimizer = Adam(learning_rate=learning_rate)

    ### Callbacks
    # early stopping
    callbacks = []
    if early_stopping:
        earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, 
                                  min_delta=min_delta, verbose=verbose)
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
    
    # verbosity for model.fit
    if verbose == 1 and running_on_first_worker:
        verbose = 1
    elif verbose == 2 and running_on_first_worker:
        verbose = 2
    else:
        verbose = 0

    # Model checkopoints are saved at the end of every epoch, if it's the best seen so far.
    if savecheckpoint_path is not None:
        os.makedirs(savecheckpoint_path, exist_ok=True)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(savecheckpoint_path, './checkpoint_epoch-{epoch:02d}.h5'),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if running_on_first_worker:
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
                        callbacks=callbacks,
                        use_multiprocessing=use_multiprocessing)
    score = model.evaluate(ds_test, steps=test_steps, verbose=verbose)
    print(f'\nScore on the test set: {score}')
    
    timing.runtime()
    
    if plot == 'plt':
        if save_plot:
            learning_curve_fname = 'learning_curve.png'
        else:
            learning_curve_fname = None
        
        if running_on_first_worker:
            plot_history(fithist.history, path=learning_curve_fname)
            if show_plot:
                show()

    if save:
        if save_path is None:
            save_path = './saved_model/'
    
        if running_on_first_worker:
            os.makedirs(save_path, exist_ok=True)
            model.save(save_path, save_format='tf')
    
    return model

