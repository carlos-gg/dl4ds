import os
import livelossplot
import numpy as np
import tensorflow as tf
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pyplot import show
from .utils import Timing
from .data_load import data_loader, get_coords


def training(model_architecture, x_train, x_val, x_test, model='edsr', scale=20, 
             interpolation='nearest', patch_size=40, batch_size=64, epochs=60, 
             steps_per_epoch=1000, validation_steps=100, test_steps=1000,
             learning_rate=1e-4, lr_decay_after=1e5,
             early_stopping=False, patience=6, min_delta=0, 
             savetoh5_name=None, savetoh5_dir='./models/', 
             plot='plt', verbosity='max', **architecture_params):
    """    
    TO-DO:
    * add other losses (SSIM, SSIM+MAE)
    * Chosing GPUs: strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
    * Choosing CPUs
    
    """
    timing = Timing()
    which_model = model
    
    if verbosity == 'max':
        verbose = 1
    elif verbosity == 'min':
        verbose = 2
    else:
        verbose = 0
        
    strategy = tf.distribute.MirroredStrategy()  
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    batch_size_per_replica = batch_size
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    
    ds_train = data_loader(x_train, scale=scale, batch_size=global_batch_size, 
                           patch_size=patch_size, model=which_model, interpolation=interpolation)
    ds_val = data_loader(x_val, scale=scale, batch_size=global_batch_size, 
                         patch_size=patch_size, model=which_model, interpolation=interpolation)
    ds_test = data_loader(x_test, scale=scale, batch_size=global_batch_size, 
                          patch_size=patch_size, model=which_model, interpolation=interpolation)

    n_channels = x_train.shape[-1]
    
    callbacks = []
    if early_stopping:
        earlystop = EarlyStopping(monitor='val_loss', mode='min', 
                                  patience=patience, min_delta=min_delta, verbose=verbose)
        callbacks.append(earlystop)
    if plot == 'llp':
        plotlosses = livelossplot.PlotLossesKerasTF()
        callbacks.append(plotlosses) 

    with strategy.scope():
        if which_model == 'edsr':
            model = model_architecture(scale=scale, n_channels=n_channels, **architecture_params)
        elif which_model == 'metasr':
            model = model_architecture(n_channels=n_channels, **architecture_params)
        if verbose == 1:
            model.summary(line_length=150)

        if isinstance(learning_rate, float):
            optimizer = Adam(learning_rate=learning_rate)
        elif isinstance(learning_rate, tuple):
            # Adam optimizer with a scheduler 
            optimizer = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[lr_decay_after], 
                                                                  values=[learning_rate[0], learning_rate[1]]))

        # Compile and train model with L1 pixel loss
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        fithist = model.fit(ds_train, epochs=epochs, steps_per_epoch=steps_per_epoch,validation_data=ds_val, 
                            validation_steps=validation_steps, verbose=verbose, callbacks=callbacks)
        score = model.evaluate(ds_test, steps=test_steps, verbose=verbose)
        print(f'\nScore on the test set: {score}')
        
        if plot == 'plt':
            plot_history(fithist.history)
            show()
        
        if savetoh5_name is not None and savetoh5_dir is not None:
            # Saving model weights
            os.makedirs(savetoh5_dir, exist_ok=True)
            model.save(os.path.join(savetoh5_dir, savetoh5_name))

            if which_model == 'edsr':
                x_test_pred = model.predict(x_test)
            elif which_model == 'metasr':
                hr_y, hr_x = np.squeeze(x_test[0]).shape
                lr_x = int(hr_x / scale)
                lr_y = int(hr_y / scale)
                coords = np.asarray(len(x_test) * [get_coords((hr_y, hr_x), (lr_y, lr_x), scale)])
                x_test_pred = model.predict((x_test, coords))
            np.save(os.path.join(savetoh5_dir, savetoh5_name) + '.npy', x_test_pred)
            
        timing.runtime()
        
        return model