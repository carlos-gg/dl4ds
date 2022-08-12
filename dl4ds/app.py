#!/usr/bin/env python

"""
absl.FLAGS-based command line app. To be executed run something like this:

python -m dl4ds.app --flagfile=params.cfg
"""

import numpy as np
import xarray as xr
import importlib.util
from absl import app, flags  

# Usign Agg MLP backend to prevent errors related to X11 unable to connect to display "localhost:10.0"
import matplotlib
matplotlib.use('Agg')

# Attempting to import horovod
try:
    import horovod.tensorflow.keras as hvd
    has_horovod = True
    hvd.init()
    if hvd.rank() == 0:
        running_on_first_worker = True
    else:
        running_on_first_worker = False
except ImportError:
    has_horovod = False
    running_on_first_worker = True

import dl4ds as dds
from dl4ds import BACKBONE_BLOCKS, UPSAMPLING_METHODS, INTERPOLATION_METHODS, LOSS_FUNCTIONS, DROPOUT_VARIANTS


FLAGS = flags.FLAGS

### EXPERIMENT
flags.DEFINE_bool('train', True, 'Training a model')
flags.DEFINE_bool('test', True, 'Testing the trained model on holdout data')
flags.DEFINE_bool('metrics', True, 'Running vaerification metrics on the downscaled arrays')
flags.DEFINE_bool('debug', False, 'If True a debug training run (2 epochs by default with 6 steps) is executed') 

### DOWNSCALING PARAMS
flags.DEFINE_enum('trainer', 'SupervisedTrainer', ['SupervisedTrainer', 'CGANTrainer'], 'Tainer')
flags.DEFINE_enum('paired_samples', 'implicit', ['implicit', 'explicit'], 'Type of learning: implicit (PerfectProg) or explicit (MOS)')
flags.DEFINE_string('data_module', None, 'Python module where the data pre-processing is done')

### MODEL
flags.DEFINE_enum('backbone', 'resnet', BACKBONE_BLOCKS, 'Backbone section')
flags.DEFINE_enum('upsampling', 'spc', UPSAMPLING_METHODS, 'Upsampling method')
flags.DEFINE_integer('time_window', None, 'Time window for training spatio-temporal models')
flags.DEFINE_integer('n_filters', 8, 'Number of convolutional filters for the first convolutional block')
flags.DEFINE_integer('n_blocks', 6, 'Number of convolutional blocks')
flags.DEFINE_integer('n_disc_filters', 32, 'Number of convolutional filters per convolutional block in the discriminator')
flags.DEFINE_integer('n_disc_blocks', 4, 'Number of residual blocks for discriminator network')
flags.DEFINE_enum('normalization', None, ['bn', 'ln'], 'Normalization')
flags.DEFINE_float('dropout_rate', 0.2, 'Dropout rate')
flags.DEFINE_enum('dropout_variant', 'vanilla', DROPOUT_VARIANTS, 'Dropout variants')
flags.DEFINE_bool('attention', False, 'Attention block in convolutional layers')
flags.DEFINE_enum('activation', 'relu', ['elu', 'relu', 'gelu', 'crelu', 'leaky_relu', 'selu'], 'Activation used in intermediate convolutional blocks')
flags.DEFINE_enum('output_activation', None, ['elu', 'relu', 'gelu', 'crelu', 'leaky_relu', 'selu'], 'Activation used in the last convolutional block')
flags.DEFINE_bool('localcon_layer', False, 'Locally connected convolutional layer')
flags.DEFINE_enum('decoder_upsampling', 'rc', UPSAMPLING_METHODS, 'Upsampling in decoder blocks (unet backbone)')
flags.DEFINE_enum('rc_interpolation', 'bilinear', INTERPOLATION_METHODS, 'Interpolation used in resize convolution upsampling')

### TRAINING PROCEDURE
flags.DEFINE_enum('device', 'GPU', ['GPU', 'CPU'], 'Device to be used: GPU or CPU')
flags.DEFINE_bool('save', True, 'Saving to disk the trained model (last epoch), metrics, run info, etc')
flags.DEFINE_string('save_path', './dl4ds_results/', 'Path for saving results to disk')
flags.DEFINE_integer('scale', 2, 'Scaling factor, positive integer')
flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_enum('loss', 'mae', LOSS_FUNCTIONS, 'Loss function')
flags.DEFINE_enum('interpolation', 'inter_area', INTERPOLATION_METHODS, 'Interpolation method')
flags.DEFINE_integer('patch_size', None, 'Patch size in number of px/gridpoints')
flags.DEFINE_integer('batch_size', 32, 'Batch size (of samples) used during training')
flags.DEFINE_multi_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_bool('gpu_memory_growth', True, 'To use GPU memory growth (gradual memory allocation)')
flags.DEFINE_bool('use_multiprocessing', True, 'To use multiprocessing for data generation')
flags.DEFINE_float('lr_decay_after', 1e5, 'Steps to tweak the learning rate using the PiecewiseConstantDecay scheduler')
flags.DEFINE_bool('early_stopping', False, 'Early stopping')
flags.DEFINE_integer('patience', 6, 'Patience in number of epochs w/o improvement for early stopping')
flags.DEFINE_float('min_delta', 0.0, 'Minimum delta improvement for early stopping')
flags.DEFINE_bool('show_plot', False, 'Show the learning curve plot on finish')
flags.DEFINE_bool('save_bestmodel', True, 'SupervisedTrainer - Whether to save the best model (epoch with the best val_loss)')
flags.DEFINE_bool('verbose', True, 'Verbosity')
flags.DEFINE_integer('checkpoints_frequency', 2, 'CGANTrainer - Frequency for saving checkpoints and the generator')

### INFERENCE/TEST
flags.DEFINE_bool('inference_array_in_hr', False, 'Whether the inference array is in high resolution')
flags.DEFINE_string('inference_save_fname', None, 'Filename for saving the inference array')



def dl4ds(argv):
    """DL4DS absl.FLAGS-based command line app.
    """
    if running_on_first_worker:
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DL4DS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    # Run mode
    if FLAGS.debug:
        epochs = 2
        steps_per_epoch = test_steps = validation_steps = 6
    else:
        epochs = FLAGS.epochs
        steps_per_epoch = test_steps = validation_steps = None 

    if running_on_first_worker:
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Loading data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    # Training data from Python script/module
    if FLAGS.data_module is not None:
        spec = importlib.util.spec_from_file_location("module.name", FLAGS.data_module)
        DATA = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DATA)
    else:
        raise ValueError('`data_module` flag must be provided (path to the data preprocessing module)')

    # Architecture parameters
    if FLAGS.time_window is None:
        if FLAGS.upsampling == 'pin':
            architecture_params = dict(
                n_filters=FLAGS.n_filters,
                n_blocks=FLAGS.n_blocks,
                normalization=FLAGS.normalization,
                dropout_rate=FLAGS.dropout_rate,
                dropout_variant=FLAGS.dropout_variant,
                attention=FLAGS.attention,
                activation=FLAGS.activation,
                localcon_layer=FLAGS.localcon_layer,
                output_activation=FLAGS.output_activation)
            if FLAGS.backbone == 'unet':
                architecture_params['decoder_upsampling'] = FLAGS.decoder_upsampling
                architecture_params['rc_interpolation'] = FLAGS.rc_interpolation
        else:
            architecture_params = dict(
                n_filters=FLAGS.n_filters,
                n_blocks=FLAGS.n_blocks,
                normalization=FLAGS.normalization,
                dropout_rate=FLAGS.dropout_rate,
                dropout_variant=FLAGS.dropout_variant,
                attention=FLAGS.attention,
                activation=FLAGS.activation,
                localcon_layer=FLAGS.localcon_layer,
                output_activation=FLAGS.output_activation,
                rc_interpolation=FLAGS.rc_interpolation)
    else:
        if FLAGS.upsampling == 'pin':
            architecture_params = dict(
                n_filters=FLAGS.n_filters,
                n_blocks=FLAGS.n_blocks,
                activation=FLAGS.activation,
                normalization=FLAGS.normalization,
                dropout_rate=FLAGS.dropout_rate,
                dropout_variant=FLAGS.dropout_variant,
                attention=FLAGS.attention,
                output_activation=FLAGS.output_activation,
                localcon_layer=FLAGS.localcon_layer)
        else:
            architecture_params = dict(
                n_filters=FLAGS.n_filters,
                activation=FLAGS.activation,
                normalization=FLAGS.normalization,
                dropout_rate=FLAGS.dropout_rate,
                dropout_variant=FLAGS.dropout_variant,
                attention=FLAGS.attention,
                output_activation=FLAGS.output_activation,
                localcon_layer=FLAGS.localcon_layer,
                rc_interpolation=FLAGS.rc_interpolation)

    if FLAGS.train:
        if running_on_first_worker:
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DL4DS Training phase >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if FLAGS.trainer == 'SupervisedTrainer':
            trainer = dds.SupervisedTrainer(
                backbone=FLAGS.backbone, 
                upsampling=FLAGS.upsampling,
                data_train=DATA.data_train, 
                data_val=DATA.data_val, 
                data_test=DATA.data_test, 
                data_train_lr=DATA.data_train_lr if FLAGS.paired_samples == 'explicit' else None, 
                data_val_lr=DATA.data_val_lr if FLAGS.paired_samples == 'explicit' else None, 
                data_test_lr=DATA.data_test_lr if FLAGS.paired_samples == 'explicit' else None, 
                predictors_train=DATA.predictors_train, 
                predictors_val=DATA.predictors_val, 
                predictors_test=DATA.predictors_test, 
                static_vars=DATA.static_vars, 
                scale=FLAGS.scale, 
                interpolation=FLAGS.interpolation,
                patch_size=FLAGS.patch_size, 
                time_window=FLAGS.time_window, 
                batch_size=FLAGS.batch_size,
                loss=FLAGS.loss, 
                epochs=epochs, 
                steps_per_epoch=steps_per_epoch, 
                validation_steps=validation_steps, 
                test_steps=test_steps,
                device=FLAGS.device, 
                gpu_memory_growth=FLAGS.gpu_memory_growth, 
                use_multiprocessing=FLAGS.use_multiprocessing, 
                learning_rate=FLAGS.learning_rate, 
                lr_decay_after=FLAGS.lr_decay_after, 
                early_stopping=FLAGS.early_stopping, 
                patience=FLAGS.patience, 
                min_delta=FLAGS.min_delta, 
                show_plot=FLAGS.show_plot, 
                save=FLAGS.save, 
                save_path=FLAGS.save_path, 
                save_bestmodel=FLAGS.save_bestmodel, 
                trained_model=None, #FLAGS.trained_model, 
                trained_epochs=0, #FLAGS.trained_epochs, 
                verbose=FLAGS.verbose, 
                **architecture_params)
        elif FLAGS.trainer == 'CGANTrainer':
            discriminator_params = dict(
                n_filters=FLAGS.n_disc_filters,
                n_res_blocks=FLAGS.n_disc_blocks,
                normalization=FLAGS.normalization,
                activation=FLAGS.activation,
                attention=FLAGS.attention)

            trainer = dds.CGANTrainer(
                backbone=FLAGS.backbone, 
                upsampling=FLAGS.upsampling,
                data_train=DATA.data_train, 
                data_test=DATA.data_test, 
                data_train_lr=DATA.data_train_lr if FLAGS.paired_samples == 'explicit' else None,
                data_test_lr=DATA.data_test_lr if FLAGS.paired_samples == 'explicit' else None,
                predictors_train=DATA.predictors_train,
                predictors_test=DATA.predictors_test,
                scale=FLAGS.scale, 
                patch_size=FLAGS.patch_size, 
                time_window=FLAGS.time_window,
                loss=FLAGS.loss,
                epochs=epochs, 
                batch_size=FLAGS.batch_size,
                learning_rates=FLAGS.learning_rate, 
                device=FLAGS.device,
                gpu_memory_growth=FLAGS.gpu_memory_growth,
                steps_per_epoch=steps_per_epoch,
                interpolation=FLAGS.interpolation, 
                static_vars=DATA.static_vars,
                checkpoints_frequency=FLAGS.checkpoints_frequency, 
                save=FLAGS.save,
                save_path=FLAGS.save_path,
                save_logs=False,
                save_loss_history=FLAGS.save,
                verbose=FLAGS.verbose,
                generator_params=architecture_params,
                discriminator_params=discriminator_params)

        trainer.run()

    if FLAGS.test:
        if running_on_first_worker:
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DL4DS Test phase >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if DATA.inference_scaler is None:
            inference_scaler = None
        else:
            inference_scaler = DATA.inference_scaler

        if not has_horovod or running_on_first_worker:
            predictor = dds.Predictor(
                trainer=trainer,
                array=DATA.inference_data, 
                array_in_hr=FLAGS.inference_array_in_hr, 
                scale=FLAGS.scale, 
                interpolation=FLAGS.interpolation, 
                predictors=DATA.inference_predictors, 
                static_vars=DATA.static_vars, 
                time_window=FLAGS.time_window, 
                batch_size=FLAGS.batch_size,
                scaler=inference_scaler,
                save_path=FLAGS.save_path, 
                save_fname=FLAGS.inference_save_fname,
                device=FLAGS.device)

            y_hat = predictor.run()

            # Saving the downscaled product in netcdf format
            y_hat_datarray = xr.DataArray(data=np.squeeze(y_hat), 
                                          dims=('time', 'lat', 'lon'), 
                                          coords={'time':DATA.gt_holdout_dataset.time, 
                                                  'lon':DATA.gt_holdout_dataset.lon, 
                                                  'lat':DATA.gt_holdout_dataset.lat})
            
            if FLAGS.save_path is not None:
                y_hat_datarray.to_netcdf(f'{FLAGS.save_path}y_hat.nc')

    if FLAGS.metrics:
        if running_on_first_worker:
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<< DL4DS Metrics computation phase >>>>>>>>>>>>>>>>>>>>>>\n')
        if not has_horovod or running_on_first_worker:
            metrics = dds.compute_metrics(
                y_test=DATA.gt_holdout_dataset, 
                y_test_hat=y_hat, 
                dpi=300, plot_size_px=1200, 
                mask=DATA.gt_mask, 
                save_path=FLAGS.save_path,
                n_jobs=-1)

if __name__ == '__main__':
    app.run(dl4ds)


