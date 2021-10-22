import os
import datetime
import numpy as np
import xarray as xr
import tensorflow as tf
from abc import ABC, abstractmethod
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Progbar
from matplotlib.pyplot import show, close
import horovod.tensorflow.keras as hvd

from . import POSTUPSAMPLING_METHODS, MODELS, SPATIAL_MODELS, SPATIOTEMP_MODELS
from .utils import (Timing, list_devices, set_gpu_memory_growth, 
                    set_visible_gpus, checkarg_model)
from .dataloader import DataGenerator, create_batch_hr_lr
from .losses import mae, mse, dssim, dssim_mae, dssim_mae_mse, dssim_mse
from .models import (net_pin, recnet_pin, net_postupsampling, 
                     recnet_postupsampling, residual_discriminator)
from .cgan import train_step


class Trainer(ABC):
    """        
    """
    def __init__(
        self,
        model_name, 
        data_train,
        loss='mae',
        batch_size=64, 
        patch_size=None,
        scale=4,
        device='GPU', 
        gpu_memory_growth=True,
        use_multiprocessing=False,
        verbose=True, 
        model_list=None,
        save=True,
        save_path=None,
        show_plot=False,
        ):
        """
        """
        self.model_name = model_name
        self.data_train = data_train
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.loss = loss
        self.scale = scale
        self.device = device
        self.gpu_memory_growth = gpu_memory_growth
        self.use_multiprocessing = use_multiprocessing
        self.verbose = verbose
        self.model_list = model_list
        self.save = save
        self.save_path = save_path
        self.show_plot = show_plot
        if self.save_path is None:
            self.save_path = './'
        else:
            if not self.save_path.endswith('/'):
                self.save_path += '/'
        self.upsampling = model_name.split('_')[-1]
        self.backbone = self.model_name.split('_')[0]
        if self.backbone.startswith('rec'):
            self.backbone = self.backbone[3:]
            self.model_is_spatiotemp = True
        else:
            self.model_is_spatiotemp = False
       
        ### Initializing Horovod
        hvd.init()

        ### Setting up devices
        if self.verbose in [1 ,2]:
            print('List of devices:')
        if self.device == 'GPU':
            if self.gpu_memory_growth:
                set_gpu_memory_growth(verbose=False)
            # pin GPU to be used to process local rank (one GPU per process)       
            set_visible_gpus(hvd.local_rank())
            devices = list_devices('physical', gpu=True, verbose=verbose) 
        elif device == 'CPU':
            devices = list_devices('physical', gpu=False, verbose=verbose)
        else:
            raise ValueError('device not recognized')

        n_devices = len(devices)
        if self.verbose in [1 ,2]:
            print ('Number of devices: {}'.format(n_devices))
        batch_size_per_replica = self.batch_size
        self.global_batch_size = batch_size_per_replica * n_devices
        if self.verbose in [1 ,2]:
            print(f'Global batch size: {self.global_batch_size}, per replica: {batch_size_per_replica}')

        # identifying the first Horovod worker (for distributed training with GPUs), or CPU training
        if (self.device == 'GPU' and hvd.rank() == 0) or self.device == 'CPU':
            self.running_on_first_worker = True
        else:
            self.running_on_first_worker = False

        ### Checking the model argument
        if self.model_list is None:
            self.model_list = MODELS
        self.model_name = checkarg_model(self.model_name, self.model_list)
        
        ### Checking scale wrt image size
        if self.patch_size is not None: 
            imsize = self.patch_size
        else:
            imsize = self.data_train.shape[-2]
        
        if self.scale is not None:
            if imsize % self.scale != 0:
                msg = 'The image size must be divisible by `scale` (remainder must be zero). '
                msg += 'Crop the images or set `patch_size` accordingly'
                raise ValueError(msg)  

        ### Choosing the loss function
        if loss == 'mae':  
            self.lossf = mae
        elif loss == 'mse':  
            self.lossf = mse
        elif loss == 'dssim':
            self.lossf = dssim
        elif loss == 'dssim_mae':
            self.lossf = dssim_mae
        elif loss == 'dssim_mse':
            self.lossf = dssim_mse
        elif loss == 'dssim_mae_mse':
            self.lossf = dssim_mae_mse
        else:
            raise ValueError('`loss` not recognized')

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    def save_results(self, model_to_save=None, folder_prefix=None):
        """ 
        Save the TF model, learning curve, running time and test score. 
        """
        if self.save:     
            if model_to_save is None:
                model_to_save = self.model

            if folder_prefix is not None:
                self.model_save_path = self.save_path + folder_prefix + self.model_name + '/'
            else:
                self.model_save_path = self.save_path + self.model_name + '/'

            if self.running_on_first_worker:
                os.makedirs(self.model_save_path, exist_ok=True)
                model_to_save.save(self.model_save_path, save_format='tf')        
                np.savetxt(self.save_path + 'running_time.txt', [self.timing.running_time], fmt='%s')
                np.savetxt(self.save_path + 'test_loss.txt', [self.test_loss], fmt='%0.6f')

            if hasattr(self, 'fithist'):
                learning_curve_fname = self.save_path + 'learning_curve.png'
                plot_history(self.fithist.history, path=learning_curve_fname)
                if self.show_plot:
                    show()
                else:
                    close()


class SupervisedTrainer(Trainer):
    """Procedure for training the supervised residual models
    """
    def __init__(
        self,
        model_name, 
        data_train, 
        data_val, 
        data_test,  
        predictors_train=None,
        predictors_val=None,
        predictors_test=None,
        loss='mae',
        batch_size=64, 
        device='GPU', 
        gpu_memory_growth=True,
        use_multiprocessing=False, 
        model_list=None,
        topography=None, 
        landocean=None,
        use_season=True,
        scale=5, 
        interpolation='bicubic', 
        patch_size=50, 
        time_window=None,
        epochs=60, 
        steps_per_epoch=None, 
        validation_steps=None, 
        test_steps=None,
        learning_rate=1e-4, 
        lr_decay_after=1e5,
        early_stopping=False, 
        patience=6, 
        min_delta=0, 
        show_plot=True, 
        save=False,
        save_path=None, 
        savecheckpoint_path=None,
        verbose=True,
        **architecture_params
        ):
        """Procedure for training supervised models.

        Parameters
        ----------
        model : str
            String with the name of the model architecture, either 'resnet_spc', 
            'resnet_bi' or 'resnet_rc'.
        data_train : 4D ndarray
            Training dataset with dims [nsamples, lat, lon, 1]. This grids must 
            correspond to the observational reference at HR, from which a 
            coarsened version will be created to produce paired samples. 
        data_val : 4D ndarray
            Validation dataset with dims [nsamples, lat, lon, 1]. This holdout 
            dataset is used at the end of each epoch to check the losses and 
            diagnose overfitting.
        data_test : 4D ndarray
            Testing dataset with dims [nsamples, lat, lon, 1]. Holdout not used
            during training. 
        predictors_train : list of ndarray, optional
            Predictor variables for trianing. Given as list of 4D ndarrays with 
            dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        predictors_val : list of ndarray, optional
            Predictor variables for validation. Given as list of 4D ndarrays
            with dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        predictors_test : list of ndarrays, optional
            Predictor variables for testing. Given as list of 4D ndarrays with 
            dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        topography : None or 2D ndarray, optional
            Elevation data.
        landocean : None or 2D ndarray, optional
            Binary land-ocean mask.
        scale : int, optional
            Scaling factor. 
        interpolation : str, optional
            Interpolation used when upsampling/downsampling the training samples.
            By default 'bicubic'. 
        patch_size : int or None, optional
            Size of the square patches used to grab training samples.
        time_window : int or None, optional
            If not None, then each sample will have a temporal dimension 
            (``time_window`` slices to the past are grabbed for the LR array).
        batch_size : int, optional
            Batch size per replica.
        epochs : int, optional
            Number of epochs or passes through the whole training dataset. 
        steps_per_epoch : int or None, optional
            Total number of steps (batches of samples) before decalrin one epoch
            finished.``batch_size * steps_per_epoch`` samples are passed per 
            epoch. If None, ``then steps_per_epoch`` is equal to the number of 
            samples diviced by the ``batch_size``.
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
            Path for saving the final model, running time and test score. If 
            None, then ``'./saved_model/'`` is used. The SavedModel format is a 
            directory containing a protobuf binary and a TensorFlow checkpoint.
        savecheckpoint_path : None or str
            Path for saving the training checkpoints. If None, then no 
            checkpoints are saved during training. 
        device : str
            Choice of 'GPU' or 'CPU' for the training of the Tensorflow models. 
        gpu_memory_growth : bool, optional
            By default, TensorFlow maps nearly all of the GPU memory of all GPUs.
            If True, we request to only grow the memory usage as is needed by 
            the process.
        show_plot : bool, optional
            If True the static plot is shown after training. 
        save_plot : bool, optional
            If True the static plot is saved to disk after training. 
        verbose : bool, optional
            Verbosity mode. False or 0 = silent. True or 1, max amount of 
            information is printed out. When equal 2, then less info is shown.
        **architecture_params : dict
            Dictionary with additional parameters passed to the neural network 
            model.
        """
        self.use_season = use_season
        if self.use_season:
            if not isinstance(data_train, xr.DataArray):
                msg = '`data_train` must be a xr.DataArray when use_season=True'
                raise TypeError(msg)
        else:
            # removing the time metadata (season is not used as input)
            data_train = data_train.values
            data_test = data_test.values
            data_val = data_val.values

        super().__init__(
            model_name=model_name, 
            data_train=data_train,
            loss=loss,
            batch_size=batch_size, 
            patch_size=patch_size,
            scale=scale,
            device=device, 
            gpu_memory_growth=gpu_memory_growth,
            use_multiprocessing=use_multiprocessing,
            verbose=verbose, 
            model_list=model_list,
            save=save,
            save_path=save_path,
            show_plot=show_plot
            )
        self.data_val = data_val
        self.data_test = data_test
        self.predictors_train = predictors_train
        if self.predictors_train is not None and not isinstance(self.predictors_train, list):
            raise TypeError('`predictors_train` must be a list of ndarrays')
        self.predictors_test = predictors_test
        if self.predictors_test is not None and not isinstance(self.predictors_test, list):
            raise TypeError('`predictors_test` must be a list of ndarrays')
        self.predictors_val = predictors_val
        if self.predictors_val is not None and not isinstance(self.predictors_val, list):
            raise TypeError('`predictors_val` must be a list of ndarrays')
        self.topography = topography 
        self.landocean = landocean
        self.interpolation = interpolation 
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.test_steps = test_steps
        self.learning_rate = learning_rate
        self.lr_decay_after = lr_decay_after
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.savecheckpoint_path = savecheckpoint_path
        self.show_plot = show_plot
        self.architecture_params = architecture_params
        self.time_window = time_window
        if self.time_window is not None and not self.model_is_spatiotemp:
            self.time_window = None
        if self.model_is_spatiotemp and self.time_window is None:
            msg = f'``model={self.model_name}``, the argument ``time_window`` must be a postive integer'
            raise ValueError(msg)

    def setup_datagen(self):
        """Setting up the data generators
        """
        datagen_params = dict(
            scale=self.scale, 
            batch_size=self.global_batch_size,
            topography=self.topography, 
            landocean=self.landocean, 
            patch_size=self.patch_size, 
            model=self.model_name, 
            interpolation=self.interpolation,
            time_window=self.time_window)
        self.ds_train = DataGenerator(self.data_train, predictors=self.predictors_train, **datagen_params)
        self.ds_val = DataGenerator(self.data_val, predictors=self.predictors_val, **datagen_params)
        self.ds_test = DataGenerator(self.data_test, predictors=self.predictors_test, **datagen_params)

    def setup_model(self):
        """Setting up the model
        """
        ### number of channels
        if self.model_name in SPATIAL_MODELS:
            n_channels = self.data_train.shape[-1]
            n_aux_channels = 0
            if self.topography is not None:
                n_channels += 1
                n_aux_channels = 1
            if self.landocean is not None:
                n_channels += 1
                n_aux_channels += 1
            if isinstance(self.data_train, xr.DataArray):
                n_channels += 4
                n_aux_channels += 4
            if self.predictors_train is not None:
                n_channels += len(self.predictors_train)
        elif self.model_name in SPATIOTEMP_MODELS:
            n_channels = self.data_train.shape[-1]
            n_aux_channels = 0
            if self.predictors_train is not None:
                n_channels += len(self.predictors_train)
            if self.topography is not None:
                n_aux_channels += 1
            if self.landocean is not None:
                n_aux_channels += 1
            if isinstance(self.data_train, xr.DataArray):
                n_aux_channels += 4

        if self.patch_size is None:
            lr_height = int(self.data_train.shape[1] / self.scale)
            lr_width = int(self.data_train.shape[2] / self.scale)
            hr_height = int(self.data_train.shape[1])
            hr_width = int(self.data_train.shape[2])
        else:
            lr_height = lr_width = int(self.patch_size / self.scale)
            hr_height = hr_width = int(self.patch_size)

        ### instantiating and fitting the model
        if self.upsampling in POSTUPSAMPLING_METHODS:
            if not self.model_is_spatiotemp:
                self.model = net_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling, 
                    scale=self.scale, 
                    lr_size=(lr_height, lr_width),
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    **self.architecture_params)
            else:
                self.model = recnet_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling, 
                    scale=self.scale, 
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    lr_size=(lr_height, lr_width),
                    time_window=self.time_window, 
                    **self.architecture_params)
        elif self.upsampling == 'pin':
            if not self.model_is_spatiotemp:
                self.model = net_pin(
                    backbone_block=self.backbone,
                    n_channels=n_channels, 
                    hr_size=(hr_height, hr_width),
                    n_aux_channels=n_aux_channels,
                    **self.architecture_params)        
            else:
                self.model = recnet_pin(
                    backbone_block=self.backbone,
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    time_window=self.time_window, 
                    **self.architecture_params)

        if self.verbose == 1 and self.running_on_first_worker:
            self.model.summary(line_length=150)

    def run(self):
        """Compiling, training and saving the model
        """
        self.timing = Timing(self.verbose)
        self.setup_datagen()
        self.setup_model()

        ### Setting up the optimizer
        if isinstance(self.learning_rate, tuple):
            ### Adam optimizer with a scheduler 
            self.learning_rate = PiecewiseConstantDecay(boundaries=[self.lr_decay_after], 
                                                        values=[self.learning_rate[0], 
                                                                self.learning_rate[1]])
        elif isinstance(self.learning_rate, float):
            # as in Goyan et al 2018 (https://arxiv.org/abs/1706.02677)
            self.learning_rate *= hvd.size()
        self.optimizer = Adam(learning_rate=self.learning_rate)

        ### Callbacks
        # early stopping
        callbacks = []
        if self.early_stopping:
            earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience, 
                                      min_delta=self.min_delta, verbose=self.verbose)
            callbacks.append(earlystop)

        # Horovod: add Horovod DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(self.optimizer)
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        
        # verbosity for model.fit
        if self.verbose == 1 and self.running_on_first_worker:
            verbose = 1
        elif self.verbose == 2 and self.running_on_first_worker:
            verbose = 2
        else:
            verbose = 0

        # Model checkopoints are saved at the end of every epoch, if it's the best seen so far.
        if self.savecheckpoint_path is not None:
            os.makedirs(self.savecheckpoint_path, exist_ok=True)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.savecheckpoint_path, './best_model'), 
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
            if self.running_on_first_worker:
                callbacks.append(model_checkpoint_callback)

        ### Compiling and training the model
        if self.steps_per_epoch is not None:
            self.steps_per_epoch = self.steps_per_epoch // hvd.size()

        self.model.compile(optimizer=self.optimizer, loss=self.lossf)
        self.fithist = self.model.fit(
            self.ds_train, 
            epochs=self.epochs, 
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.ds_val, 
            validation_steps=self.validation_steps, 
            verbose=self.verbose if self.running_on_first_worker else False, 
            callbacks=callbacks,
            use_multiprocessing=self.use_multiprocessing)
        
        if self.running_on_first_worker:
            self.test_loss = self.model.evaluate(
                self.ds_test, 
                steps=self.test_steps, 
                verbose=verbose)
            
            if self.verbose:
                print(f'\nScore on the test set: {self.test_loss}')
            
            self.timing.runtime()

        self.save_results(self.model)


class CGANTrainer(Trainer):
    """
    """
    def __init__(
        self,
        model_name,
        data_train,
        data_test,
        predictors_train=None,
        predictors_test=None,
        scale=5, 
        patch_size=50, 
        time_window=None,
        loss='mae',
        epochs=60, 
        batch_size=16,
        learning_rates=(2e-4, 2e-4),
        device='GPU',
        gpu_memory_growth=True,
        model_list=None,
        steps_per_epoch=None,
        interpolation='bicubic', 
        topography=None, 
        landocean=None, 
        checkpoints_frequency=5, 
        savecheckpoint_path=None,
        save=False,
        save_path=None,
        save_logs=False,
        save_loss_history=True,
        generator_params={},
        discriminator_params={},
        verbose=True,
        ):
        """Procedure for training CGAN models.
    
        Parameters
        ----------
        model_name : str
            String with the name of the model architecture, either 'resnet_spc', 
            'resnet_bi' or 'resnet_rc'. Used as a the CGAN generator.
        data_train : 4D ndarray
            Training dataset with dims [nsamples, lat, lon, 1].
        data_test : 4D ndarray
            Testing dataset with dims [nsamples, lat, lon, 1]. Holdout not used
            during training. 
        predictors_train : list of ndarray, optional
            Predictor variables for trianing. Given as list of 4D ndarrays with 
            dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        predictors_test : list of ndarray, optional
            Predictor variables for testing. Given as list of 4D ndarrays with 
            dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        epochs : int, optional
            Number of epochs or passes through the whole training dataset. 
        steps_per_epoch : int, optional
            ``batch_size * steps_per_epoch`` samples are passed per epoch.
        scale : int, optional
            Scaling factor. 
        interpolation : str, optional
            Interpolation used when upsampling/downsampling the training samples.
            By default 'bicubic'. 
        patch_size : int, optional
            Size of the square patches used to grab training samples.
        batch_size : int, optional
            Batch size per replica.
        topography : None or 2D ndarray, optional
            Elevation data.
        landocean : None or 2D ndarray, optional
            Binary land-ocean mask.
        checkpoints_frequency : int, optional
            The training loop saves a checkpoint every ``checkpoints_frequency`` 
            epochs. If None, then no checkpoints are saved during training. 
        savecheckpoint_path : None or str
            Path for saving the training checkpoints. If None, then no checkpoints
            are saved during training. 
        device : str
            Choice of 'GPU' or 'CPU' for the training of the Tensorflow models. 
        gpu_memory_growth : bool, optional
            By default, TensorFlow maps nearly all of the GPU memory of all GPUs.
            If True, we request to only grow the memory usage as is needed by the 
            process.
        verbose : bool, optional
            Verbosity mode. False or 0 = silent. True or 1, max amount of 
            information is printed out. When equal 2, then less info is shown.
        """
        super().__init__(
            model_name=model_name, 
            data_train=data_train, 
            loss=loss, 
            batch_size=batch_size, 
            patch_size=patch_size, 
            scale=scale, 
            device=device, 
            gpu_memory_growth=gpu_memory_growth,
            verbose=verbose, 
            model_list=model_list, 
            save=save, 
            save_path=save_path, 
            show_plot=False
            )
        self.data_test = data_test
        self.scale = scale
        self.patch_size = patch_size
        self.time_window = time_window
        self.predictors_train = predictors_train
        if self.predictors_train is not None and not isinstance(self.predictors_train, list):
            raise TypeError('`predictors_train` must be a list of ndarrays')
        self.predictors_test = predictors_test
        if self.predictors_test is not None and not isinstance(self.predictors_test, list):
            raise TypeError('`predictors_test` must be a list of ndarrays')
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.steps_per_epoch = steps_per_epoch
        self.interpolation = interpolation 
        self.topography = topography 
        self.landocean = landocean
        self.checkpoints_frequency = checkpoints_frequency
        self.savecheckpoint_path = savecheckpoint_path
        self.save_loss_history = save_loss_history
        self.save_logs = save_logs
        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        self.gentotal = []
        self.gengan = []
        self.gen_pxloss = []
        self.disc = []

        self.time_window = time_window
        if self.time_window is not None and not self.model_is_spatiotemp:
            self.time_window = None
        if self.model_is_spatiotemp and self.time_window is None:
            msg = f'``model={self.model_name}``, the argument ``time_window`` must be a postive integer'
            raise ValueError(msg)

    def setup_model(self):
        """
        """
        n_channels = self.data_train.shape[-1]
        n_aux_channels = 0
        if self.topography is not None:
            if not self.model_is_spatiotemp:
                n_channels += 1
            n_aux_channels += 1
        if self.landocean is not None:
            if not self.model_is_spatiotemp:
                n_channels += 1
            n_aux_channels += 1
        if isinstance(self.data_train, xr.DataArray):
            if not self.model_is_spatiotemp:
                n_channels += 4
            n_aux_channels += 4
        if self.predictors_train is not None:
            n_channels += len(self.predictors_train)
        
        if self.model_is_spatiotemp:
            if self.patch_size is None:
                lr_height = int(self.data_train.shape[1] / self.scale)
                lr_width = int(self.data_train.shape[2] / self.scale)
            else:
                lr_height = lr_width = int(self.patch_size / self.scale)  

        # Generator
        if self.upsampling in POSTUPSAMPLING_METHODS:
            if not self.model_is_spatiotemp:
                self.generator = net_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling,
                    scale=self.scale, 
                    n_channels=n_channels,
                    n_aux_channels=n_aux_channels,
                    **self.generator_params)
            else:
                self.generator = recnet_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling, 
                    scale=self.scale, 
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    lr_size=(lr_height, lr_width),
                    time_window=self.time_window, 
                    **self.generator_params)
        elif self.upsampling == 'pin':
            if not self.model_is_spatiotemp:
                self.generator = net_pin(
                    backbone_block=self.backbone,
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    **self.generator_params)            
            else:
                self.generator = recnet_pin(
                    backbone_block=self.backbone,
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    time_window=self.time_window, 
                    **self.generator_params)

        # Discriminator
        n_channels_disc = n_channels[0] if isinstance(n_channels, tuple) else n_channels
        self.discriminator = residual_discriminator(n_channels=n_channels_disc, 
                                                    scale=self.scale, 
                                                    model=self.model_name,
                                                    **self.discriminator_params)
        
        if self.verbose == 1 and self.running_on_first_worker:
            self.generator.summary(line_length=150)
            self.discriminator.summary(line_length=150)

    def run(self):
        """
        """
        self.timing = Timing(self.verbose)
        self.setup_model()

        # Optimizers
        if isinstance(self.learning_rates, tuple):
            genlr, dislr = self.learning_rates
        elif isinstance(self.learning_rates, float):
            genlr = dislr = self.learning_rates
        generator_optimizer = tf.keras.optimizers.Adam(genlr, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(dislr, beta_1=0.5)
        
        if self.save_logs:
            log_dir = "cgan_logs/"
            log_path = log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(log_path)
        else:
            summary_writer = None

        # Checkpoint
        if self.savecheckpoint_path is not None:
            checkpoint_prefix = os.path.join(self.savecheckpoint_path + 'chkpts/', 'checkpoint_epoch')
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=self.generator, discriminator=self.discriminator)

        n_samples = self.data_train.shape[0]
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(n_samples / self.batch_size)

        # creating a single ndarrays concatenating list of ndarray variables along the last dimension 
        if self.predictors_train is not None:
            self.predictors_train = np.concatenate(self.predictors_train, axis=-1)
        else:
            self.predictors_train = None

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            pb_i = Progbar(self.steps_per_epoch, stateful_metrics=['gen_total_loss', 
                                                                   'gen_crosentr_loss', 
                                                                   'gen_mae_loss', 
                                                                   'disc_loss'])

            for i in range(self.steps_per_epoch):
                res = create_batch_hr_lr(
                    self.data_train,
                    batch_size=self.global_batch_size,
                    predictors=self.predictors_train, 
                    scale=self.scale, 
                    topography=self.topography, 
                    landocean=self.landocean, 
                    patch_size=self.patch_size, 
                    time_window=self.time_window,
                    model=self.model_name, 
                    interpolation=self.interpolation)

                hr_array = res[0]
                lr_array = res[1]
                static_array = None
                if self.topography is not None or self.landocean is not None or isinstance(self.data_train, xr.DataArray):
                    static_array = res[2]

                losses = train_step(
                    lr_array, 
                    hr_array, 
                    generator=self.generator, 
                    discriminator=self.discriminator, 
                    generator_optimizer=generator_optimizer, 
                    discriminator_optimizer=discriminator_optimizer, 
                    epoch=epoch, 
                    gen_pxloss_function=self.lossf,
                    summary_writer=summary_writer, 
                    first_batch=True if epoch==0 and i==0 else False,
                    static_array=static_array)
                
                gen_total_loss, gen_gan_loss, gen_px_loss, disc_loss = losses
                lossvals = [('gen_total_loss', gen_total_loss), 
                            ('gen_crosentr_loss', gen_gan_loss), 
                            ('gen_px_loss', gen_px_loss), 
                            ('disc_loss', disc_loss)]
                
                if self.running_on_first_worker:
                    pb_i.add(1, values=lossvals)
            
            self.gentotal.append(gen_total_loss)
            self.gengan.append(gen_gan_loss)
            self.gen_pxloss.append(gen_px_loss)
            self.disc.append(disc_loss)
            
            if self.savecheckpoint_path is not None:
                # Horovod: save checkpoints only on worker 0 to prevent other 
                # workers from corrupting it
                if self.running_on_first_worker:
                    if (epoch + 1) % self.checkpoints_frequency == 0:
                        checkpoint.save(file_prefix=checkpoint_prefix)
        
        # Horovod: save last checkpoint only on worker 0 to prevent other 
        # workers from corrupting it
        if self.savecheckpoint_path is not None and self.running_on_first_worker:
            checkpoint.save(file_prefix=checkpoint_prefix)

        if self.save_loss_history and self.running_on_first_worker:
            losses_array = np.array((self.gentotal, self.gengan, self.gen_pxloss, self.disc))
            np.save('./losses.npy', losses_array)

        self.timing.checktime()

        ### Loss on the Test set
        if self.predictors_test is not None:
            self.predictors_test = np.concatenate(self.predictors_test, axis=-1)
        else:
            self.predictors_test = None
            
        if self.running_on_first_worker:
            test_steps = int(self.data_test.shape[0] / self.batch_size)
            
            res = create_batch_hr_lr(
                self.data_test,
                batch_size=test_steps,
                predictors=self.predictors_test, 
                scale=self.scale, 
                topography=self.topography, 
                landocean=self.landocean, 
                patch_size=self.patch_size, 
                time_window=self.time_window,
                model=self.model_name, 
                interpolation=self.interpolation,
                shuffle=False)

            hr_arrtest = tf.cast(res[0], tf.float32)
            lr_arrtest = tf.cast(res[1], tf.float32)
            static_array_test = None
            if self.topography is not None or self.landocean is not None or isinstance(self.data_train, xr.DataArray):
                static_array_test = tf.cast(res[2], tf.float32)
                input_test = [lr_arrtest, static_array_test]
            else:
                input_test = lr_arrtest
            
            y_test_pred = self.generator.predict(input_test)
            self.test_loss = self.lossf(hr_arrtest, y_test_pred)
            print(f'\n{self.lossf.__name__} on the test set: {self.test_loss}')
        
        self.timing.runtime()

        self.save_results(self.generator, folder_prefix='cgan_')
        

