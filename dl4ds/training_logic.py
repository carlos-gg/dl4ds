import os
import datetime
import numpy as np
import livelossplot
import tensorflow as tf
from abc import ABC, abstractmethod
from plot_keras_history import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Progbar
from matplotlib.pyplot import show
import horovod.tensorflow.keras as hvd

from .utils import (Timing, list_devices, set_gpu_memory_growth, 
                    set_visible_gpus, checkarg_model, MODELS)
from .dataloader import DataGenerator, create_batch_hr_lr
from .losses import dssim, dssim_mae, dssim_mae_mse, dssim_mse
from .resnet_preupsampling import resnet_bi, recresnet_bi
from .resnet_postupsampling import resnet_postupsampling, recresnet_postupsampling
from .cgan import train_step
from .discriminator import residual_discriminator


class Trainer(ABC):
    """        
    """
    def __init__(
        self,
        model_name, 
        loss='mae',
        batch_size=64, 
        device='GPU', 
        gpu_memory_growth=True,
        use_multiprocessing=False,
        verbose=True, 
        model_list=None,
        save=True,
        save_path=None
        ):
        """
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.loss = loss
        self.device = device
        self.gpu_memory_growth = gpu_memory_growth
        self.use_multiprocessing = use_multiprocessing
        self.verbose = verbose
        self.save = save
        self.save_path = save_path
        self.timing = Timing()
       
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
        if model_list is None:
            model_list = MODELS
        self.model_name = checkarg_model(self.model_name, model_list)

        ### Choosing the loss function
        if loss == 'mae':  # L1 pixel loss
            self.lossf = tf.keras.losses.MeanAbsoluteError()
        elif loss == 'mse':  # L2 pixel loss
            self.lossf = tf.keras.losses.MeanSquaredError()
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

    def save_model(self, model_to_save=None):
        if model_to_save is None:
            model_to_save = self.model

        if self.save_path is None:
            self.save_path = './' + self.model_name + '/'
    
        if self.running_on_first_worker:
            os.makedirs(self.save_path, exist_ok=True)
            model_to_save.save(self.save_path, save_format='tf')        
        

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
        plot='plt', 
        show_plot=True, 
        save_plot=False,
        save=False,
        save_path=None, 
        savecheckpoint_path='./checkpoints/',
        verbose=True,
        **architecture_params
        ):
        """Procedure for training the supervised residual models

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
        predictors_train : tuple of 4D ndarray, optional
            Predictor variables for trianing. Given as tuple of 4D ndarray with 
            dims [nsamples, lat, lon, 1]. 
        predictors_val : tuple of 4D ndarray, optional
            Predictor variables for validation. Given as tuple of 4D ndarray 
            with dims [nsamples, lat, lon, 1]. 
        predictors_test : tuple of 4D ndarray, optional
            Predictor variables for testing. Given as tuple of 4D ndarray with 
            dims [nsamples, lat, lon, 1]. 
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
            Path for saving the final model. If None, then ``'./saved_model/'`` 
            is used. The SavedModel format is a directory containing a protobuf 
            binary and a TensorFlow checkpoint.
        savecheckpoint_path : None or str
            Path for saving the training checkpoints. If None, then no 
            checkpoints are saved during training. 
        device : str
            Choice of 'GPU' or 'CPU' for the training of the Tensorflow models. 
        gpu_memory_growth : bool, optional
            By default, TensorFlow maps nearly all of the GPU memory of all GPUs.
            If True, we request to only grow the memory usage as is needed by 
            the process.
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
            Dictionary with additional parameters passed to the neural network 
            model.
        """
        super().__init__(model_name, loss, batch_size, device, gpu_memory_growth,
                         use_multiprocessing, verbose, model_list, save, save_path)
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.predictors_train = predictors_train
        self.predictors_val = predictors_val
        self.predictors_test = predictors_test
        self.topography = topography 
        self.landocean = landocean
        self.scale = scale
        self.interpolation = interpolation 
        self.patch_size = patch_size 
        self.time_window = time_window
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
        self.plot = plot
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.architecture_params = architecture_params

        self.setup_datagen()
        self.setup_model()
        self.run()
        if self.save:
            self.save_model(self.model)

    def setup_datagen(self):
        """Setting up the data generators
        """
        if self.patch_size is not None and self.patch_size % self.scale != 0:
            raise ValueError('`patch_size` must be divisible by `scale` (remainder must be zero)')

        recmodels = ['recresnet_spc', 'recresnet_rc', 'recresnet_bi']
        if self.time_window is not None and self.model_name not in recmodels:
            msg = f'``time_window={self.time_window}``, choose a model that handles samples with a temporal dimension'
            raise ValueError(msg)
        if self.model_name in recmodels and self.time_window is None:
            msg = f'``model={self.model_name}``, the argument ``time_window`` must be a postive integer'
            raise ValueError(msg)

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
        if self.model_name in ['resnet_spc', 'resnet_bi', 'resnet_rc', 'resnet_dc']:
            n_channels = self.data_train.shape[-1]
            if self.topography is not None:
                n_channels += 1
            if self.landocean is not None:
                n_channels += 1
            if self.predictors_train is not None:
                n_channels += len(self.predictors_train)
        elif self.model_name in ['recresnet_spc', 'recresnet_rc', 'recresnet_dc', 'recresnet_bi']:
            n_var_channels = self.data_train.shape[-1]
            n_st_channels = 0
            if self.predictors_train is not None:
                n_var_channels += len(self.predictors_train)
            if self.topography is not None:
                n_st_channels += 1
            if self.landocean is not None:
                n_st_channels += 1
            n_channels = (n_var_channels, n_st_channels)

        ### instantiating and fitting the model
        if self.model_name in ['resnet_spc', 'resnet_rc', 'resnet_dc']:
            upsampling_module_name = self.model_name.split('_')[-1]
            self.model = resnet_postupsampling(
                upsampling=upsampling_module_name, 
                scale=self.scale, 
                n_channels=n_channels, 
                **self.architecture_params)
        elif self.model_name == 'resnet_bi':
            self.model = resnet_bi(
                n_channels=n_channels, 
                **self.architecture_params)        
        elif self.model_name in ['recresnet_spc', 'recresnet_rc', 'recresnet_dc']:
            upsampling_module_name = self.model_name.split('_')[-1]
            self.model = recresnet_postupsampling(
                upsampling=upsampling_module_name, 
                scale=self.scale, 
                n_channels=n_channels, 
                time_window=self.time_window, 
                **self.architecture_params)
        elif self.model_name == 'recresnet_bi':
            self.model = recresnet_bi(
                n_channels=n_channels, 
                time_window=self.time_window, 
                **self.architecture_params)

        if self.verbose == 1 and self.running_on_first_worker:
            self.model.summary(line_length=150)

    def run(self):
        """Compiling, training and saving the model
        """
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
        # loss plotting
        if self.plot == 'llp':
            plotlosses = livelossplot.PlotLossesKerasTF()
            callbacks.append(plotlosses) 

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
                os.path.join(self.savecheckpoint_path, './checkpoint_epoch-{epoch:02d}.h5'),
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
            verbose=self.verbose, 
            callbacks=callbacks,
            use_multiprocessing=self.use_multiprocessing)
        self.score = self.model.evaluate(
            self.ds_test, 
            steps=self.test_steps, 
            verbose=verbose)
        print(f'\nScore on the test set: {self.score}')
        
        self.timing.runtime()
        
        if self.plot == 'plt':
            if self.save_plot:
                learning_curve_fname = self.model_name + '_learning_curve.png'
            else:
                learning_curve_fname = None
            
            if self.running_on_first_worker:
                plot_history(self.fithist.history, path=learning_curve_fname)
                if self.show_plot:
                    show()


class CGANTrainer(Trainer):
    """
    """
    def __init__(
        self,
        model_name,
        data_train,
        data_test,
        scale=5, 
        patch_size=50, 
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
        savecheckpoint_path='./checkpoints/',
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
        x_train : 4D ndarray
            Training dataset with dims [nsamples, lat, lon, 1].
        x_test : 4D ndarray
            Testing dataset with dims [nsamples, lat, lon, 1]. Holdout not used
            during training. 
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
        super().__init__(model_name, loss, batch_size, device, gpu_memory_growth,
                         verbose=verbose, model_list=model_list, save=save, 
                         save_path=save_path)
        self.data_train = data_train
        self.data_test = data_test
        self.scale = scale
        self.patch_size = patch_size
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
        
        self.n_channels = 1
        if self.topography is not None:
            self.n_channels += 1
        if self.landocean is not None:
            self.n_channels += 1
        
        self.setup_model()
        self.run()
        if self.save:
            self.save_model(self.generator)

    def setup_model(self):
        """
        """
        # Generator
        if self.model_name in ['resnet_spc', 'resnet_rc', 'resnet_dc']:
            upsampling_module_name = self.model_name.split('_')[-1]
            self.generator = resnet_postupsampling(
                upsampling=upsampling_module_name,
                scale=self.scale, 
                n_channels=self.n_channels,
                **self.generator_params)
        elif self.model_name == 'resnet_bi':
            self.generator = resnet_bi(n_channels=self.n_channels, 
                                       **self.generator_params)
            
        # Discriminator
        self.discriminator = residual_discriminator(n_channels=self.n_channels, 
                                                    scale=self.scale, 
                                                    model=self.model,
                                                    **self.discriminator_params)
        
        if self.verbose == 1 and self.running_on_first_worker:
            self.generator.summary(line_length=150)
            self.discriminator.summary(line_length=150)

    def run(self):
        """
        """
        # Optimizers
        if isinstance(self.learning_rates, tuple):
            genlr, dislr = self.learning_rates
        elif isinstance(self.learning_rates, float):
            genlr = dislr = self.learning_rates
        generator_optimizer = tf.keras.optimizers.Adam(genlr, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(dislr, beta_1=0.5)
        
        if self.save_logs:
            log_dir = "cgan_logs/"
            summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            summary_writer = None

        # Checkpoint
        if self.savecheckpoint_path is not None:
            checkpoint_prefix = os.path.join(self.savecheckpoint_path, 'checkpoint_epoch')
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                            discriminator_optimizer=discriminator_optimizer,
                                            generator=self.generator, discriminator=self.discriminator)
        
        if self.patch_size is not None and self.patch_size % self.scale != 0:
            raise ValueError('`patch_size` must be divisible by `scale` (remainder must be zero)')

        n_samples = self.data_train.shape[0]
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(n_samples / self.batch_size)

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            pb_i = Progbar(self.steps_per_epoch, stateful_metrics=['gen_total_loss', 
                                                                   'gen_crosentr_loss', 
                                                                   'gen_mae_loss', 
                                                                   'disc_loss'])

            for i in range(self.steps_per_epoch):
                hr_array, lr_array = create_batch_hr_lr(
                    self.data_train,
                    batch_size=self.global_batch_size,
                    tuple_predictors=None, 
                    scale=self.scale, 
                    topography=self.topography, 
                    landocean=self.landocean, 
                    patch_size=self.patch_size, 
                    model=self.model, 
                    interpolation=self.interpolation)

                losses = train_step(
                    lr_array, 
                    hr_array, 
                    self.generator, 
                    self.discriminator, 
                    generator_optimizer, 
                    discriminator_optimizer, 
                    epoch, 
                    self.lossf,
                    summary_writer, 
                    first_batch=True if epoch==0 and i==0 else False)
                
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
        if self.checkpoints_frequency is not None and self.running_on_first_worker:
            checkpoint.save(file_prefix=checkpoint_prefix)

        if self.save_loss_history and self.running_on_first_worker:
            losses_array = np.array((self.gentotal, self.gengan, self.gen_pxloss, self.disc))
            np.save('./losses.npy', losses_array)

        self.timing.checktime()

        ### Loss on the Test set
        if self.running_on_first_worker:
            test_steps = int(self.data_test.shape[0] / self.batch_size)
            
            hr_arrtest, lr_arrtest = create_batch_hr_lr(
                self.data_test,
                batch_size=test_steps,
                tuple_predictors=None, 
                scale=self.scale, 
                topography=self.topography, 
                landocean=self.landocean, 
                patch_size=self.patch_size, 
                model=self.model, 
                interpolation=self.interpolation,
                shuffle=False)

            y_test_pred = self.generator.predict(lr_arrtest)
            test_loss = self.lossf(hr_arrtest, y_test_pred)
            print(f'\n{self.lossf} on the test set: {test_loss}')
        
        self.timing.runtime()

