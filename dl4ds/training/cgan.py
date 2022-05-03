"""
Training procedure for Conditional GAN models. Follows Isola et al. 2016
"""

import os
import datetime
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import logging
tf.get_logger().setLevel(logging.ERROR)

try:
    import horovod.tensorflow as hvd
    has_horovod = True
except ImportError:
    has_horovod = False

from ..utils import Timing
from ..dataloader import create_batch_hr_lr
from ..models import (net_pin, recnet_pin, net_postupsampling, 
                     recnet_postupsampling, residual_discriminator)
from ..models import (net_postupsampling, recnet_postupsampling, net_pin, 
                     recnet_pin, unet_pin)
from .. import POSTUPSAMPLING_METHODS
from .base import Trainer


class CGANTrainer(Trainer):
    """
    """
    def __init__(
        self,
        backbone,
        upsampling,
        data_train,
        data_test,
        data_train_lr=None,
        data_test_lr=None,
        predictors_train=None,
        predictors_test=None,
        scale=5, 
        patch_size=None, 
        time_window=True,
        loss='mae',
        epochs=60, 
        batch_size=16,
        learning_rates=(2e-4, 2e-4),
        device='GPU',
        gpu_memory_growth=True,
        model_list=None,
        steps_per_epoch=None,
        interpolation='inter_area', 
        static_vars=None,
        checkpoints_frequency=0, 
        save=False,
        save_path=None,
        save_logs=False,
        save_loss_history=True,
        generator_params={},
        discriminator_params={},
        verbose=True,
        ):
        """Training conditional adversarial generative models.
    
        Parameters
        ----------
        backbone : str
            String with the name of the backbone block used for the CGAN 
            generator.
        upsampling : str
            String with the name of the upsampling method used for the CGAN 
            generator.
        data_train : 4D ndarray or xr.DataArray
            Training dataset with dims [nsamples, lat, lon, 1]. These grids must 
            correspond to the observational reference at HR, from which a 
            coarsened version will be created to produce paired samples. 
        data_test : 4D ndarray or xr.DataArray
            Testing dataset with dims [nsamples, lat, lon, 1]. Holdout not used
            during training, but only to compute metrics with the final model.
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
        patch_size : int, optional
            Size of the square patches used to grab training samples.
        batch_size : int, optional
            Batch size per replica.
        learning_rates : float or tuple of floats or list of floats, optional
            Learning rate for both the generator and discriminator. If a 
            tuple/list is given, it corresponds to the learning rates of the
            generator and the discriminator (in that order).
        static_vars : None or list of 2D ndarrays, optional
            Static variables such as elevation data or a binary land-ocean mask.
        checkpoints_frequency : int, optional
            The training loop saves a checkpoint every ``checkpoints_frequency`` 
            epochs. If None, then no checkpoints are saved during training. 
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
            backbone=backbone,
            upsampling=upsampling, 
            data_train=data_train, 
            data_train_lr=data_train_lr,
            time_window=time_window,
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
        self.data_test_lr = data_test_lr
        self.scale = scale
        self.patch_size = patch_size
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
        self.static_vars = static_vars 
        if self.static_vars is not None:
            for i in range(len(self.static_vars)):
                if isinstance(self.static_vars[i], xr.DataArray):
                    self.static_vars[i] = self.static_vars[i].values
        self.checkpoints_frequency = checkpoints_frequency
        self.save_loss_history = save_loss_history
        self.save_logs = save_logs
        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        self.gentotal = []
        self.gengan = []
        self.gen_pxloss = []
        self.disc = []

        self.time_window = time_window
        if self.time_window is not None and not self.model_is_spatiotemporal:
            self.time_window = None
        if self.model_is_spatiotemporal and self.time_window is None:
            raise ValueError('The argument `time_window` must be a postive integer for spatio-temporal models')

    def setup_model(self):
        """
        """
        n_channels = self.data_train.shape[-1]
        n_aux_channels = 0
        if self.model_is_spatiotemporal:
            n_channels = self.data_train.shape[-1]
            n_aux_channels = 0
            if self.predictors_train is not None:
                n_channels += len(self.predictors_train)
            if self.static_vars is not None:
                n_aux_channels += len(self.static_vars)
        else:
            n_channels = self.data_train.shape[-1]
            n_aux_channels = 0
            if self.static_vars is not None:
                n_channels += len(self.static_vars)
                n_aux_channels = len(self.static_vars)
            if self.predictors_train is not None:
                n_channels += len(self.predictors_train)
        
        if self.patch_size is None:
            lr_height = int(self.data_train.shape[1] / self.scale)
            lr_width = int(self.data_train.shape[2] / self.scale)
            hr_height = int(self.data_train.shape[1])
            hr_width = int(self.data_train.shape[2])
        else:
            lr_height = lr_width = int(self.patch_size / self.scale)
            hr_height = hr_width = int(self.patch_size)

        # Generator
        if self.upsampling in POSTUPSAMPLING_METHODS:
            if self.model_is_spatiotemporal:
                self.generator = recnet_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling, 
                    scale=self.scale, 
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    lr_size=(lr_height, lr_width),
                    time_window=self.time_window, 
                    **self.generator_params)
            else:
                self.generator = net_postupsampling(
                    backbone_block=self.backbone,
                    upsampling=self.upsampling,
                    scale=self.scale, 
                    n_channels=n_channels,
                    n_aux_channels=n_aux_channels,
                    lr_size=(lr_height, lr_width),
                    **self.generator_params)
            
        elif self.upsampling == 'pin':
            if self.model_is_spatiotemporal:
                self.generator = recnet_pin(
                    backbone_block=self.backbone,
                    n_channels=n_channels, 
                    n_aux_channels=n_aux_channels,
                    hr_size=(hr_height, hr_width),
                    time_window=self.time_window, 
                    **self.generator_params)
            else:
                if self.backbone == 'unet':
                    self.generator = unet_pin(
                        backbone_block=self.backbone,
                        n_channels=n_channels,
                        n_aux_channels=n_aux_channels,
                        hr_size=(hr_height, hr_width),
                        **self.generator_params)
                else:
                    self.generator = net_pin(
                        backbone_block=self.backbone,
                        n_channels=n_channels, 
                        n_aux_channels=n_aux_channels,
                        hr_size=(hr_height, hr_width),
                        **self.generator_params)            

        # Discriminator
        n_channels_disc = n_channels[0] if isinstance(n_channels, tuple) else n_channels
        self.discriminator = residual_discriminator(n_channels=n_channels_disc, 
                                                    scale=self.scale, 
                                                    upsampling=self.upsampling,
                                                    is_spatiotemporal=self.model_is_spatiotemporal,
                                                    lr_size=(lr_height, lr_width),
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
        if isinstance(self.learning_rates, (tuple, list)) and len(self.learning_rates) > 1:
            genlr, dislr = self.learning_rates
        elif isinstance(self.learning_rates, float) or (isinstance(self.learning_rates, (tuple, list)) and len(self.learning_rates) == 1):
            if isinstance(self.learning_rates, (tuple, list)) and len(self.learning_rates) == 1:
                self.learning_rates = self.learning_rates[0]
            genlr = dislr = self.learning_rates
        generator_optimizer = tf.keras.optimizers.Adam(genlr, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(dislr, beta_1=0.5)
        
        if self.save_logs:
            log_dir = "cgan_logs/"
            log_path = log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = tf.summary.create_file_writer(log_path)
        else:
            summary_writer = None

        # Checkpoints
        if self.checkpoints_frequency > 0:
            checkpoint_prefix = os.path.join(self.savecheckpoint_path, 'checkpoints/', 'epoch')
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             discriminator_optimizer=discriminator_optimizer,
                                             generator=self.generator, discriminator=self.discriminator)   

        # creating a single ndarray concatenating list of ndarray predictors along the last dimension 
        if self.predictors_train is not None:
            self.predictors_train = np.concatenate(self.predictors_train, axis=-1)
        else:
            self.predictors_train = None

        # shuffling the order of the available indices (n samples)
        if self.time_window is not None:
            self.n = self.data_train.shape[0] - self.time_window
        else:
            self.n = self.data_train.shape[0]
        self.indices_train = np.random.permutation(np.arange(self.n))

        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(self.n / self.batch_size)

        if isinstance(self.data_train, xr.DataArray):
            # self.time_metadata = self.data_train.time.copy()  # get time metadata
            self.data_train = self.data_train.values
        if isinstance(self.data_train_lr, xr.DataArray):
            self.data_train_lr = self.data_train_lr.values

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            pb_i = Progbar(self.steps_per_epoch, 
                           stateful_metrics=['gen_total_loss', 'gen_crosentr_loss', 
                                             'gen_mae_loss', 'disc_loss'])

            for i in range(self.steps_per_epoch):
                res = create_batch_hr_lr(
                    self.indices_train,
                    i,
                    self.data_train, 
                    self.data_train_lr,
                    upsampling=self.upsampling,
                    scale=self.scale, 
                    batch_size=self.batch_size, 
                    patch_size=self.patch_size,
                    time_window=self.time_window,
                    static_vars=self.static_vars, 
                    predictors=self.predictors_train,
                    interpolation=self.interpolation,
                    time_metadata=None)
               
                if self.static_vars is not None:
                    [lr_array, aux_hr], [hr_array] = res
                else:
                    [lr_array], [hr_array] = res

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
                    static_array=aux_hr)
                
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
            
            if self.checkpoints_frequency > 0:
                # Horovod: save checkpoints only on worker 0 to prevent other 
                # workers from corrupting it
                if self.running_on_first_worker:
                    if (epoch + 1) % self.checkpoints_frequency == 0:
                        checkpoint.save(file_prefix=checkpoint_prefix)
                        # saving the generator in tf format
                        self.generator.save(self.savecheckpoint_path + f'/checkpoints/save_epoch{epoch + 1}')
        
        # Horovod: save last checkpoint only on worker 0 to prevent other 
        # workers from corrupting it
        if self.checkpoints_frequency > 0 and self.running_on_first_worker:
            checkpoint.save(file_prefix=checkpoint_prefix)

        if self.save_loss_history and self.running_on_first_worker:
            losses_array = np.array((self.gentotal, self.gengan, self.gen_pxloss, self.disc))
            np.save(self.save_path + './losses.npy', losses_array)

        self.timing.checktime()

        ### Loss on the Test set
        if self.predictors_test is not None:
            self.predictors_test = np.concatenate(self.predictors_test, axis=-1)
        else:
            self.predictors_test = None

        if isinstance(self.data_test, xr.DataArray):
            # self.time_metadata_test = self.data_test.time.copy()  # time metadata
            self.data_test = self.data_test.values
        else:
            self.time_metadata_test = None
        if isinstance(self.data_test_lr, xr.DataArray):
            self.data_test_lr = self.data_test_lr.values

        # shuffling the order of the available indices (n samples)
        if self.time_window is not None:
            self.n_test = self.data_test.shape[0] - self.time_window
        else:
            self.n_test = self.data_test.shape[0]
        self.indices_test = np.random.permutation(np.arange(self.n_test))

        if self.running_on_first_worker:            
            res = create_batch_hr_lr(
                self.indices_test,
                0,
                self.data_test, 
                self.data_test_lr,
                upsampling=self.upsampling,
                scale=self.scale, 
                batch_size=self.n_test, 
                patch_size=self.patch_size,
                time_window=self.time_window,
                static_vars=self.static_vars, 
                predictors=self.predictors_test,
                interpolation=self.interpolation,
                time_metadata=None)
            
            if self.static_vars is not None:
                [lr_array, aux_hr], [hr_array] = res
                hr_arrtest = tf.cast(hr_array, tf.float32)
                lr_arrtest = tf.cast(lr_array, tf.float32)
                auxhr_arrtest = tf.cast(aux_hr, tf.float32)
                input_test = [lr_arrtest, auxhr_arrtest]
            else:
                [lr_array], [hr_array] = res
                lr_arrtest = tf.cast(lr_array, tf.float32)
                input_test = [lr_arrtest]
            
            y_test_pred = self.generator.predict(input_test)
            self.test_loss = self.lossf(hr_arrtest, y_test_pred)
            print(f'\n{self.lossf.__name__} on the test set: {self.test_loss}')
        
        self.timing.runtime()

        self.save_results(self.generator, folder_prefix='cgan_')


def load_checkpoint(
    checkpoint_dir, 
    checkpoint_number, 
    backbone,
    upsampling, 
    scale, 
    input_height_width, 
    n_static_vars=0, 
    n_predictors=0,
    time_window=None, 
    n_blocks=(20, 4), 
    n_filters=(8, 32), 
    attention=False,
    localcon_layer=False):
    """
    """
    n_channels = 1
    n_aux_channels = 0
    if n_static_vars > 0:
        n_channels += n_static_vars
        n_aux_channels += n_static_vars
    if n_predictors > 0:
        n_channels += n_predictors

    if time_window is not None and time_window > 1:
        model_is_spatiotemporal = True
    else:
        model_is_spatiotemporal = False

    # generator
    if upsampling in POSTUPSAMPLING_METHODS:
        if model_is_spatiotemporal:
            generator = recnet_postupsampling(
                backbone_block=backbone, upsampling=upsampling, scale=scale, 
                n_channels=n_channels, n_aux_channels=n_aux_channels, 
                n_filters=n_filters[0], n_blocks=n_blocks[0], lr_size=input_height_width,
                n_channels_out=1, time_window=time_window, attention=attention, 
                localcon_layer=localcon_layer)
        else:
            generator = net_postupsampling(
                backbone_block=backbone, upsampling=upsampling, scale=scale, 
                n_channels=n_channels, n_aux_channels=n_aux_channels, 
                n_filters=n_filters[0], n_blocks=n_blocks[0], lr_size=input_height_width,
                n_channels_out=1, attention=attention, localcon_layer=localcon_layer)
        
    elif upsampling == 'pin':
        if model_is_spatiotemporal:
            generator = recnet_pin(
                backbone_block=backbone, n_channels=n_channels, 
                n_aux_channels=n_aux_channels, hr_size=input_height_width,
                n_filters=n_filters[0], n_blocks=n_blocks[0], 
                n_channels_out=1, time_window=time_window, 
                attention=attention, localcon_layer=localcon_layer)
        else: 
            generator = net_pin(
                backbone_block=backbone, n_channels=n_channels, 
                n_aux_channels=n_aux_channels, hr_size=input_height_width,
                n_filters=n_filters[0], n_blocks=n_blocks[0], 
                n_channels_out=1, attention=attention, localcon_layer=localcon_layer)
        
    # discriminator
    discriminator = residual_discriminator(
        n_channels=n_channels, upsampling=upsampling, is_spatiotemporal=model_is_spatiotemporal, 
        scale=scale, lr_size=input_height_width, n_filters=n_filters[1], n_res_blocks=n_blocks[1],
        attention=attention)
    
    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)
    checkpoint.restore(checkpoint_path + '-' + str(checkpoint_number))
    return generator, generator_optimizer, discriminator, discriminator_optimizer


def generator_loss(disc_generated_output, gen_output, target, gen_pxloss_function, 
                   lambda_scaling_factor=100):
    """
    Generator loss:
    The generator loss is then calculated from the discriminator’s 
    classification – it gets rewarded if it successfully fools the discriminator, 
    and gets penalized otherwise. 

    * It is a sigmoid cross entropy loss of the discriminator output and an 
    array of ones.
    * The paper also includes L1 loss which is MAE (mean absolute error) between 
    the generated image and the target image. The px loss is controlled with the 
    ``gen_pxloss_function`` argument
    * This allows the generated image to become structurally similar to the 
    target image.
    * The formula to calculate the total generator loss is:

    loss = gan_loss + LAMBDA * px_loss,
    
    where LAMBDA = 100 was decided by the authors of the paper.
    """
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # binary crossentropy
    gan_loss = binary_crossentropy(tf.ones_like(disc_generated_output), 
                                   disc_generated_output)
    # px loss, regularization
    px_loss = gen_pxloss_function(target, gen_output)
    total_gen_loss = gan_loss + (lambda_scaling_factor * px_loss)
    return total_gen_loss, gan_loss, px_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Discriminator loss:
    * The discriminator loss function takes 2 inputs; real images, generated 
    images
    * real_loss is a sigmoid cross entropy loss of the real images and an array 
    of ones(since these are the real images)
    * generated_loss is a sigmoid cross entropy loss of the generated images and 
    an array of zeros(since these are the fake images)
    * Then the total_loss is the sum of real_loss and the generated_loss
    """
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)  
    real_loss = binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = binary_crossentropy(tf.zeros_like(disc_generated_output), 
                                         disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def train_step(lr_array, hr_array, generator, discriminator, generator_optimizer, 
               discriminator_optimizer, epoch, gen_pxloss_function, 
               summary_writer, first_batch, static_array=None):
    """
    Training:
    * For each example input generate an output.
    * The discriminator receives the input_image and the generated image as the 
    first input. The second input is the input_image and the target_image.
    * Next, we calculate the generator and the discriminator loss.
    * Then, we calculate the gradients of loss with respect to both the 
    generator and the discriminator variables(inputs) and apply those to the optimizer.
    """
    lr_array = tf.cast(lr_array, tf.float32)
    hr_array = tf.cast(hr_array, tf.float32)
    if static_array is not None:
        static_array = tf.cast(static_array, tf.float32)
        input_generator = [lr_array, static_array]
    else:
        input_generator = lr_array         

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # running the generator
        gen_array = generator(input_generator, training=True)
        # running the discriminator using both the reference and generated HR images
        disc_real_output = discriminator([lr_array, hr_array], training=True)
        disc_generated_output = discriminator([lr_array, gen_array], training=True)
        # computing the losses
        gen_total_loss, gen_gan_loss, gen_px_loss = generator_loss(disc_generated_output, 
                                                                   gen_array, 
                                                                   hr_array, 
                                                                   gen_pxloss_function)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    if has_horovod:
        # Horovod: add Horovod Distributed GradientTape.
        gen_tape = hvd.DistributedGradientTape(gen_tape)
        disc_tape = hvd.DistributedGradientTape(disc_tape)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    if summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_px_loss', gen_px_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
    if has_horovod:
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(generator.variables, root_rank=0)
            hvd.broadcast_variables(generator_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(discriminator.variables, root_rank=0)
            hvd.broadcast_variables(discriminator_optimizer.variables(), root_rank=0)

    return gen_total_loss, gen_gan_loss, gen_px_loss, disc_loss 