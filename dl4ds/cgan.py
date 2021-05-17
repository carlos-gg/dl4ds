"""
Training procedure for Conditional GAN models. Follows Isola et al. 2016
"""

import os 
import datetime
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import horovod.tensorflow as hvd
import numpy as np

from .resnet_int import resnet_int
from .resnet_rec import resnet_rec
from .resnet_spc import resnet_spc
from .discriminator import residual_discriminator
from .dataloader import create_pair_hr_lr
from .utils import Timing, list_devices, set_gpu_memory_growth, set_visible_gpus


def load_checkpoint(checkpoint_dir, checkpoint_number, scale, model='resnet_spc', 
                    topography=None, landocean=None, 
                    n_res_blocks=(20, 4), n_filters=64, attention=False):
    """
    """
    n_channels = 1
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1

    # generator
    if model not in ['resnet_spc', 'resnet_int', 'resnet_rec']:
        raise ValueError('`model` not recognized. Must be one of the following: resnet_spc, resnet_int, resnet_rec')

    if model == 'resnet_spc':
        generator = resnet_spc(scale=scale, n_channels=n_channels, n_filters=n_filters, 
                               n_res_blocks=n_res_blocks[0], n_channels_out=1, attention=attention)
    elif model == 'resnet_rec':
        generator = resnet_rec(scale=scale, n_channels=n_channels, n_filters=n_filters, 
                               n_res_blocks=n_res_blocks[0], n_channels_out=1, attention=attention)
    elif model == 'resnet_int':
        generator = resnet_int(n_channels=n_channels, n_filters=n_filters, 
                               n_res_blocks=n_res_blocks[0], n_channels_out=1, attention=attention)
        
    # discriminator
    discriminator = residual_discriminator(n_channels=n_channels, n_filters=n_filters, scale=scale,
                                           n_res_blocks=n_res_blocks[1], model=model, attention=attention)
    
    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)
    checkpoint.restore(checkpoint_path + '-' + str(checkpoint_number))
    return generator, generator_optimizer, discriminator, discriminator_optimizer


def generator_loss(disc_generated_output, gen_output, target, lambda_scaling_factor=100):
    """
    Generator loss:
    The generator loss is then calculated from the discriminator’s 
    classification – it gets rewarded if it successfully fools the discriminator, 
    and gets penalized otherwise. 

    * It is a sigmoid cross entropy loss of the discriminator output and an 
    array of ones.
    * The paper also includes L1 loss which is MAE (mean absolute error) between 
    the generated image and the target image.
    * This allows the generated image to become structurally similar to the 
    target image.
    * The formula to calculate the total generator 
    loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was 
    decided by the authors of the paper.
    """
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # binary crossentropy
    gan_loss = binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error, regularization
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_scaling_factor * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


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
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
    real_loss = binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def training_cgan(
    model, 
    x_train,
    x_test, 
    epochs, 
    steps_per_epoch=None,
    scale=5, 
    patch_size=50, 
    batch_size=16,
    interpolation='bicubic', 
    topography=None, 
    landocean=None, 
    checkpoints_frequency=5, 
    savecheckpoint_path='./checkpoints/',
    n_res_blocks=(20, 4), 
    n_filters=64, 
    attention=False,
    device='GPU',
    gpu_memory_growth=True,
    verbose=True):
    """
    
    Parameters
    ----------
    model : str
        String with the name of the model architecture, either 'resnet_spc', 
        'resnet_int' or 'resnet_rec'. Used as a the CGAN generator.
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
    if not isinstance(model, str) and model not in ['resnet_spc', 'resnet_int', 'resnet_rec']:          
        raise ValueError('`model` not recognized. Must be one of the following: resnet_spc, resnet_int, resnet_rec')

    timing = Timing()

    # Initialize Horovod
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

    gentotal = []
    gengan = []
    genl1 = []
    disc = []
    
    log_dir = "cgan_logs/"
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    n_channels = 1
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1

    # generator
    if model == 'resnet_spc':
        generator = resnet_spc(scale=scale, n_channels=n_channels, 
                               n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                               n_channels_out=1, attention=attention)
    elif model == 'resnet_int':
        generator = resnet_int(n_channels=n_channels, n_filters=n_filters, 
                               n_res_blocks=n_res_blocks[0], n_channels_out=1, 
                               attention=attention)
    elif model == 'resnet_rec':
        generator = resnet_rec(scale=scale, n_channels=n_channels, 
                               n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                               n_channels_out=1, attention=attention)
        
    # discriminator
    discriminator = residual_discriminator(n_channels=n_channels, 
                                           n_filters=n_filters, scale=scale,
                                           n_res_blocks=n_res_blocks[1], 
                                           model=model, attention=attention)
    
    if verbose == 1 and running_on_first_worker:
        generator.summary(line_length=150)
        discriminator.summary(line_length=150)

    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # checkpoint
    if savecheckpoint_path is not None and savecheckpoint_path is not None:
        checkpoint_prefix = os.path.join(savecheckpoint_path, 'checkpoint_epoch')
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator, discriminator=discriminator)
    
    if patch_size % scale != 0:
        raise ValueError('`patch_size` must be divisible by `scale` (remainder must be zero)')

    n_samples = x_train.shape[0]
    if steps_per_epoch is None:
        steps_per_epoch = int(n_samples / batch_size)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        pb_i = Progbar(steps_per_epoch, stateful_metrics=['gen_total_loss', 
                                                          'gen_crosentr_loss', 
                                                          'gen_mae_loss', 
                                                          'disc_loss'])

        for i in range(steps_per_epoch):
            hr_array, lr_array = create_batch_hr_lr(
                x_train,
                batch_size=global_batch_size,
                tuple_predictors=None, 
                scale=scale, 
                topography=topography, 
                landocean=landocean, 
                patch_size=patch_size, 
                model=model, 
                interpolation=interpolation)

            losses = train_step(
                lr_array, 
                hr_array, 
                generator, 
                discriminator, 
                generator_optimizer, 
                discriminator_optimizer, 
                epoch, 
                summary_writer, 
                first_batch=True if epoch==0 and i==0 else False)
            
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = losses
            lossvals = [('gen_total_loss', gen_total_loss), 
                        ('gen_crosentr_loss', gen_gan_loss), 
                        ('gen_mae_loss', gen_l1_loss), 
                        ('disc_loss', disc_loss)]
            
            if running_on_first_worker:
                pb_i.add(1, values=lossvals)
        
        gentotal.append(gen_total_loss)
        gengan.append(gen_gan_loss)
        genl1.append(gen_l1_loss)
        disc.append(disc_loss)
        
        if savecheckpoint_path is not None and savecheckpoint_path is not None:
            # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting it
            if running_on_first_worker:
                if (epoch + 1) % checkpoints_frequency == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)
    
    # Horovod: save last checkpoint only on worker 0 to prevent other workers from corrupting it
    if checkpoints_frequency is not None and running_on_first_worker:
        checkpoint.save(file_prefix=checkpoint_prefix)

    losses_array = np.array((gentotal, gengan, genl1, disc))
    np.save('./losses.npy', losses_array)

    timing.checktime()

    # MAE on Test set
    if running_on_first_worker:
        test_steps = int(x_test.shape[0] / batch_size)
        
        hr_arrtest, lr_arrtest = create_batch_hr_lr(
            x_train,
            batch_size=test_steps,
            tuple_predictors=None, 
            scale=scale, 
            topography=topography, 
            landocean=landocean, 
            patch_size=patch_size, 
            model=model, 
            interpolation=interpolation,
            shuffle=False)

        y_test_pred = generator.predict(lr_arrtest)
        maes = tf.keras.metrics.mean_absolute_error(hr_arrtest, y_test_pred)
        maes_pairs = np.mean(maes, axis=(1,2))
        mean_mae = np.mean(maes_pairs)
        print(f'\nMAE on the test set: {mean_mae}')
    
    timing.runtime()

    return generator


def create_batch_hr_lr(x_train, batch_size, tuple_predictors, scale, topography, 
                       landocean, patch_size, model, interpolation, shuffle=True):
    """
    """
    if shuffle:
        indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
    else:
        indices = np.arange(x_train.shape[0])
    batch_hr_images = []
    batch_lr_images = []

    for i in indices:
        hr_array, lr_array = create_pair_hr_lr(x_train[i],
                    tuple_predictors=tuple_predictors, 
                    scale=scale, 
                    topography=topography, 
                    landocean=landocean, 
                    patch_size=patch_size, 
                    model=model, 
                    interpolation=interpolation)
        batch_lr_images.append(lr_array)
        batch_hr_images.append(hr_array)

    batch_lr_images = np.asarray(batch_lr_images)
    batch_hr_images = np.asarray(batch_hr_images) 
    return batch_hr_images, batch_lr_images



@tf.function
def train_step(lr_array, hr_array, generator, discriminator, generator_optimizer, 
               discriminator_optimizer, epoch, summary_writer, first_batch):
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
    hr_array =  tf.cast(hr_array, tf.float32)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # running the generator
        gen_array = generator(lr_array, training=True)
        # running the discriminator using both the reference and generated HR images
        disc_real_output = discriminator([lr_array, hr_array], training=True)
        disc_generated_output = discriminator([lr_array, gen_array], training=True)
        # computing the losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_array, hr_array)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Horovod: add Horovod Distributed GradientTape.
    gen_tape = hvd.DistributedGradientTape(gen_tape)
    disc_tape = hvd.DistributedGradientTape(disc_tape)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
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

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss 