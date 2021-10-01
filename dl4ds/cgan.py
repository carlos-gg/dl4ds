"""
Training procedure for Conditional GAN models. Follows Isola et al. 2016
"""

import os 
import tensorflow as tf
import horovod.tensorflow as hvd

from .models import (net_pin, recnet_pin, net_postupsampling, 
                     recnet_postupsampling)
from .discriminator import residual_discriminator
from .utils import checkarg_model
from . import POSTUPSAMPLING_METHODS, SPATIAL_MODELS, SPATIOTEMP_MODELS


def load_checkpoint(checkpoint_dir, checkpoint_number, scale, model='resnet_spc', 
                    topography=None, landocean=None, time_window=None,
                    n_res_blocks=(20, 4), n_filters=64, attention=False):
    """
    """
    n_channels = 1
    if topography is not None:
        n_channels += 1
    if landocean is not None:
        n_channels += 1

    # generator
    model = checkarg_model(model)
    upsampling = model.split('_')[-1]
    backbone = model.split('_')[0]
    if backbone.startswith('rec'):
        backbone = backbone[3:]

    if upsampling in POSTUPSAMPLING_METHODS:
        if model in SPATIAL_MODELS:
            generator = model = net_postupsampling(
                backbone_block=backbone, upsampling=upsampling, scale=scale, 
                n_channels=n_channels, n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                n_channels_out=1, attention=attention)
        elif model in SPATIOTEMP_MODELS:
            generator = model = recnet_postupsampling(
                backbone_block=backbone, upsampling=upsampling, scale=scale, 
                n_channels=n_channels, n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                n_channels_out=1, time_window=time_window, attention=attention)
    elif upsampling == 'pin':
        if model in SPATIAL_MODELS:
            generator = net_pin(backbone_block=backbone, n_channels=n_channels, 
                                n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                                n_channels_out=1, attention=attention)
        elif model in SPATIOTEMP_MODELS:
            generator = recnet_pin(backbone_block=backbone, n_channels=n_channels, 
                                   n_filters=n_filters, n_res_blocks=n_res_blocks[0], 
                                   n_channels_out=1, time_window=time_window, 
                                   attention=attention)
        
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
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
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
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
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