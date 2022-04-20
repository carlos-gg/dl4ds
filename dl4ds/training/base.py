"""
Base training class
"""

import os
import xarray as xr
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from matplotlib.pyplot import show, close
import logging
tf.get_logger().setLevel(logging.ERROR)

try:
    import horovod.tensorflow.keras as hvd
    has_horovod = True
except ImportError:
    has_horovod = False

from ..utils import (list_devices, set_gpu_memory_growth, plot_history,
                     set_visible_gpus, check_compatibility_upsbackb)
from ..losses import (mae, mse, dssim, dssim_mae, dssim_mae_mse, dssim_mse,
                      msdssim, msdssim_mae)


class Trainer(ABC):
    """        
    """
    def __init__(
        self,
        backbone,
        upsampling, 
        data_train,
        data_train_lr=None,
        time_window=None,
        use_season=True,
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
        savecheckpoint_path=None,
        show_plot=False,
        ):
        """
        """
        self.use_season = use_season
        if self.use_season:
            if not hasattr(data_train, 'time'):
                raise TypeError('input data must be a xr.DataArray and have' 
                                'time metadata when use_season=True')
        
        # checking training data split (both hr and lr)
        self.data_train = data_train
        if not isinstance(self.data_train, (xr.DataArray, np.ndarray)):
            msg = '`data_train` must be a np.ndarray or xr.DataArray object'
            raise TypeError(msg)
        if not self.data_train.ndim > 3:
            msg = '`data_train` must be at least 4D [samples, lat, lon, variables]'
            raise ValueError(msg)
        self.data_train_lr = data_train_lr
        if self.data_train_lr is not None:
            if not isinstance(self.data_train_lr, (xr.DataArray, np.ndarray)):
                msg = '`data_train_lr` must be a np.ndarray or xr.DataArray object'
                raise TypeError(msg)
            if self.data_train_lr.shape[0] != self.data_train.shape[0]:
                msg = '`data_train_lr` and `data_train` must contain '
                msg += 'the same number of samples (equal 1st dim lenght)'
                raise ValueError(msg)
            if not self.data_train_lr.ndim > 3:
                msg = '`data_train_lr` must be at least 4D [samples, lat, lon, variables]'
                raise ValueError(msg)

        self.backbone, self.upsampling = check_compatibility_upsbackb(backbone,
                                                                      upsampling, 
                                                                      time_window)
        self.time_window = time_window
        if self.time_window is not None and self.time_window > 1:
            self.model_is_spatiotemporal = True
        else:
            self.model_is_spatiotemporal = False
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
        if self.save_path is None:
            self.save_path = './'
        else:
            if not self.save_path.endswith('/'):
                self.save_path += '/'
        self.savecheckpoint_path = savecheckpoint_path
        if self.savecheckpoint_path is None and self.save:
            self.savecheckpoint_path = self.save_path
        self.show_plot = show_plot
       
        if has_horovod:
            ### Initializing Horovod
            hvd.init()

        ### Setting up devices
        if self.device == 'GPU':
            if self.gpu_memory_growth:
                set_gpu_memory_growth()
            if has_horovod:
                # pin GPU to be used to process local rank (one GPU per process)       
                set_visible_gpus(hvd.local_rank())
            devices = list_devices('physical', gpu=True, verbose=verbose) 
        elif device == 'CPU':
            devices = list_devices('physical', gpu=False, verbose=verbose)
        else:
            raise ValueError('device not recognized')

        n_devices = len(devices)            
        batch_size_per_replica = self.batch_size
        self.global_batch_size = batch_size_per_replica * n_devices
        if self.verbose in [1 ,2]:
            print ('Number of devices: {}'.format(n_devices))
            if n_devices > 1:
                print(f'Global batch size: {self.global_batch_size}, per replica: {batch_size_per_replica}')
            else:
                print(f'Global batch size: {self.global_batch_size}')

        # distributed training with GPUs first Horovod worker
        cond1 = self.device == 'GPU' and has_horovod and hvd.rank() == 0
        # single GPU training without horovod
        cond2 = self.device == 'GPU' and not has_horovod
        # CPU training
        cond3 = self.device == 'CPU'
        if cond1 or cond2 or cond3:
            self.running_on_first_worker = True
        else:
            self.running_on_first_worker = False
        
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
        elif loss == 'msdssim':
            self.lossf = msdssim
        elif loss == 'msdssim_mae':
            self.lossf = msdssim_mae
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
                self.model_save_path = self.save_path + folder_prefix + self.backbone + '_' + self.upsampling + '/'
            else:
                self.model_save_path = self.save_path + self.backbone + '_' + self.upsampling + '/'

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

