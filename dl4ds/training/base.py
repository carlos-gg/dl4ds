"""
Base training class
"""

import os
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from matplotlib.pyplot import show, close
import horovod.tensorflow.keras as hvd
import logging
tf.get_logger().setLevel(logging.ERROR)

from .. import MODELS
from ..utils import (list_devices, set_gpu_memory_growth, plot_history,
                    set_visible_gpus, checkarg_model)
from ..losses import (mae, mse, dssim, dssim_mae, dssim_mae_mse, dssim_mse,
                     msdssim, msdssim_mae)


class Trainer(ABC):
    """        
    """
    def __init__(
        self,
        model_name, 
        data_train,
        data_train_lr=None,
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
        self.model_name = model_name
        self.use_season = use_season
        if self.use_season:
            if not hasattr(data_train, 'time'):
                raise TypeError('input data must be a xr.DataArray and have' 
                                'time metadata when use_season=True')
        
        self.data_train = data_train
        self.data_train_lr = data_train_lr
        if self.data_train_lr is not None:
            if self.data_train_lr.shape[0] != self.data_train.shape[0]:
                msg = '`data_train_lr` and `data_train` must contain '
                msg += 'the same number of samples (equal 1st dim lenght)'
                raise ValueError(msg)
        
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

