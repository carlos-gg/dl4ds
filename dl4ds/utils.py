import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime


BACKBONE_BLOCKS = ('convnet', 'resnet', 'densenet')
PREFIX_SAMPLE_TYPE = ('', 'rec')
NETS = [p + b for p in PREFIX_SAMPLE_TYPE for b in BACKBONE_BLOCKS]
UPSAMPLING_METHODS = ('spc',  # pixel shuffle or subpixel convolution in post-upscaling
                      'rc',  # resize convolution in post-upscaling
                      'dc',  # deconvolution or transposed convolution in post-upscaling
                      'pin')  # pre-upsampling via (bicubic) interpolation
POSTUPSAMPLING_METHODS = ('spc', 'rc', 'dc')
SPATIAL_MODELS = [p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
SPATIOTEMP_MODELS = ['rec' + p + '_' + u for p in BACKBONE_BLOCKS for u in UPSAMPLING_METHODS]
MODELS = [n + '_' + u for n in NETS for u in UPSAMPLING_METHODS]


def spatial_to_temporal_samples(array, time_window):
    """ """
    n_samples, y, x, n_channels = array.shape
    n_t_samples = n_samples - (time_window - 1)
    array_out = np.zeros((n_t_samples, time_window, y, x, n_channels))
    for i in range(n_t_samples):
        array_out[i] = array[i: i+time_window]
    return array_out


def checkarg_model(model, model_list=MODELS):
    """ """
    if not isinstance(model, str) or model not in model_list:
        msg = f'`model` not recognized. Must be one of the following: {model_list}'
        raise ValueError(msg)
    else:
        return model


def checkarg_backbone(backbone_block):
    """ """ 
    if not isinstance(backbone_block, str) or backbone_block not in BACKBONE_BLOCKS:
        msg = f'`backbone_block` not recognized. Must be one of the following: {BACKBONE_BLOCKS}'
        raise ValueError(msg)
    else:
        return backbone_block


def checkarg_upsampling(upsampling):
    """ """ 
    if not isinstance(upsampling, str) or upsampling not in UPSAMPLING_METHODS:
        msg = f'`upsampling` not recognized. Must be one of the following: {UPSAMPLING_METHODS}'
        raise ValueError(msg)
    else:
        return upsampling


def checkarg_dropout_variant(dropout):
    """ """
    if dropout is None:
        return dropout
    elif isinstance(dropout, str):
        if dropout not in ['spatial', 'gaussian']:
            msg = '`dropout_variant` must be either None or str (`gaussian` or `spatial`)'
            raise ValueError(msg)


def set_gpu_memory_growth(verbose=True):
    physical_devices = list_devices(verbose=verbose) 
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(physical_devices)


def list_devices(which='physical', gpu=True, verbose=True):
    if gpu:
        dev = 'GPU'
    else:
        dev = 'CPU'
    if which == 'physical':
        devices = tf.config.list_physical_devices(dev)
    elif which == 'logical':
        devices = tf.config.list_logical_devices(dev)
    if verbose:
        print(devices)
    return devices


def set_visible_gpus(*gpu_indices):
    gpus = list_devices('physical', gpu=True)
    wanted_gpus = [gpus[i] for i in gpu_indices]
    tf.config.set_visible_devices(wanted_gpus, 'GPU') 
    print(list_devices('logical'))


def rank(x):
    return len(x.get_shape().as_list())


class Timing():
    """ 
    """
    sep = '-' * 80

    def __init__(self, verbose=True):
        """ 
        Timing utility class.

        Parameters
        ----------
        verbose : bool
            Verbosity.

        """
        self.verbose = verbose
        self.running_time = None
        self.checktimes = list()
        self.starting_time = datetime.now()
        self.starting_time_fmt = self.starting_time.strftime("%Y-%m-%d %H:%M:%S")
        if self.verbose:
            print(self.sep)
            print(f"Starting time: {self.starting_time_fmt}")
            print(self.sep)
        
    def runtime(self):
        """ 
        """
        self.running_time = str(datetime.now() - self.starting_time)
        if self.verbose:
            print(self.sep)
            print(f"Final running time: {self.running_time}")
            print(self.sep)
    
    def checktime(self):
        """
        """
        checktime = str(datetime.now() - self.starting_time)
        self.checktimes.append(checktime)
        if self.verbose:
            print(self.sep)
            print(f"Timing: {checktime}")
            print(self.sep)


def crop_array(array, size, yx=None, position=False, exclude_borders=False, 
               get_copy=False):
    """
    Return a square cropped version of a 2D, 3D or 4D ndarray.
    
    Parameters
    ----------
    array : numpy ndarray
        Input image (2D ndarray) or cube (3D or 4D ndarray).
    size : int
        Size of the cropped image.
    yx : tuple of int or None, optional
        Y,X coordinate of the bottom-left corner. If None then a random
        position will be chosen.
    position : bool, optional
        If set to True return also the coordinates of the bottom-left corner.
    get_copy : bool, optional
        If True a cropped copy of the intial array is returned. By default a
        sliced view of the array is returned.
    
    Returns
    -------
    cropped_array : numpy ndarray
        Cropped ndarray. By default a view is returned, unless ``get_copy``
        is True. 
    y, x : int
        [position=True] Y,X coordinates.
    """    
    if array.ndim not in [2, 3, 4]:
        raise TypeError('Input array is not a 2D, 3D, or 4D ndarray')
    if not isinstance(size, int):
        raise TypeError('`Size` must be integer')
    if array.ndim in [2, 3]: 
        # assuming 3D ndarray as multichannel grid [lat,lon,vars] or [y,x,channels]
        array_size_y = array.shape[0]
        array_size_x = array.shape[1] 
    elif array.ndim == 4:
        # assuming 4D ndarray [aux dim, lat, lon, vars] or [time,y,x,channels]
        array_size_y = array.shape[1]
        array_size_x = array.shape[2]
    if size > array_size_y or size > array_size_x: 
        msg = "`Size` larger than the input image size"
        raise ValueError(msg)

    if yx is not None and isinstance(yx, tuple):
        y, x = yx
    else:
        # random location
        if exclude_borders:
            y = np.random.randint(1, array_size_y - size - 1)
            x = np.random.randint(1, array_size_x - size - 1)
        else:
            y = np.random.randint(0, array_size_y - size)
            x = np.random.randint(0, array_size_x - size)

    y0, y1 = y, int(y + size)
    x0, x1 = x, int(x + size)

    if y0 < 0 or x0 < 0 or y1 > array_size_y or x1 > array_size_x:
        raise RuntimeError(f'Cropped image cannot be obtained with size={size}, y={y}, x={x}')

    if get_copy:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1].copy()
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :].copy()
        elif array.ndim == 4:
            cropped_array = array[:, y0: y1, x0: x1, :].copy()
    else:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1]
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :]
        elif array.ndim == 4:
            cropped_array = array[:, y0: y1, x0: x1, :]

    if position:
        return cropped_array, y, x
    else:
        return cropped_array


def resize_array(array, newsize, interpolation='bicubic', squeezed=True):
    """
    Return a resized version of a 2D or [y,x] 3D ndarray [y,x,channels] or
    4D ndarray [time,y,x,channels] via interpolation.
    
    Parameters
    ----------
    array : numpy ndarray 
        Input ndarray.
    newsize : tuple of int
        New size in X,Y.
    interpolation : str, optional
        Interpolation mode.
    squeezed : bool, optional
        If True, the output will be squeezed (any dimension with lenght 1 will
        be removed).

    Returns
    -------
    resized_arr : numpy ndarray
        Interpolated array with size ``newsize``.
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    size_x, size_y = newsize
    
    if array.ndim in [2, 3]:
        resized_arr = cv2.resize(array, (size_x, size_y), interpolation=interp)
    elif array.ndim == 4:
        n = array.shape[0]
        n_ch = array.shape[-1]
        resized_arr = np.zeros((n, size_y, size_x, n_ch))
        for i in range(n):
            ti = cv2.resize(array[i], (size_x, size_y), interpolation=interp)
            if n_ch == 1:
                ti = np.expand_dims(ti, -1)
            resized_arr[i] = ti

    if squeezed:
        resized_arr = np.squeeze(resized_arr)
    return resized_arr
