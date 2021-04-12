import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime


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


def set_visible_gpu(indices=(0)):
    physical_devices = list_devices('physical')
    tf.config.set_visible_devices(physical_devices[indices], 'GPU') 
    print(list_devices('logical'))


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


def crop_array(array, size, yx=None, position=False, get_copy=False):
    """
    Return a square cropped version of a 2D or 3D ndarray.
    
    Parameters
    ----------
    array : numpy ndarray
        Input image (2D ndarray) or cube (3D ndarray).
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
    # assuming 3D ndarray as multichannel image [lat, lon, vars]
    array_size_y = array.shape[0]
    array_size_x = array.shape[1] 
    
    if array.ndim not in [2, 3]:
        raise TypeError('Input array is not a 2D or 3D ndarray.')
    if not isinstance(size, int):
        raise TypeError('`Size` must be integer')
    if size > array_size_y or size > array_size_x: 
        msg = "`Size` larger than the input image size"
        raise ValueError(msg)

    if yx is not None and isinstance(yx, tuple):
        y, x = yx
    else:
        # random location
        y = np.random.randint(0, array_size_y - size - 1)
        x = np.random.randint(0, array_size_x - size - 1)

    y0, y1 = y, int(y + size)
    x0, x1 = x, int(x + size)

    if y0 < 0 or x0 < 0 or y1 > array_size_y or x1 > array_size_x:
        raise RuntimeError(f'Cropped image cannot be obtained with size={size}, y={y}, x={x}')

    if get_copy:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1].copy()
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :].copy()
    else:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1]
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :].copy()

    if position:
        return cropped_array, y, x
    else:
        return cropped_array


def resize_array(array, newsize, interpolation='bicubic'):
    """
    Return a resized version of a 2D or [x,y] 3D ndarray [x,y,channels], via
    interpolation.
    
    Parameters
    ----------
    array : numpy ndarray 
        Ndarray.
    newsize : tuple of int
        New size in X,Y.
    interpolation : 
        Interpolation mode.

    Returns
    -------
    resized_array : numpy ndarray
        Interpolated array with size ``newsize``.
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    size_x, size_y = newsize
    resized_array = cv2.resize(array, (size_x,size_y), interpolation=interp)
    return resized_array
