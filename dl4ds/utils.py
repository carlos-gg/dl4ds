import numpy as np
import tensorflow as tf
import xarray as xr
import cv2
from datetime import datetime

import matplotlib.pyplot as plt
from typing import List, Dict, Type, Union, Tuple, Callable
import os
import math
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensorflow.keras.callbacks import History

from . import BACKBONE_BLOCKS, DROPOUT_VARIANTS, LOSS_FUNCTIONS, UPSAMPLING_METHODS, INTERPOLATION_METHODS
from . import losses


def spatial_to_spatiotemporal_samples(array, time_window):
    """Add one dimension to spatial array [n_samples or time, lat, lon, vars] 
    in order to have [n_samples, time_window, lat, lon, vars].
    """
    n_samples, y, x, n_channels = array.shape
    n_t_samples = n_samples - (time_window - 1)
    array_out = np.zeros((n_t_samples, time_window, y, x, n_channels))
    for i in range(n_t_samples):
        array_out[i] = array[i: i+time_window]
    return array_out


def spatiotemporal_to_spatial_samples(array, time_window):
    """Remove dimension `time_window` from `array`, resulting in a sequence of
    spatial samples/grids. `time_window` is a dimension assumed to be in the 
    second place.

    ###TO-DO : other ways to collapse the time_window dimension
    """
    _, timew, _, _, _ = array.shape
    if timew != time_window:
        raise ValueError(
            '`time_window` must be located in the second position [n_samples, time_window, lat, lon, vars]')
    array_out = array[:, 0, :, :, :]
    array_out = np.concatenate([array_out, array[-1, 1:, :, :, :]], axis=0)
    return array_out


def checkarray_ndim(array, ndim=3, add_axis_position=-1):
    """Check the np.ndarray has at least `ndim` dimensions. If needed a new
    dimension (of lenght 1) is added at the position given by `add_axis_position`.
    """
    if array.ndim < ndim:
        return np.expand_dims(array, axis=add_axis_position)
    else:
        return array


def check_compatibility_upsbackb(backbone, upsampling, time_window):
    """Check that the upsampling and backbone arguments are compatible. 

    Parameters
    ----------
    backbone : str
        Backbone block.
    upsampling : str
        Upsampling method. 
    time_window : int
        Time window for spatio-temporal samples.
    """
    upsampling = checkarg_upsampling(upsampling)
    backbone = checkarg_backbone(backbone)
    # encoder decoder (unet) backbone only with pre-upsampling
    if backbone == 'unet' and upsampling != 'pin':
        raise ValueError('`unet` backbone only works with `pin` pre-upsampling')
    # unet and convnext work only with spatial samples
    if backbone in ['convnext', 'unet'] and time_window is not None:
        msg = '`unet` and `convnext` backbones only work with spatial samples '
        msg += '(`time_window` must be None)'
        raise ValueError(msg)
    return backbone, upsampling


def checkarg_upsampling(upsampling):
    """Check the argument ``upsampling``.

    Parameters
    ----------
    upsampling : str
        Upsampling method. 
    """ 
    if not isinstance(upsampling, str):
        raise TypeError('`upsampling` must be a string')

    if upsampling not in UPSAMPLING_METHODS:
        msg = f'`upsampling` not recognized. Must be one of the '
        msg += f'following: {UPSAMPLING_METHODS}. Got {upsampling}'
        raise ValueError(msg)
    else:
        return upsampling


def checkarg_backbone(backbone):
    """Check the argument ``backbone``.

    Parameters
    ----------
    backbone : str
        Backbone block. 
    """ 
    if not isinstance(backbone, str):
        raise TypeError('`backbone` must be a string')

    if backbone not in BACKBONE_BLOCKS:
        msg = f'`backbone` not recognized. Must be one of the '
        msg += f'following: {BACKBONE_BLOCKS}. Got {backbone}'
        raise ValueError(msg)
    else:
        return backbone


def checkarg_dropout_variant(dropout_variant):
    """Check the argument ``dropout_variant``.

    Parameters
    ----------
    dropout_variant : str
        Desired dropout variant.  
    """
    if dropout_variant is None or dropout_variant == 'vanilla':
        return dropout_variant
    elif isinstance(dropout_variant, str):
        if dropout_variant not in DROPOUT_VARIANTS:
            msg = f"`dropout_variant` must be None or one of {DROPOUT_VARIANTS}, got {dropout_variant}"
            raise ValueError(msg)
        else:
            return dropout_variant


def checkarg_loss(loss):
    """Check the argument ``loss``.

    Parameters
    ----------
    loss : str
        Loss/cost function.  
    """
    if isinstance(loss, str):
        if loss not in LOSS_FUNCTIONS:
            msg = f"`loss` must be one of {LOSS_FUNCTIONS}, got {loss}"
            raise ValueError(msg)
        else:
            if loss == 'mae':  
                return losses.mae
            elif loss == 'mse':  
                return losses.mse
            elif loss == 'dssim':
                return losses.dssim
            elif loss == 'dssim_mae':
                return losses.dssim_mae
            elif loss == 'dssim_mse':
                return losses.dssim_mse
            elif loss == 'dssim_mae_mse':
                return losses.dssim_mae_mse
            elif loss == 'msdssim':
                return losses.msdssim
            elif loss == 'msdssim_mae':
                return losses.msdssim_mae
            elif loss == 'msdssim_mae_mse':
                return losses.msdssim_mae_mse
    else:
        raise TypeError('`loss` must be a string, one of {LOSS_FUNCTIONS}')


def set_gpu_memory_growth():
    physical_devices = list_devices(verbose=False) 
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


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
        print('List of devices:')
        print(devices)
    return devices


def set_visible_gpus(*gpu_indices):
    gpus = list_devices('physical', gpu=True, verbose=False)
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
    Return a square cropped version of a 2D, 3D or 4D or 5D ndarray.
    
    Parameters
    ----------
    array : numpy ndarray
        Input image (2D ndarray) or cube (3D, 4D or 5D ndarray).
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
    if array.ndim not in [2, 3, 4, 5]:
        raise TypeError('Input array is not a 2D, 3D, or 4D ndarray')
    if not isinstance(size, int):
        raise TypeError('`Size` must be integer')
    if array.ndim in [2, 3]: 
        # assuming 3D ndarray as multichannel grid [lat, lon, vars] or [y, x, channels]
        array_size_y = array.shape[0]
        array_size_x = array.shape[1] 
    elif array.ndim == 4:
        # assuming 4D ndarray [aux dim, lat, lon, vars] or [time, y, x, channels]
        array_size_y = array.shape[1]
        array_size_x = array.shape[2]
    elif array.ndim == 5:
        # assuming 4D ndarray [aux dim, time, y, x, channels]
        array_size_y = array.shape[2]
        array_size_x = array.shape[3]
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
        elif array.ndim == 5:
            cropped_array = array[:, :, y0: y1, x0: x1, :].copy()
    else:
        if array.ndim == 2:
            cropped_array = array[y0: y1, x0: x1]
        elif array.ndim == 3:
            cropped_array = array[y0: y1, x0: x1, :]
        elif array.ndim == 4:
            cropped_array = array[:, y0: y1, x0: x1, :]
        elif array.ndim == 5:
            cropped_array = array[:, :, y0: y1, x0: x1, :]

    if position:
        return cropped_array, y, x
    else:
        return cropped_array


def resize_array(array, newsize, interpolation='inter_area', squeezed=True, 
                 keep_dynamic_range=False):
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
    if interpolation not in INTERPOLATION_METHODS:
        raise ValueError(f'`interpolation` must be one of {INTERPOLATION_METHODS}. Received {interpolation}')
    if array.dtype in ['bool', 'int', 'int64']:
        array = array.astype('int')
        interpolation = 'nearest'  # only nearest is supported in opencv for int

    if interpolation == 'nearest':
        intmethod = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        intmethod = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        intmethod = cv2.INTER_LINEAR
    elif interpolation == 'inter_area':
        intmethod = cv2.INTER_AREA
    elif interpolation == 'lanczos':
        intmethod = cv2.INTER_LANCZOS4

    size_x, size_y = newsize
    
    if array.ndim in [2, 3]:
        resized_arr = cv2.resize(array, (size_x, size_y), interpolation=intmethod)
        if resized_arr.ndim == 2 and array.ndim == 3:
            resized_arr = np.expand_dims(resized_arr, -1)
    elif array.ndim == 4:
        n = array.shape[0]
        n_ch = array.shape[-1]
        resized_arr = np.zeros((n, size_y, size_x, n_ch))
        for i in range(n):
            ti = cv2.resize(array[i], (size_x, size_y), interpolation=intmethod)                
            resized_arr[i] = np.expand_dims(ti, -1) if n_ch ==1 else ti
    else:
        raise RuntimeError(f'Wrong dimensions, got {array.ndim}')

    if squeezed:
        resized_arr = np.squeeze(resized_arr)
    if keep_dynamic_range:
        resized_arr = np.clip(resized_arr, a_min=array.min(), a_max=array.max())
    return resized_arr


" -----------------------------------------------------------------------------"
" -----------------------------------------------------------------------------"
"""Methods for plotting a keras model training history. Adapted from 
https://github.com/LucaCappelletti94/plot_keras_history """

def plot_history(
    histories: Union[History, List[History], Dict[str, List[float]], pd.DataFrame, List[pd.DataFrame], str, List[str]],
    style: str = "-",
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Callable = None,
    path: str = None,
    single_graphs: bool = False,
    max_epochs: Union[int, str] = "max",
    monitor: str = None,
    monitor_mode: str = "max",
    log_scale_metrics: bool = False,
    title: str = None,
) -> Tuple[Union[Figure, List[Figure]], Union[Axes, List[Axes]]]:
    """Plot given training histories.

    Parameters
    ----------------------------
    histories,
        the histories to plot.
        This parameter can either be a single or multiple dataframes
        or one or more paths to the stored CSVs or JSON of the history.
    style:str="-",
        the style to use when plotting the graphs.
    side:int=5,
        the side of every sub-graph.
    graphs_per_row:int=4,
        number of graphs per row.
    customization_callback:Callable=None,
        callback for customising axis.
    path:str=None,
        where to save the graphs, by defalut nowhere.
    single_graphs:bool=False,
        whetever to create the graphs one by one.
    max_epochs: Union[int, str] = "max",
        Number of epochs to plot. Can either be "max", "min" or a positive integer value.
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    title: str = None,
        Title to put on top of the subplots.

    Raises
    --------------------------
    ValueError,
        If monitor_mode is not either "min" or "max".
    ValueError,
        If max_epochs is not either "min", "max" or a numeric integer.
    """
    # Some parameters validation
    if monitor_mode not in ("min", "max"):
        raise ValueError("Given monitor mode '{}' is not supported.".format(monitor_mode))
    if max_epochs not in ("min", "max") and not isinstance(max_epochs, int):
        raise ValueError("Given parameter max_epochs '{}' is not supported.".format(max_epochs))
    # If the histories are not provided as a list, we normalized it
    # to a list.
    if not isinstance(histories, list):
        histories = [histories]
    # If the path is not None, we prepare the directory where to
    # store the created image(s).
    if path is not None:
        directory_name = os.path.dirname(path)
        # The directory name may be an empty string.
        if directory_name:
            os.makedirs(directory_name, exist_ok=True)

    # Normalize the training histories.
    histories = [_to_dataframe(history)._get_numeric_data() for history in histories]

    # Filter out the epochs as required.
    if max_epochs in ("max", "min"):
        epochs = [len(history) for history in histories]
        if max_epochs == "max":
            max_epochs = max(epochs)

        if max_epochs == "min":
            max_epochs = min(epochs)

    histories = [history[:max_epochs] for history in histories]
    average_history = pd.concat(histories)
    average_history = average_history.groupby(average_history.index).mean()

    if single_graphs:
        return list(zip(*[
            _plot_history(
                [history[columns] for history in histories],
                average_history,
                style,
                side,
                graphs_per_row,
                customization_callback,
                "{path}/{c}.png".format(path=path, c=columns[0]),
                log_scale_metrics,
                title=title,
            )
            for columns in _get_column_tuples(histories[0])
        ]))
    else:
        return _plot_history(histories, average_history, style, side,
                             graphs_per_row, customization_callback, path,
                             log_scale_metrics, title=title)


def _plot_history(
    histories: pd.DataFrame,
    average_history: pd.DataFrame = None,
    style: str = "-",
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Callable = None,
    path: str = None,
    max_epochs: int = None,
    log_scale_metrics: bool = False,
    monitor: str = None,
    best_point_x: int = None,
    title: str = None,
) -> Tuple[Figure, Axes]:
    """Plot given training histories.

    Parameters
    -------------------------------
    histories: pd.DataFrame,
        The histories to plot.
    average_history: pd.DataFrame = None,
        Average histories, if multiple histories were given.
    style: str = "-",
        The style to use when plotting the graphs.
    side: int=5,
        The side of every sub-graph.
    graphs_per_row: int = 4,
        Number of graphs per row.
    customization_callback: Callable = None,
        Callback for customising axis.
    path:str = None,
        Where to save the graphs, by defalut nowhere.
    monitor: str = None,
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    best_point_x: int = None,
        Point to be highlighted as best.
    title: str = None,
        Title to put on top of the subplots.
    """
    x_label = "Epochs" if histories[0].index.name is None else histories[0].index.name
    metrics = [c[0]for c in _get_column_tuples(histories[0])]
    number_of_metrics = len(metrics)
    w = min(number_of_metrics, graphs_per_row)
    h = math.ceil(number_of_metrics/graphs_per_row)
    fig, axes = plt.subplots(h, w, figsize=(side*w, side*h), dpi=200, constrained_layout=True)
    flat_axes = np.array(axes).flatten()

    for i, history in enumerate([average_history] + histories):
        for metric, axis in zip(metrics, flat_axes):
            for name, kind in zip(*(((metric, f"val_{metric}"), ("Train", "Test"))
                    if f"val_{metric}" in history
                    else ((metric, ), ("", )))):
                col = history[name]
                if i == 0:
                    if best_point_x is not None:
                        best_point_y = col.values[best_point_x]
                        if len(kind) == 0:
                            kind = f"Best value ({monitor})"
                        else:
                            kind = f"{kind} best value ({monitor})"
                    else:
                        best_point_y = col.iloc[-1]
                        if len(kind) == 0:
                            kind = f"Last value"
                        else:
                            kind = f"{kind} last value"

                    line = axis.plot(col.index.values, col.values,
                                     style, label='{kind}: {val:0.4f}'.format(
                                     kind=kind, val=best_point_y), zorder=10000)[0]
                    if best_point_x is not None:
                        best_point_y = col.values[best_point_x]
                        axis.scatter([best_point_x], [best_point_y], s=30,
                                     alpha=0.9, color=line.get_color(), zorder=10000)
                        axis.hlines(best_point_y, 0, best_point_x, linestyles="dashed",
                                    color=line.get_color(), alpha=0.5)
                        axis.vlines(best_point_x, 0, best_point_y, linestyles="dashed",
                                    color=line.get_color(), alpha=0.5)
                else:
                    axis.plot(col.index.values, col.values, style, alpha=0.3)

    for metric, axis in zip(metrics, flat_axes):
        alias = metric.capitalize()
        axis.set_xlabel(x_label)
        if log_scale_metrics:
            axis.set_yscale("log")
        axis.set_ylabel("{alias}{scale}".format(
            alias=alias,
            scale=" (Log scale)" if log_scale_metrics else ""
        ))
        axis.set_title(alias)
        axis.grid(True)
        axis.legend()
        # if is_normalized_metric(metric):
        #     axis.set_ylim(-0.05, 1.05)
        if history.shape[0] <= 4:
            axis.set_xticks(range(history.shape[0]))
        if customization_callback is not None:
            customization_callback(axis)

    for axis in flat_axes[len(metrics):]:
        axis.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=20)

    if path is not None:
        fig.savefig(path)

    return fig, axes


def _to_dataframe(history: Union[History, pd.DataFrame, Dict, str]) -> pd.DataFrame:
    """Return given history normalized to a dataframe.
    Parameters
    -----------------------------
    history: Union[pd.DataFrame, Dict, str],
        The history object to be normalized.
        Supported values are:
        - pandas DataFrames
        - Dictionaries
        - History object from Keras Callbacks
        - Paths to csv and json files
    Raises
    -----------------------------
    TypeError,
        If given history object is not supported.
    Returns
    -----------------------------
    Normalized pandas dataframe history object.
    """
    if isinstance(history, pd.DataFrame):
        return history
    if isinstance(history, Dict):
        return pd.DataFrame(history)
    if isinstance(history, History):
        return _to_dataframe(history.history)
    if isinstance(history, str):
        if "csv" in history.split("."):
            return pd.read_csv(history)
        if "json" in history.split("."):
            return pd.read_json(history)
    raise TypeError("Given history object of type {history_type} is not currently supported!".format(
        history_type=type(history)))


def _get_column_tuples(history: pd.DataFrame) -> List[List[str]]:
    """Return tuples of the columns to plot.
    Parameters
    -----------------------
    history: pd.DataFrame,
        Pandas dataframe with the training history.
    Returns
    -----------------------
    List of the tuples of columns
    """
    return [[c, ] if f"val_{c}" not in history else [c,  f"val_{c}"]
        for c in history.columns if not c.startswith("val_") and history[c].notna().all()]

