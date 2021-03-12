import tensorflow as tf
from datetime import datetime


def gpu_memory_growth(verbose=True):
    physical_devices = gpu_get_devices() 
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(physical_devices)


def gpu_get_devices(which='physical'):
    if which == 'physical':
        devices = tf.config.list_physical_devices('GPU')
    elif which == 'logical':
        devices = tf.config.list_logical_devices('GPU')
    return devices


def gpu_set_visible(indices=(0)):
    physical_devices = gpu_get_devices('physical')
    tf.config.set_visible_devices(physical_devices[indices], 'GPU') 
    print(gpu_get_devices('logical'))


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

