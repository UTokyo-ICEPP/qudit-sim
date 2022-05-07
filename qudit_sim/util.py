from typing import Callable, Union, Any
import sys
import tempfile
import enum
import numpy as np
import h5py

twopi = 2. * np.pi

time_units = ['s', 'ms', 'us', 'ns']

class FrequencyScale(enum.Enum):
    Hz = 0
    kHz = 1
    MHz = 2
    GHz = 3
    
    @property
    def frequency_value(self):
        return np.power(10., 3 * self.value)
    
    @property
    def frequency_unit(self):
        return self.name
    
    @property
    def pulsatance_value(self):
        return self.frequency_value * twopi
    
    @property
    def pulsatance_unit(self):
        return self.name.replace('Hz', 'rad/s')
    
    @property
    def time_unit(self):
        return time_units[self.value]

