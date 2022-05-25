"""Miscellaneous utility classes and functions."""

from typing import Union, Tuple, Callable
import enum
from dataclasses import dataclass
import numpy as np

# Type for the callable time-dependent Hamiltonian coefficient
CallableCoefficient = Callable[[Union[float, np.ndarray], dict], Union[complex, np.ndarray]]
HamiltonianCoefficient = Union[str, np.ndarray, CallableCoefficient]

@dataclass(frozen=True)
class PulseSimResult:
    """Return type of pulse_sim.

    See the docstring of pulse_sim for why this class is necessary.
    """
    times: np.ndarray
    expect: Union[np.ndarray, None]
    states: Union[np.ndarray, None]
    dim: Tuple[int, ...]


_time_units = ['s', 'ms', 'us', 'ns']

class FrequencyScale(enum.Enum):
    """Frequency and corresponding time units."""
    Hz = 0
    kHz = 1
    MHz = 2
    GHz = 3

    auto = None

    @property
    def frequency_value(self):
        return np.power(10., 3 * self.value)

    @property
    def frequency_unit(self):
        return self.name

    @property
    def pulsatance_value(self):
        return self.frequency_value * 2. * np.pi

    @property
    def pulsatance_unit(self):
        return self.name.replace('Hz', 'rad/s')

    @property
    def time_value(self):
        return np.power(10., -3 * self.value)

    @property
    def time_unit(self):
        return _time_units[self.value]

    @classmethod
    def find_scale(cls, val):
        for scale in reversed(cls):
            if scale is cls.auto:
                continue

            if val > 0.1 * scale.pulsatance_value:
                return scale

        raise RuntimeError(f'Could not find a proper scale for value {val}')
