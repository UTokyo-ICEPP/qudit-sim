"""Miscellaneous utility classes and functions."""

from typing import Union, Tuple, Callable
import enum
from dataclasses import dataclass
import numpy as np

# Type for the callable time-dependent Hamiltonian coefficient
CallableCoefficient = Callable[[Union[float, np.ndarray], dict], Union[complex, np.ndarray]]
HamiltonianCoefficient = Union[str, np.ndarray, CallableCoefficient]

@dataclass(frozen=True)
class Frame:
    """Frame specification for a single level gap of a qudit."""
    frequency: np.ndarray
    phase: np.ndarray


@dataclass(frozen=True)
class PulseSimResult:
    """Return type of pulse_sim.

    See the docstring of pulse_sim for why this class is necessary.
    """
    times: np.ndarray
    expect: Union[np.ndarray, None]
    states: Union[np.ndarray, None]
    dim: Tuple[int, ...]
    frame: Tuple[Frame, ...]


_frequency_units = list(f'{f}{u}' for u in ['Hz', 'kHz', 'MHz', 'GHz', 'mHz'] for f in ['', '10', '100'])
_time_units = ['s'] + list(f'{f}{u}' for u in ['ms', 'us', 'ns', 'ps'] for f in ['100', '10', ''])[:-1] + ['ks', '100s', '10s']

class FrequencyScale(enum.Enum):
    """Frequency and corresponding time units."""
    mHz = -3
    tenmHz = -2
    hundredmHz = -1
    Hz = 0
    tenHz = 1
    hundredHz = 2
    kHz = 3
    tenkHz = 4
    hundredkHz = 5
    MHz = 6
    tenMHz = 7
    hundredMHz = 8
    GHz = 9
    tenGHz = 10
    hundredGHz = 11

    auto = None

    @property
    def frequency_value(self):
        return np.power(10., self.value)

    @property
    def frequency_unit(self):
        return _frequency_units[self.value]

    @property
    def pulsatance_value(self):
        return self.frequency_value * 2. * np.pi

    @property
    def pulsatance_unit(self):
        return self.name.replace('Hz', 'rad/s')

    @property
    def time_value(self):
        return np.power(10., -self.value)

    @property
    def time_unit(self):
        return _time_units[self.value]

    @classmethod
    def find_energy_scale(cls, val):
        for scale in cls:
            if scale is cls.auto:
                raise RuntimeError(f'Could not find a proper energy scale for value {val}')

            if 0.1 * val < scale.pulsatance_value:
                return scale

    @classmethod
    def find_time_scale(cls, val):
        for scale in cls:
            if scale is cls.auto:
                continue

            if val > scale.time_value:
                return scale

        raise RuntimeError(f'Could not find a proper time scale for value {val}')
