"""Pulse shape library"""

from typing import List, Union, Optional
import copy
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

s = 1.
ns = 1.e-9

class Pulse:
    @staticmethod
    def _zero(t, args):
        return np.zeros_like(t)
    
    def __init__(self, start, duration, frequency=None, phase=None):
        assert duration > 0., 'Pulse duration must be positive'
        self.start = start
        self.end = start + duration
        self.frequency = frequency
        self.phase = phase
        
    def __call__(self, t, args):
        zero = np.array(0., dtype=np.complex128)
        fval = np.asarray(self._call(t, args), dtype=np.complex128)
        return np.select(
            [(t >= self.start) & (t < self.end)],
            [fval], default=zero)
    
    def __mul__(self, c):
        if not isinstance(c, (float, complex)):
            raise TypeError(f'Multiplication between Pulse and {type(c).__name__} is invalid')
            
        self._scale(c)
        

class Gaussian(Pulse):
    def __init__(
        self,
        start: float,
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        center: Optional[float] = None,
        zero_ends: bool = True,
        frequency: Optional[float] = None,
        phase: Optional[float] = None
    ):
        super().__init__(start, duration, frequency=frequency, phase=phase)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'

        self.sigma = sigma

        if center is None:
            self.center = (self.end + self.start) * 0.5
        else:
            self.center = center
            
        if zero_ends:
            assert np.abs(self.center - (self.end + self.start) * 0.5) < 1. * ns, \
                'Zero-ended Gaussian must be centered'
            
            x = (self.end - self.start) * 0.5 / self.sigma
            self.offset = np.exp(-np.square(x) * 0.5)
            self.amp = np.asarray(amp / (1. - self.offset), dtype=np.complex128)

        else:
            self.amp = np.asarray(amp, dtype=np.complex128)
            self.offset = 0.
            
    def _call(self, t, args):
        x = (t - self.center) / self.sigma
        return np.asarray(self.amp * (np.exp(-np.square(x) * 0.5) - self.offset),
                          dtype=np.complex128)
    
    def _scale(self, c):
        self.amp *= c
            

class GaussianSquare(Pulse):
    def __init__(
        self,
        start: float,
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        width: float,
        zero_ends: bool = True,
        frequency: Optional[float] = None,
        phase: Optional[float] = None
    ):
        super().__init__(start, duration, frequency=frequency, phase=phase)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'
        assert width < duration, 'GaussianSquare width must be less than duration'
        
        self.gauss_width = duration - width
        self.gauss_left = Gaussian(start, self.gauss_width, amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)
        self.gauss_right = Gaussian(start + width, self.gauss_width, amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)

        self.amp = np.asarray(amp, dtype=np.complex128)
        self.width = width

    def _call(self, t, args):
        t1 = self.start + self.gauss_width / 2.
        t2 = t1 + self.width
        # piecewise determines the output dtype from the first argument
        tlist = np.asarray(t, dtype=np.complex128)
        return np.piecewise(tlist,
            [t < t1, t >= t2],
            [self.gauss_left._call, self.gauss_right._call, self.amp],
            args)
    
    def _scale(self, c):
        self.amp *= c

    
class Drag(Gaussian):
    def __init__(
        self,
        start: float,
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        beta: float,
        center: Optional[float] = None,
        zero_ends: bool = True,
        frequency: Optional[float] = None,
        phase: Optional[float] = None
    ):
        super().__init__(
            start,
            duration,
            amp=amp,
            sigma=sigma,
            center=center,
            zero_ends=zero_ends,
            frequency=frequency,
            phase=phase
        )
        
        self.beta = beta
        
    def _call(self, t, args):
        gauss = super()._call(t, args)
        dgauss = -(t - self.center) / np.square(self.sigma) * gauss

        return gauss + 1.j * self.beta * dgauss
    
    
class Sequence(Pulse):
    def __init__(
        self,
        pulses: List[Pulse],
        frequency: Optional[float] = None,
        phase: Optional[float] = None
    ):
        
        self.pulses = list()
        current_frequency = None
        current_phase = None
        for pulse in pulses:
            if isinstance(pulse, ShiftFrequency):
                if current_frequency is None:
                    current_frequency = 1.j * pulse.value
                else:
                    current_frequency += pulse.value
                    
            elif isinstance(pulse, ShiftPhase):
                if current_phase is None:
                    current_phase = 1.j * pulse.value
                else:
                    current_phase += pulse.value
                    
            elif isinstance(pulse, SetFrequency):
                current_frequency = pulse.value

            elif isinstance(pulse, SetPhase):
                current_phase = pulse.value
                
            else:
                pulse = copy.deepcopy(pulse)
                
                if isinstance(current_frequency, float):
                    if pulse.frequency is not None:
                        logger.warning('Overriding the frequency set for pulse %s', pulse)
                    pulse.frequency = current_frequency
                elif isinstance(current_frequency, complex):
                    if pulse.frequency is None:
                        pulse.frequency = current_frequency
                    else:
                        pulse.frequency += -1.j * current_frequency
                        
                if pulse.frequency is not None:
                    current_frequency = current_frequency

                if isinstance(current_phase, float):
                    if pulse.phase is not None:
                        logger.warning('Overriding the phase set for pulse %s', pulse)
                    pulse.phase = current_phase
                elif isinstance(current_phase, complex):
                    if pulse.phase is None:
                        pulse.phase = current_phase
                    else:
                        pulse.phase += -1.j * current_phase
                        
                if pulse.phase is not None:
                    current_phase = current_phase
                    
                self.pulses.append(pulse)
        
        assert all((self.pulses[i].end <= self.pulses[i + 1].start) for i in range(len(pulses) - 1))
        
        super().__init__(self.pulses[0].start, self.pulses[-1].end - self.pulses[0].start,
                         frequency=frequency, phase=phase)
        
    def _call(self, t, args):
        # piecewise determines the output dtype from tlist
        tlist = np.asarray(t, dtype=np.complex128)
        condlist = [((t >= p.start) & (t < p.end)) for p in self.pulses]
        funclist = [p._call for p in self.pulses]
        return np.piecewise(tlist, condlist, funclist, args)

    def _scale(self, c):
        for pulse in self.pulses:
            pulse._scale(c)

@dataclass
class ShiftFrequency:
    value: float

@dataclass
class ShiftPhase:
    value: float

@dataclass
class SetFrequency:
    value: float

@dataclass
class SetPhase:
    value: float
    