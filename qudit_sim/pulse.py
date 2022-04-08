"""Pulse shape library"""

from typing import List, Union, Optional
import copy
import logging
from collections import namedtuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

s = 1.
ns = 1.e-9

class PulseSequence(list):
    def generate_fn(self, initial_frequency, reference_frequency, reference_phase=0.):
        funclist = []
        timelist = []
        
        def modulate(frequency, phase_offset, time, pulse):
            detuning = frequency - reference_frequency
            offset = phase_offset - reference_phase
            def fun(t, args):
                return np.exp(1.j * detuning * t + offset) * pulse(t - time, args)
            
            return fun

        frequency = initial_frequency
        phase_offset = 0.
        time = 0.
        for inst in self:
            if isinstance(inst, ShiftFrequency):
                frequency += inst.value
            elif isinstance(inst, ShiftPhase):
                phase_offset += inst.value
            elif isinstance(inst, SetFrequency):
                frequency = inst.value
            elif isinstance(inst, SetPhase):
                phase_offset = inst.value - frequency * time
            elif isinstance(inst, Delay):
                funclist.append(0.)
                timelist.append(time)
                time += inst.value
            elif isinstance(inst, Pulse):
                funclist.append(modulate(frequency, phase_offset, inst))
                timelist.append(time)
                time += inst.duration
                
        timelist.append(time)

        def fn(t, args):
            # piecewise determines the output dtype from tlist
            tlist = np.asarray(t, dtype=np.complex128)
            condlist = [((t >= start) & (t < end)) for start, end in zip(timelist[:-1], timelist[1:])]
            return np.piecewise(tlist, condlist, funclist, args)
        
        return fn


class Pulse:
    def __init__(self, duration):
        assert duration > 0., 'Pulse duration must be positive'
        self.duration = duration
        
    def __call__(self, t, args):
        raise NotImplementedError('Pulse is an ABC')
    
    def __mul__(self, c):
        if not isinstance(c, (float, complex)):
            raise TypeError(f'Multiplication between Pulse and {type(c).__name__} is invalid')
            
        self._scale(c)
        

class Gaussian(Pulse):
    def __init__(
        self,
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        center: Optional[float] = None,
        zero_ends: bool = True
    ):
        super().__init__(duration)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'

        self.sigma = sigma

        if center is None:
            self.center = self.duration * 0.5
        else:
            self.center = center
            
        if zero_ends:
            assert np.abs(self.center - self.duration * 0.5) < 1. * ns, \
                'Zero-ended Gaussian must be centered'
            
            x = self.duration * 0.5 / self.sigma
            self.offset = np.exp(-np.square(x) * 0.5)
            self.amp = np.asarray(amp / (1. - self.offset), dtype=np.complex128)

        else:
            self.amp = np.asarray(amp, dtype=np.complex128)
            self.offset = 0.
            
    def __call__(self, t, args):
        x = (t - self.center) / self.sigma
        return np.asarray(self.amp * (np.exp(-np.square(x) * 0.5) - self.offset),
                          dtype=np.complex128)
    
    def _scale(self, c):
        self.amp *= c
            

class GaussianSquare(Pulse):
    def __init__(
        self,
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        width: float,
        zero_ends: bool = True
    ):
        super().__init__(duration)
        
        assert sigma > 1. * ns, 'Gaussian sigma must be greater than 1 ns'
        assert width < duration, 'GaussianSquare width must be less than duration'
        
        self.gauss_width = duration - width
        self.gauss_left = Gaussian(self.gauss_width, amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)
        self.gauss_right = Gaussian(self.gauss_width, amp=amp, sigma=sigma,
                                    center=None, zero_ends=zero_ends)

        self.amp = np.asarray(amp, dtype=np.complex128)
        self.width = width

    def __call__(self, t, args):
        t1 = self.gauss_width / 2.
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
        duration: float,
        amp: Union[float, complex],
        sigma: float,
        beta: float,
        center: Optional[float] = None,
        zero_ends: bool = True
    ):
        super().__init__(
            duration,
            amp=amp,
            sigma=sigma,
            center=center,
            zero_ends=zero_ends
        )
        
        self.beta = beta
        
    def _call(self, t, args):
        gauss = super()(t, args)
        dgauss = -(t - self.center) / np.square(self.sigma) * gauss

        return gauss + 1.j * self.beta * dgauss
    
            
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
    
@dataclass
class Delay:
    value: float
