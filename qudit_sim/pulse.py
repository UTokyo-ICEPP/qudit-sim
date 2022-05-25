"""Pulse shape library."""

from typing import List, Union, Optional, Any
import copy
import logging
from dataclasses import dataclass
import numpy as np

from .util import FrequencyScale, CallableCoefficient

logger = logging.getLogger(__name__)

s = FrequencyScale.Hz.time_value
ns = FrequencyScale.GHz.time_value

class PulseSequence(list):
    """Pulse sequence.

    This class represents a sequence of instructions (pulse, delay, frequency/phase shift/set)
    given to a single channel. In practice, the class is implemented as a subclass of Python
    list with a single additional function `generate_fn`.
    """

    @property
    def duration(self):
        d = sum(inst.value for inst in self if isinstance(inst, Delay))
        d += sum(inst.duration for inst in self if isinstance(inst, Pulse))
        return d

    def generate_fn(
        self,
        frame_frequency: float,
        drive_base: complex,
        rwa: bool = True,
        initial_frequency: Optional[float] = None
    ) -> tuple:
        r"""Generate the X and Y drive coefficients and the maximum frequency appearing in the sequence.

        The return values are equivalent to what is returned by `drive.DriveTerm.generate_fn` (in fact
        this function is called within `DriveTerm.generate_fn` when the drive amplitude is given as a
        PulseSequence).

        Args:
            frame_frequency: Frame frequency :math:`\xi_k^{l}`.
            drive_base: Factor :math:`\alpha_{jk} e^{i \rho_{jk}} \frac{\Omega_j}{2}`.
            rwa: If True, returns the RWA coefficients.
            initial_frequency: Initial carrier frequency. Can be None if the first instruction
                is SetFrequency.

        Returns:
            A 3-tuple of X and Y coefficients and the maximum frequency appearing in the term.
        """
        funclist_x = []
        funclist_y = []
        timelist = []

        frequency = initial_frequency
        max_frequency = 0. if frequency is None else frequency
        phase_offset = 0.
        time = 0.

        for inst in self:
            if isinstance(inst, ShiftFrequency):
                frequency += inst.value
                max_frequency = max(max_frequency, frequency)
            elif isinstance(inst, ShiftPhase):
                phase_offset += inst.value
            elif isinstance(inst, SetFrequency):
                frequency = inst.value
                max_frequency = max(max_frequency, frequency)
            elif isinstance(inst, SetPhase):
                phase_offset = inst.value - frequency * time
            elif isinstance(inst, Delay):
                funclist_x.append(0.)
                funclist_y.append(0.)
                timelist.append(time)
                time += inst.value
            elif isinstance(inst, Pulse):
                funclist_x.append(inst.modulate(drive_base, frequency, phase_offset,
                                                frame_frequency, time, 'x', rwa))
                funclist_y.append(inst.modulate(drive_base, frequency, phase_offset,
                                                frame_frequency, time, 'y', rwa))
                timelist.append(time)
                time += inst.duration

        timelist.append(time)

        def make_fn(timelist, funclist):
            def fn(t, args):
                # piecewise determines the output dtype from tlist
                t = np.asarray(t, dtype=np.float)
                condlist = [((t >= start) & (t < end)) for start, end in zip(timelist[:-1], timelist[1:])]
                return np.piecewise(t, condlist, funclist, args)

            return fn

        fn_x = make_fn(timelist, funclist_x)
        fn_y = make_fn(timelist, funclist_y)

        return fn_x, fn_y, max_frequency

    def envelope(self, t: Union[float, np.ndarray], args: Any = None) -> np.ndarray:
        """Return the envelope of the sequence as a function of time.

        This function is mostly for visualization purposes. Phase and frequency information is lost in the
        returned array.

        Args:
            t: Time or array of time points.

        Returns:
            Pulse sequence envelope (complex) as a function of time.
        """
        funclist = []
        timelist = []

        time = 0.

        def make_shifted(pulse, start_time):
            def fn(t):
                return pulse(t - start_time, args)

            return fn

        for inst in self:
            if isinstance(inst, Delay):
                funclist.append(0.)
                timelist.append(time)
                time += inst.value
            elif isinstance(inst, Pulse):
                funclist.append(make_shifted(inst, time))
                timelist.append(time)
                time += inst.duration

        timelist.append(time)

        # piecewise determines the output dtype from tlist
        t = np.asarray(t, dtype=np.float)
        condlist = [((t >= start) & (t < end)) for start, end in zip(timelist[:-1], timelist[1:])]
        return np.piecewise(t, condlist, funclist)


class Pulse:
    """Base class for all pulse shapes.

    Args:
        duration: Pulse duration.
    """

    def __init__(self, duration: float):
        assert duration > 0., 'Pulse duration must be positive'
        self.duration = duration

    def __call__(self, t, args=None):
        raise NotImplementedError('Pulse is an ABC')

    def __mul__(self, c):
        if not isinstance(c, (float, complex)):
            raise TypeError(f'Multiplication between Pulse and {type(c).__name__} is invalid')

        instance = copy.deepcopy(self)
        instance._scale(c)
        return instance

    def modulate(
        self,
        drive_base: complex,
        frequency: float,
        phase_offset: float,
        frame_frequency: float,
        start_time: float,
        symmetry: str,
        rwa: bool
    ) -> CallableCoefficient:
        r"""Modulate the tone at the given amplitude and frequency with this pulse.

        Args:
            drive_base: Factor :math:`\alpha_{jk} e^{i \rho_{jk}} \frac{\Omega_j}{2}`.
            frequency: Tone frequency.
            phase_offset: Tone phase offset.
            frame_frequency: Frequency of the observing frame.
            start_time: Start time of the pulse.
            symmetry: 'x' or 'y'.
            rwa: Whether to apply the rotating-wave approximation.

        Returns:
            A function of time that returns the modulated signal.
        """
        if rwa:
            if symmetry == 'x':
                value = lambda envelope, phase: envelope.real * np.cos(phase) + envelope.imag * np.sin(phase)
            else:
                value = lambda envelope, phase: envelope.imag * np.cos(phase) - envelope.real * np.sin(phase)

            def fun(t, args=None):
                envelope = drive_base * self.__call__(t - start_time, args)
                phase = (frequency - frame_frequency) * t + phase_offset
                return value(envelope, phase)

        else:
            if symmetry == 'x':
                frame_fn = np.cos
            else:
                frame_fn = np.sin

            def fun(t, args=None):
                double_envelope = 2. * drive_base * self.__call__(t - start_time, args)
                phase = frequency * t + phase_offset
                prefactor = double_envelope.real * np.cos(phase) + double_envelope.imag * np.sin(phase)
                return prefactor * frame_fn(frame_frequency * t)

        return fun


class Gaussian(Pulse):
    """Gaussian pulse.

    Args:
        duration: Pulse duration.
        amp: Gaussian height relative to the drive base amplitude.
        sigma: Gaussian width in seconds.
        center: Center of the Gaussian as time in seconds from the beginning of the pulse.
        zero_ends: Whether to "ground" the pulse by removing the pedestal and rescaling the
            amplitude accordingly.
    """
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

    def __call__(self, t, args=None):
        x = (t - self.center) / self.sigma
        return np.asarray(self.amp * (np.exp(-np.square(x) * 0.5) - self.offset),
                          dtype=np.complex128)

    def _scale(self, c):
        self.amp *= c


class GaussianSquare(Pulse):
    """Gaussian-square pulse.

    Args:
        duration: Pulse duration.
        amp: Gaussian height relative to the drive base amplitude.
        sigma: Gaussian width in seconds.
        zero_ends: Whether to "ground" the pulse by removing the pedestal and rescaling the
            amplitude accordingly.
    """
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

        self.t1 = (duration - width) / 2.

        self.amp = amp
        self.width = width

        self.gauss_left = Gaussian(duration=self.t1 * 2., amp=amp, sigma=sigma,
                                   center=None, zero_ends=zero_ends)
        self.gauss_right = Gaussian(duration=self.t1 * 2., amp=amp, sigma=sigma,
                                    center=None, zero_ends=zero_ends)

    def _right_tail(self, t, args):
        return self.gauss_right(t - self.width, args)

    def __call__(self, t, args=None):
        # piecewise determines the output dtype from the first argument
        tlist = np.asarray(t, dtype=np.complex128)
        return np.piecewise(tlist,
            [t < self.t1, t >= self.t1 + self.width],
            [self.gauss_left, self._right_tail, self.amp],
            args)

    def _scale(self, c):
        self.amp *= c
        self.gauss_left._scale(c)
        self.gauss_right._scale(c)


class Drag(Gaussian):
    r"""DRAG pulse.

    Args:
        duration: Pulse duration.
        amp: Gaussian height relative to the drive base amplitude.
        sigma: Gaussian width in seconds.
        beta: DRAG :math:`\beta` factor.
        center: Center of the Gaussian as time in seconds from the beginning of the pulse.
        zero_ends: Whether to "ground" the pulse by removing the pedestal and rescaling the
            amplitude accordingly.
    """
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

    def __call__(self, t, args=None):
        gauss = super().__call__(t, args)
        dgauss = -(t - self.center) / np.square(self.sigma) * gauss

        return gauss + 1.j * self.beta * dgauss


@dataclass(frozen=True)
class ShiftFrequency:
    """Frequency shift in rad/s."""
    value: float

@dataclass(frozen=True)
class ShiftPhase:
    """Phase shift (virtual Z)."""
    value: float

@dataclass(frozen=True)
class SetFrequency:
    """Frequency setting in rad/s."""
    value: float

@dataclass(frozen=True)
class SetPhase:
    """Phase setting."""
    value: float

@dataclass(frozen=True)
class Delay:
    """Delay in seconds."""
    value: float
