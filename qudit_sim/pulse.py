"""Pulse shape library."""

from typing import List, Union, Optional, Any
import copy
import logging
from dataclasses import dataclass
import numpy as np

from .drive import CallableCoefficient

logger = logging.getLogger(__name__)

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

    @property
    def frequency(self):
        frequency = 0.
        max_frequency = 0.
        for inst in self:
            if isinstance(inst, ShiftFrequency):
                frequency += inst.value
                max_frequency = max(max_frequency, frequency)
            elif isinstance(inst, SetFrequency):
                frequency = inst.value
                max_frequency = max(max_frequency, frequency)

        return max_frequency

    def __str__(self):
        return f'PulseSequence([{", ".join(str(inst) for inst in self)}])'

    def generate_fn(
        self,
        frame_frequency: float,
        drive_base: complex,
        rwa: bool = False
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
            X and Y coefficient functions.
        """
        if len(self) == 0:
            return 0., 0.

        assert isinstance(self[0], SetFrequency), 'First instruction of a PulseSequence must be SetFrequency'

        funclist_x = []
        funclist_y = []
        timelist = []

        frequency = None
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

        fn_x = _make_sequence_fn(timelist, funclist_x)
        fn_y = _make_sequence_fn(timelist, funclist_y)

        return fn_x, fn_y

    def envelope(self, t: Union[float, np.ndarray], args: Any = None) -> np.ndarray:
        """Return the envelope of the sequence as a function of time.

        This function is mostly for visualization purposes. Phase and frequency information is lost in the
        returned array.

        Args:
            t: Time or array of time points.
            args: Second argument to the pulse envelope functions.

        Returns:
            Pulse sequence envelope (complex) as a function of time.
        """
        funclist = []
        timelist = []

        time = 0.

        for inst in self:
            if isinstance(inst, Delay):
                funclist.append(0.)
                timelist.append(time)
                time += inst.value
            elif isinstance(inst, Pulse):
                funclist.append(_make_shifted_fn(inst, time))
                timelist.append(time)
                time += inst.duration

        timelist.append(time)

        # piecewise determines the output dtype from tlist
        tlist = np.asarray(t, dtype=np.complex128)
        condlist = [((t >= start) & (t < end)) for start, end in zip(timelist[:-1], timelist[1:])]
        return np.piecewise(tlist, condlist, funclist, args)


def _make_sequence_fn(timelist, funclist):
    def fn(t, args):
        # piecewise determines the output dtype from tlist
        tlist = np.asarray(t, dtype=np.float)
        condlist = [((t >= start) & (t < end)) for start, end in zip(timelist[:-1], timelist[1:])]
        return np.piecewise(tlist, condlist, funclist, args)

    return fn

def _make_shifted_fn(pulse, start_time):
    def fn(t, args):
        return pulse(t - start_time, args)

    return fn

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

    def __str__(self):
        return f'Pulse(duration={self.duration})'

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

        self.sigma = sigma

        if center is None:
            self.center = self.duration * 0.5
        else:
            self.center = center

        if zero_ends:
            assert np.isclose(self.center, self.duration * 0.5), \
                'Zero-ended Gaussian must be centered'

            x = self.duration * 0.5 / self.sigma
            self._pedestal = np.exp(-np.square(x) * 0.5)
            self._gaus_amp = np.asarray(amp / (1. - self._pedestal), dtype=np.complex128)

        else:
            self._pedestal = 0.
            self._gaus_amp = np.asarray(amp, dtype=np.complex128)

    @property
    def amp(self) -> complex:
        return complex(self._gaus_amp * (1. - self._pedestal))

    def __call__(self, t, args=None):
        x = (t - self.center) / self.sigma
        return np.asarray(self._gaus_amp * (np.exp(-np.square(x) * 0.5) - self._pedestal),
                          dtype=np.complex128)

    def __str__(self):
        return (f'Gaussian(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, center={self.center},'
                f' zero_ends={self._pedestal != 0.})')

    def _scale(self, c):
        self._gaus_amp *= c


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
        zero_ends: bool = True,
        rise: bool = True,
        fall: bool = True
    ):
        if not (rise or fall):
            raise ValueError("That's just a square pulse, dude")

        super().__init__(duration)

        assert width < duration, 'GaussianSquare width must be less than duration'

        self.sigma = sigma
        self.width = width

        self._condlist = list()
        self._funclist = list()

        ramp_duration = duration - width
        if rise and fall:
            gauss_duration = ramp_duration
        else:
            gauss_duration = ramp_duration * 2.

        if rise:
            self.t_plateau = gauss_duration / 2.
            self.gauss_rise = Gaussian(duration=gauss_duration, amp=amp, sigma=sigma,
                                       center=None, zero_ends=zero_ends)

            self._condlist.append(lambda t: t < self.t_plateau)
            self._funclist.append(self.gauss_rise)
        else:
            self.t_plateau = 0.
            self.gauss_rise = None

        if fall:
            self.gauss_fall = Gaussian(duration=gauss_duration, amp=amp, sigma=sigma,
                                       center=None, zero_ends=zero_ends)

            self._condlist.append(lambda t: t >= self.t_plateau + self.width)
            self._funclist.append(self._fall_tail)
        else:
            self.gauss_fall = None

        self._funclist.append(complex(amp))

    @property
    def amp(self) -> complex:
        return self._funclist[-1]

    def _fall_tail(self, t, args):
        gauss_t0 = self.t_plateau + self.width - self.gauss_fall.duration / 2.
        return self.gauss_fall(t - gauss_t0, args)

    def _make_condlist(self, t):
        return list(cond(t) for cond in self._condlist)

    def __call__(self, t, args=None):
        # np.piecewise determines the output dtype from the first argument
        tlist = np.asarray(t, dtype=np.complex128)
        return np.piecewise(tlist, self._make_condlist(t), self._funclist, args)

    def __str__(self):
        rise = self.gauss_rise is not None
        fall = self.gauss_fall is not None

        if rise:
            zero_ends = self.gauss_rise._pedestal != 0.
        else:
            zero_ends = self.gauss_fall._pedestal != 0.

        return (f'GaussianSquare(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, width={self.width}, '
                f' zero_ends={zero_ends}, rise={rise}, fall={fall})')

    def _scale(self, c):
        self._funclist[-1] *= c

        if self.gauss_rise:
            self.gauss_rise._scale(c)
        if self.gauss_fall:
            self.gauss_fall._scale(c)


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

    def __str__(self):
        return (f'Drag(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, beta={self.beta}, '
                f'center={self.center}, zero_ends={self._pedestal != 0.})')


class Square(Pulse):
    """Square (constant) pulse.

    Args:
        duration: Pulse duration.
        amp: Pulse height relative to the drive base amplitude.
    """
    def __init__(
        self,
        duration: float,
        amp: Union[float, complex]
    ):
        super().__init__(duration)

        self.amp = complex(amp)

    def __call__(self, t, args=None):
        return np.full_like(t, self.amp)

    def __str__(self):
        return f'Square(duration={self.duration}, amp={self.amp})'

    def _scale(self, c):
        self.amp *= c


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
