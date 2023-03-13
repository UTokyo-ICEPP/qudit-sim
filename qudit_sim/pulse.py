r"""
============================================
Pulse shape library (:mod:`qudit_sim.pulse`)
============================================

.. currentmodule:: qudit_sim.pulse

Classes in this module represent pulse envelopes. Subclasses of Pulse can be passed to
``HamiltonianBuilder.add_drive`` either as the ``amplitude`` parameter or through the PulseSequence
class (``sequence`` parameter).
"""

from typing import List, Union, Optional, Any, Sequence, Callable
from numbers import Number
import copy
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from .expression import ParameterExpression, Constant, Parameter, TimeFunction, ArrayType, array_like
from .config import config

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
        funclist = list()
        time = 0.

        for inst in self:
            if isinstance(inst, Delay):
                funclist.append((time, 0.))
                time += inst.value
            elif isinstance(inst, Pulse):
                pulse = copy.copy(inst)
                pulse.tzero = time
                funclist.append((time, pulse))
                time += inst.duration

        funclist.append((time, None))

        result = 0.
        for time, func in funclist[:-1]:
            if isinstance(func, TimeFunction):
                result = np.where(t > time, func(t, args), result)
            else:
                result = np.where(t > time, func, result)

        result = np.where(t > timelist[-1], 0., result)

        return result


class Pulse(TimeFunction):
    """Base class for all pulse shapes.

    Args:
        duration: Pulse duration.
    """

    def __init__(
        self,
        duration: float,
        fn: Callable[[Tuple[Any, ...]], ReturnType],
        parameters: Optional[Tuple[str, ...]] = None,
        tzero: float = 0.
    ):
        assert duration > 0., 'Pulse duration must be positive'
        self.duration = duration
        super().__init__(fn, parameters, tzero)

    def __str__(self):
        return f'Pulse(duration={self.duration}, tzero={self.tzero})'


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
        amp: Union[float, complex, ParameterExpression],
        sigma: float,
        center: Optional[float] = None,
        zero_ends: bool = True,
        tzero: float = 0.
    ):
        if isinstance(amp, Number):
            self.amp = ConstantExpression(complex(amp))
        else:
            self.amp = amp

        self.sigma = sigma

        if center is None:
            self.center = duration * 0.5
        else:
            self.center = center

        if zero_ends:
            assert np.isclose(self.center, self.duration * 0.5), \
                'Zero-ended Gaussian must be centered'

            x = self.duration * 0.5 / self.sigma
            self._pedestal = np.exp(-np.square(x) * 0.5)
        else:
            self._pedestal = 0.

        npmod = config.npmod

        def fn(t, args):
            x = (t - self.center) / self.sigma
            v = npmod.exp(-npmod.square(x) * 0.5)
            amp = self.amp.evaluate(args)
            return npmod.asarray(amp / (1. - self._pedestal) * (v - self._pedestal),
                                 dtype='complex128')

        super().__init__(duration, fn, self.amp.parameters, tzero)

    def __str__(self):
        return (f'Gaussian(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, center={self.center},'
                f' zero_ends={self._pedestal != 0.}, tzero={self.tzero})')


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
        amp: Union[float, complex, ParameterExpression],
        sigma: float,
        width: float,
        zero_ends: bool = True,
        rise: bool = True,
        fall: bool = True,
        tzero: float = 0.
    ):
        if not (rise or fall):
            raise ValueError("That's just a square pulse, dude")

        assert width < duration, 'GaussianSquare width must be less than duration'

        if isinstance(amp, Number):
            self.amp = Constant(complex(amp))
        else:
            self.amp = amp

        self.sigma = sigma
        self.width = width

        ramp_duration = duration - width
        if rise and fall:
            gauss_duration = ramp_duration
        else:
            gauss_duration = ramp_duration * 2.

        def dummy_fn(t, args):
            return 0.

        if rise:
            self.t_plateau = gauss_duration / 2.
            self.gauss_rise = Gaussian(duration=gauss_duration, amp=self.amp, sigma=sigma,
                                       center=None, zero_ends=zero_ends)
            rise_edge = self.gauss_rise
        else:
            self.t_plateau = 0.
            self.gauss_rise = None
            rise_edge = dummy_fn

        if fall:
            fall_tzero = self.t_plateau + self.width - gauss_duration / 2.
            self.gauss_fall = Gaussian(duration=gauss_duration, amp=self.amp, sigma=sigma,
                                       center=None, zero_ends=zero_ends, tzero=fall_tzero)
            fall_tail = self.gauss_fall
        else:
            self.gauss_fall = None
            fall_tail = dummy_fn

        def fn(t, args):
            return npmod.where(
                t <= self.t_plateau,
                rise_edge(t, args),
                npmod.where(
                    t <= self.t_plateau + self.width,
                    self.amp.evaluate(args),
                    fall_tail(t, args)
                )
            )

        super().__init__(duration, fn, self.amp.parameters, tzero=tzero)

    def __str__(self):
        rise = self.gauss_rise is not None
        fall = self.gauss_fall is not None

        if rise:
            zero_ends = self.gauss_rise._pedestal != 0.
        else:
            zero_ends = self.gauss_fall._pedestal != 0.

        return (f'GaussianSquare(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, width={self.width}, '
                f' zero_ends={zero_ends}, rise={rise}, fall={fall})')


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
        amp: Union[float, complex, ParameterExpression],
        sigma: float,
        beta: float,
        center: Optional[float] = None,
        zero_ends: bool = True,
        tzero: float = 0.
    ):
        super().__init__(
            duration,
            amp=amp,
            sigma=sigma,
            center=center,
            zero_ends=zero_ends,
            tzero=tzero
        )

        self.beta = beta

        gauss_fn = self.fn
        npmod = config.npmod

        def fn(t, args):
            gauss = gauss_fn(t, args)
            dgauss = -(t - self.center) / npmod.square(self.sigma) * gauss

            return gauss + 1.j * self.beta * dgauss

        self.fn = fn

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
        amp: Union[float, complex, ParameterExpression],
        tzero: float = 0.
    ):
        if isinstance(amp, Number):
            self.amp = Constant(complex(amp))
        else:
            self.amp = amp

        npmod = config.npmod

        def fn(t, args):
            return npmod.full_like(t, self.amp.evaluate(args))

        super().__init__(duration, fn, self.amp.parameters, tzero=tzero)

    def __str__(self):
        return f'Square(duration={self.duration}, amp={self.amp})'
