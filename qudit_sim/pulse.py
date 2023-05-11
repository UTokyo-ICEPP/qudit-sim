r"""
============================================
Pulse shape library (:mod:`qudit_sim.pulse`)
============================================

.. currentmodule:: qudit_sim.pulse

Classes in this module represent pulse envelopes. Subclasses of Pulse can be passed to
``HamiltonianBuilder.add_drive`` either as the ``amplitude`` parameter or through the PulseSequence
class (``sequence`` parameter).
"""

from numbers import Number
from types import ModuleType
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np

from .expression import Constant, ParameterExpression, ReturnType, TimeFunction, TimeType

class Pulse(TimeFunction):
    """Base class for all pulse shapes.

    Args:
        duration: Pulse duration.
    """

    def __init__(
        self,
        duration: float,
        fn: Callable[[TimeType, Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Tuple[str, ...] = (),
        tzero: float = 0.
    ):
        assert duration > 0., 'Pulse duration must be positive'
        self.duration = duration
        super().__init__(fn, parameters, tzero)

    def __str__(self) -> str:
        return f'Pulse(duration={self.duration}, tzero={self.tzero})'


class ScalablePulse(Pulse):
    """Pulse with parametrized amplitude."""
    def __init__(
        self,
        duration: float,
        amp: Union[float, complex, ParameterExpression],
        fn: Callable[[TimeType, Tuple[Any, ...], ModuleType], ReturnType],
        parameters: Tuple[str, ...] = (),
        tzero: float = 0.
    ):
        if isinstance(amp, Number):
            self._amp = Constant(complex(amp))
        else:
            self._amp = amp.copy()

        parameters = self._amp.parameters + parameters

        super().__init__(duration, fn, parameters, tzero)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        if isinstance(self._amp, Constant):
            return self._amp.evaluate()
        else:
            return self._amp

    def _scale(self, value: ReturnType, args: Tuple[Any, ...], npmod: ModuleType):
        return value * self._amp.evaluate(args, npmod)


class Gaussian(ScalablePulse):
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
        self.sigma = sigma

        if center is None:
            self.center = duration * 0.5
        else:
            self.center = center

        if zero_ends:
            assert np.isclose(self.center, duration * 0.5), \
                'Zero-ended Gaussian must be centered'

            x = duration * 0.5 / self.sigma
            self._pedestal = np.exp(-np.square(x) * 0.5)
        else:
            self._pedestal = 0.

        super().__init__(duration, amp, self._fn, tzero=tzero)

    def __str__(self) -> str:
        return (f'Gaussian(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, center={self.center},'
                f' zero_ends={self._pedestal != 0.}, tzero={self.tzero})')

    @property
    def zero_ends(self) -> bool:
        return self._pedestal != 0.

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        x_over_sqrt2 = (t - self.center) / self.sigma * np.sqrt(0.5)
        v = npmod.exp(-npmod.square(x_over_sqrt2))
        norm_val = npmod.asarray((1. - self._pedestal) * (v - self._pedestal), dtype='complex128')
        return self._scale(norm_val, args, npmod)

    def copy(self) -> 'Gaussian':
        return Gaussian(self.duration, self.amp, self.sigma, self.center, self.zero_ends, self.tzero)


class GaussianSquare(ScalablePulse):
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

        self.sigma = sigma
        self.width = width

        ramp_duration = duration - width
        if rise and fall:
            gauss_duration = ramp_duration
        else:
            gauss_duration = ramp_duration * 2.

        if rise:
            self.t_plateau = gauss_duration / 2.
            self.gauss_rise = Gaussian(duration=gauss_duration, amp=1., sigma=sigma,
                                       center=None, zero_ends=zero_ends)
        else:
            self.t_plateau = 0.
            self.gauss_rise = None

        if fall:
            fall_tzero = self.t_plateau + self.width - gauss_duration / 2.
            self.gauss_fall = Gaussian(duration=gauss_duration, amp=1., sigma=sigma,
                                       center=None, zero_ends=zero_ends, tzero=fall_tzero)
        else:
            self.gauss_fall = None

        if rise and fall:
            fn = self._fn_full
        elif rise:
            fn = self._fn_left
        elif fall:
            fn = self._fn_right

        super().__init__(duration, amp, fn, tzero=tzero)

    def __str__(self) -> str:
        rise = self.gauss_rise is not None
        fall = self.gauss_fall is not None

        if rise:
            zero_ends = self.gauss_rise._pedestal != 0.
        else:
            zero_ends = self.gauss_fall._pedestal != 0.

        return (f'GaussianSquare(duration={self.duration}, amp={self.amp}, sigma={self.sigma},'
                f' width={self.width}, zero_ends={zero_ends}, rise={rise}, fall={fall},'
                f' tzero={self.tzero})')

    @property
    def zero_ends(self) -> bool:
        return (self.gauss_rise and self.gauss_rise.zero_ends) or \
            (self.gauss_fall and self.gauss_fall.zero_ends)

    def _fn_full(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        t = npmod.asarray(t)

        value_left = npmod.asarray(t <= self.t_plateau) * (self.gauss_rise(t, npmod=npmod) - 1.)
        value_right = npmod.asarray(t > self.t_plateau + self.width) * (self.gauss_fall(t, npmod=npmod) - 1.)
        return self._scale((value_left + value_right + 1.), args, npmod)

    def _fn_left(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        t = npmod.asarray(t)

        value_left = npmod.asarray(t <= self.t_plateau) * (self.gauss_rise(t, npmod=npmod) - 1.)
        return self._scale(value_left + 1., args, npmod)

    def _fn_right(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        t = npmod.asarray(t)

        value_right = npmod.asarray(t > self.t_plateau + self.width) * (self.gauss_fall(t, npmod=npmod) - 1.)
        return self._scale(value_right + 1., args, npmod)

    def copy(self) -> 'GaussianSquare':
        return GaussianSquare(self.duration, self.amp, self.sigma, self.width, self.zero_ends,
                              self.gauss_rise is not None, self.gauss_fall is not None, self.tzero)


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
        beta: Union[float, ParameterExpression],
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

        if isinstance(beta, Number):
            self._beta = Constant(float(beta))
        else:
            self._beta = beta.copy()

        self.parameters += self._beta.parameters

    def __str__(self) -> str:
        return (f'Drag(duration={self.duration}, amp={self.amp}, sigma={self.sigma}, beta={self.beta}, '
                f'center={self.center}, zero_ends={self.zero_ends}, tzero={self.tzero})')

    @property
    def beta(self) -> Union[float, ParameterExpression]:
        if isinstance(self._beta, Constant):
            return self._beta.evaluate()
        else:
            return self._beta

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        t = npmod.asarray(t)

        gauss = super()._fn(t, args, npmod)
        dgauss = -(t - self.center) / npmod.square(self.sigma) * gauss

        beta = self._beta.evaluate(args[-len(self._beta.parameters):], npmod)

        return gauss + 1.j * beta * dgauss

    def copy(self) -> 'Drag':
        return Drag(self.duration, self.amp, self.sigma, self.beta, self.center, self.zero_ends,
                    self.tzero)


class Square(ScalablePulse):
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
        super().__init__(duration, amp, self._fn, tzero=tzero)

    def __str__(self) -> str:
        return f'Square(duration={self.duration}, amp={self.amp}, tzero={self.tzero})'

    def _fn(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self._scale(npmod.ones_like(t), args, npmod)

    def copy(self) -> 'Square':
        return Square(self.duration, self.amp, self.tzero)
