r"""
==========================================
Drive Hamiltonian (:mod:`qudit_sim.drive`)
==========================================

.. currentmodule:: qudit_sim.drive

See :ref:`drive-hamiltonian` for theoretical background.
"""
import copy
import re
import warnings
from dataclasses import dataclass
from numbers import Number
from types import ModuleType
from typing import Callable, Optional, Union, Tuple, List, Any
import jax.numpy as jnp
import numpy as np

from .expression import (ArrayType, ConstantFunction, Expression, ParameterExpression, Parameter,
                         PiecewiseFunction, ReturnType, TimeFunction, TimeType, array_like)
from .pulse import Pulse


class OscillationFunction(TimeFunction):
    def __init__(
        self,
        op: Callable[[array_like, ModuleType], ReturnType],
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        self.op = op
        self.frequency = frequency
        self.phase = phase

        if isinstance(frequency, ParameterExpression):
            if isinstance(phase, ParameterExpression):
                fn = self._fn_PE_PE
                parameters = frequency.parameters + phase.parameters
            else:
                fn = self._fn_PE_float
                parameters = frequency.parameters
        else:
            if isinstance(phase, ParameterExpression):
                fn = self._fn_float_PE
                parameters = phase.parameters
            else:
                fn = self._fn_float_float
                parameters = ()

        super().__init__(fn, parameters)

    def _fn_PE_PE(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        freq_n_params = len(self.frequency.parameters)
        return self.op(self.frequency.evaluate(args[:freq_n_params], npmod) * t
                       + self.phase.evaluate(args[freq_n_params:], npmod),
                       npmod)

    def _fn_PE_float(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self.op(self.frequency.evaluate(args, npmod) * t + self.phase, npmod)

    def _fn_float_PE(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self.op(self.frequency * t + self.phase.evaluate(args, npmod), npmod)

    def _fn_float_float(
        self,
        t: TimeType,
        args: Tuple[Any, ...] = (),
        npmod: ModuleType = np
    ) -> ReturnType:
        return self.op(self.frequency * t + self.phase, npmod)


class CosFunction(OscillationFunction):
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(lambda x, npmod: npmod.cos(x), frequency, phase)

    def __str__(self) -> str:
        return f'cos({self.frequency} * {self._targ()} + {self.phase})'

    def __repr__(self) -> str:
        return f'CosFunction({repr(self.frequency)}, {repr(self.phase)})'


class SinFunction(OscillationFunction):
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(lambda x, npmod: npmod.sin(x), frequency, phase)

    def __str__(self) -> str:
        return f'sin({self.frequency} * {self._targ()} + {self.phase})'

    def __repr__(self) -> str:
        return f'SinFunction({repr(self.frequency)}, {repr(self.phase)})'


class ExpFunction(OscillationFunction):
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(lambda x, npmod: npmod.cos(x) + 1.j * npmod.sin(x), frequency, phase)

    def __str__(self) -> str:
        return f'exp(1j * ({self.frequency} * {self._targ()} + {self.phase}))'

    def __repr__(self) -> str:
        return f'ExpFunction({repr(self.frequency)}, {repr(self.phase)})'


@dataclass(frozen=True)
class ShiftFrequency:
    """Frequency shift in rad/s."""
    value: Union[float, Parameter]

@dataclass(frozen=True)
class ShiftPhase:
    """Phase shift (virtual Z)."""
    value: float

@dataclass(frozen=True)
class SetFrequency:
    """Frequency setting in rad/s."""
    value: Union[float, Parameter]

@dataclass(frozen=True)
class SetPhase:
    """Phase setting."""
    value: float

@dataclass(frozen=True)
class Delay:
    """Delay in seconds."""
    value: float
