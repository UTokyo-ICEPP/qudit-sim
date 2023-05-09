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

HamiltonianCoefficient = Union[str, ArrayType, TimeFunction]


class DriveTerm:
    r"""Data class representing a drive.

    Args:
        frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
            that starts with SetFrequency.
        amplitude: Function :math:`r(t)`.
        sequence: Drive sequence. If this argument is present and not None, all other arguments are ignored,
            except when the sequence does not start with SetFrequency and frequency is not None.
    """
    def __init__(
        self,
        frequency: Optional[Union[float, Parameter]] = None,
        amplitude: Union[float, complex, str, np.ndarray, Parameter, Callable] = 1.+0.j,
        sequence: Optional[List[Any]] = None
    ):
        if sequence is None:
            if frequency is None or amplitude is None:
                raise RuntimeError('Frequency and amplitude must be set if not using a PulseSequence.')

            self._sequence = [SetFrequency(frequency), amplitude]
        else:
            self._sequence = list(sequence)
            if frequency is not None and not isinstance(self._sequence[0], SetFrequency):
                self._sequence.insert(0, SetFrequency(frequency))

    @property
    def frequency(self) -> Union[float, ParameterExpression, None]:
        frequencies = set(inst.value for inst in self._sequence if isinstance(inst, SetFrequency))
        if len(frequencies) == 1:
            return frequencies.pop()
        else:
            # Unique frequency cannot be defined
            return None

    @property
    def amplitude(self) -> Union[float, complex, str, np.ndarray, Parameter, Callable, None]:
        if len(self._sequence) == 2:
            return self._sequence[1]
        else:
            # Unique amplitude cannot be defined
            return None

    def generate_fn(
        self,
        frame_frequency: float,
        drive_base: complex,
        rwa: bool,
        as_timefn: bool = False
    ) -> Tuple[HamiltonianCoefficient, HamiltonianCoefficient]:
        r"""Generate the coefficients for X and Y drives.

        Args:
            frame_frequency: Frame frequency :math:`\xi_k^{l}`.
            drive_base: Factor :math:`\alpha_{jk} e^{i \rho_{jk}} \frac{\Omega_j}{2}`.
            rwa: If True, returns the RWA coefficients.
            as_timefn: If True, force all terms to be TimeFunctions.

        Returns:
            X and Y coefficient functions.
        """
        funclist = list()

        frequency = None
        phase_offset = 0.
        time = 0.

        if rwa:
            _generate_single = _generate_single_rwa
        else:
            _generate_single = _generate_single_full

        for inst in self._sequence:
            if isinstance(inst, ShiftFrequency):
                if frequency is None:
                    raise RuntimeError('ShiftFrequency called before SetFrequency')

                frequency += inst.value
            elif isinstance(inst, ShiftPhase):
                phase_offset += inst.value
            elif isinstance(inst, SetFrequency):
                frequency = inst.value
            elif isinstance(inst, SetPhase):
                if frequency is None:
                    raise RuntimeError('SetPhase called before SetFrequency')

                phase_offset = inst.value - frequency * time
            elif isinstance(inst, Delay):
                funclist.append((time, ConstantFunction(0.), ConstantFunction(0.)))
                time += inst.value
            else:
                if frequency is None:
                    raise RuntimeError('Pulse called before SetFrequency')

                envelope = _make_envelope(inst, drive_base, phase_offset, as_timefn)

                fn_x, fn_y = _generate_single(envelope, frequency, frame_frequency)

                funclist.append((time, fn_x, fn_y))

                if isinstance(inst, Pulse):
                    time += inst.duration
                else:
                    if time != 0. and re.search('[^a-zA-Z_]t[^a-zA-Z0-9_]', inst):
                        warnings.warn('Possibly time-dependent string amplitude in a sequence'
                                      ' detected; this is not supported.', UserWarning)

                    # Indefinite drive
                    time = np.inf
                    break

        funclist.append((time, None, None))

        if len(funclist) == 1:
            raise ValueError('No drive amplitude specified')

        elif len(funclist) == 2:
            fn_x = funclist[0][1]
            fn_y = funclist[0][2]

        elif all(isinstance(func, TimeFunction) for _, func, _ in funclist[:-1]):
            timelist = list(f[0] for f in funclist)
            xlist = list(f[1] for f in funclist[:-1])
            ylist = list(f[2] for f in funclist[:-1])

            fn_x = PiecewiseFunction(timelist, xlist)
            fn_y = PiecewiseFunction(timelist, ylist)

        elif all(isinstance(func, np.ndarray) for _, func, _ in funclist[:-1]):
            fn_x = np.concatenate(list(x for _, x, _ in funclist[:-1]))
            fn_y = np.concatenate(list(y for _, _, y in funclist[:-1]))

        else:
            raise ValueError('Cannot generate a Hamiltonian coefficient from amplitude types'
                             f' {list(type(func) for _, func, _ in funclist[:-1])}')

        return fn_x, fn_y


def _make_envelope(inst, drive_base, phase_offset, as_timefn):
    if isinstance(phase_offset, Number):
        phase_factor = np.exp(-1.j * phase_offset)
    else:
        phase_factor = (phase_offset * (-1.j)).exp()

    if isinstance(inst, str):
        # If this is actually a static expression, convert to complex
        try:
            inst = complex(eval(inst))
        except:
            pass

    if isinstance(inst, (float, complex, Parameter)):
        # static envelope
        envelope = phase_factor * drive_base * inst
        if as_timefn:
            envelope = ConstantFunction(envelope)

    elif isinstance(inst, str):
        if as_timefn:
            raise TypeError(f'String amplitude expression {inst} cannot be converted'
                            ' to a TimeFunction')
        if isinstance(phase_factor, ParameterExpression):
            raise TypeError(f'String amplitude expression {inst} not compatible with'
                            ' parameterized phase offset')

        envelope = f'({phase_factor * drive_base}) * ({inst})'

    elif isinstance(inst, np.ndarray):
        envelope = phase_factor * drive_base * inst
        if as_timefn:
            raise TypeError(f'Array amplitude cannot be converted to a TimeFunction')

    elif callable(inst):
        if not isinstance(inst, TimeFunction):
            inst = TimeFunction(inst)

        envelope = inst * (phase_factor * drive_base)

    else:
        raise TypeError(f'Unsupported amplitude type f{type(inst)}')

    return envelope


def _generate_single_rwa(envelope, frequency, frame_frequency):
    detuning = frequency - frame_frequency

    if isinstance(frequency, ParameterExpression):
        is_resonant = False
    else:
        is_resonant = np.isclose(detuning, 0.)

    if isinstance(detuning, ParameterExpression) or isinstance(envelope, (np.ndarray, Expression)):
        if is_resonant:
            if isinstance(envelope, Expression) and not isinstance(envelope, TimeFunction):
                envelope = ConstantFunction(envelope)

            return envelope.real, envelope.imag
        else:
            fun = ExpFunction(-detuning) * envelope
            return fun.real, fun.imag

    elif isinstance(envelope, (float, complex)):
        if is_resonant:
            return envelope.real, envelope.imag
        else:
            fn_x = []
            fn_y = []
            if envelope.real != 0.:
                fn_x.append(f'({envelope.real} * cos({detuning} * t))')
                fn_y.append(f'({-envelope.real} * sin({detuning} * t))')
            if envelope.imag != 0.:
                fn_x.append(f'({envelope.imag} * sin({detuning} * t))')
                fn_y.append(f'({envelope.imag} * cos({detuning} * t))')

            return ' + '.join(fn_x), ' + '.join(fn_y)

    else: # str
        if is_resonant:
            return f'({envelope}).real', f'({envelope}).imag'
        else:
            return (f'({envelope}).real * cos({detuning} * t) + ({envelope}).imag * sin({detuning} * t)',
                    f'({envelope}).imag * cos({detuning} * t) - ({envelope}).real * sin({detuning} * t)')


def _generate_single_full(envelope, frequency, frame_frequency):
    if isinstance(frequency, ParameterExpression) or isinstance(envelope, (np.ndarray, Expression)):
        envelope *= 2.
        labframe_fn = (ExpFunction(-frequency) * envelope).real

        if frame_frequency == 0.:
            return labframe_fn, ConstantFunction(0.)
        else:
            return labframe_fn * CosFunction(frame_frequency), labframe_fn * SinFunction(frame_frequency)

    else:
        if isinstance(envelope, (float, complex)):
            envelope *= 2.

            labframe_fn_terms = []
            if envelope.real != 0.:
                labframe_fn_terms.append(f'({envelope.real} * cos({frequency} * t))')
            if envelope.imag != 0.:
                labframe_fn_terms.append(f'({envelope.imag} * sin({frequency} * t))')

            labframe_fn = ' + '.join(labframe_fn_terms)
            if len(labframe_fn_terms) > 1:
                labframe_fn = f'({labframe_fn})'

        else: # str
            labframe_fn = f'(2. * ({envelope}) * (cos({frequency} * t) - 1.j * sin({frequency} * t))).real'

        if frame_frequency == 0.:
            return labframe_fn, ''
        else:
            return (f'{labframe_fn} * cos({frame_frequency} * t)',
                    f'{labframe_fn} * sin({frame_frequency} * t)')


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
