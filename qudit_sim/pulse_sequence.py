r"""
================================================
Pulse sequence (:mod:`qudit_sim.pulse_sequence`)
================================================

.. currentmodule:: qudit_sim.pulse_sequence

Implementation of pulse sequence.
"""
import copy
import re
import warnings
from dataclasses import dataclass
from numbers import Number
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import jax.numpy as jnp
import numpy as np

from .drive import (CosFunction, Delay, ExpFunction, SetFrequency, SetPhase, ShiftFrequency,
                    ShiftPhase, SinFunction)
from .expression import (ArrayType, ConstantFunction, Expression, ParameterExpression, Parameter,
                         PiecewiseFunction, ReturnType, TimeFunction, TimeType, array_like)
from .pulse import Pulse

HamiltonianCoefficient = Union[str, ArrayType, TimeFunction]


class PulseSequence(list):
    """Pulse sequence.

    This class represents a sequence of instructions (pulse, delay, frequency/phase shift/set)
    given to a single channel.
    """
    @property
    def duration(self) -> float:
        d = sum(inst.value for inst in self if isinstance(inst, Delay))
        d += sum(inst.duration for inst in self if isinstance(inst, Pulse))
        return d

    @property
    def frequency(self) -> Union[float, ParameterExpression, None]:
        frequencies = set(inst.value for inst in self if isinstance(inst, SetFrequency))
        if len(frequencies) == 1:
            return frequencies.pop()
        else:
            # Unique frequency cannot be defined
            return None

    @property
    def amplitude(self) -> Union[float, complex, str, np.ndarray, Parameter, Callable, None]:
        if len(self) == 2:
            return self[1]
        else:
            # Unique amplitude cannot be defined
            return None

    def __str__(self) -> str:
        return f'[{", ".join(str(inst) for inst in self)}]'

    def max_frequency(self, args: Dict[str, Any] = {}) -> float:
        maxf = 0.
        for inst in self:
            if isinstance(inst, SetFrequency):
                if isinstance(inst.value, Parameter):
                    try:
                        freq = args[inst.value.name]
                    except KeyError:
                        raise ValueError(f'Value of {inst.value.name} not found in args')
                else:
                    freq = inst.value

                maxf = max(maxf, freq)

        return maxf

    def envelope(self, t: Union[float, np.ndarray], args: Dict[str, Any] = {}) -> np.ndarray:
        """Return the envelope of the sequence as a function of time.

        This function is mostly for visualization purposes. Phase and frequency information is lost
        in the returned array.

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
            elif isinstance(inst, TimeFunction):
                fn = copy.copy(inst)
                fn.tzero = time
                funclist.append((time, fn))
                time += inst.duration

        funclist.append((time, None))

        result = 0.
        for time, func in funclist[:-1]:
            if isinstance(func, TimeFunction):
                result = np.where(t > time, func(t, args), result)
            else:
                result = np.where(t > time, func, result)

        result = np.where(t > funclist[-1][0], 0., result)

        return result

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

        for inst in self:
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
            return (f'({envelope}).real * cos({detuning} * t)'
                    ' + ({envelope}).imag * sin({detuning} * t)',
                    f'({envelope}).imag * cos({detuning} * t)'
                    ' - ({envelope}).real * sin({detuning} * t)')


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
