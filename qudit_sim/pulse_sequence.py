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
        instlist = self._make_instlist()

        tlist = list()
        flist = list()
        for time, frequency, phase_offset, inst in instlist:
            envelope = _make_envelope(inst, 1., phase_offset, True)


            tlist.append(time)
            flist.append(envelope)
            ylist.append(fn_y)
        funclist = list()

        frequency = None
        phase_offset = 0.
        time = 0.

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
                funclist.append((time, 0.))
                time += inst.value
            else:
                if frequency is None:
                    raise RuntimeError('Pulse called before SetFrequency')

                envelope = _make_envelope(inst, 1., phase_offset, False)
                funclist.append((time, envelope))

                if isinstance(inst, Pulse):
                    time += inst.duration
                else:
                    if time != 0. and isinstance(inst, str) and \
                        re.search('[^a-zA-Z_]t[^a-zA-Z0-9_]', inst):
                        warnings.warn('Possibly time-dependent string amplitude in a sequence'
                                      ' detected; this is not supported.', UserWarning)
                    # Indefinite drive
                    time = np.inf
                    break

        funclist.append((time, None, None))

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
        instlist = self._make_instlist()

        tlist = list()
        xlist = list()
        ylist = list()
        for time, frequency, phase_offset, inst in instlist:
            envelope = _make_envelope(inst, drive_base, phase_offset, as_timefn)

            if frequency is None:
                # End of list or the very specific case of initial Delay without SetFrequency
                # -> envelope is 0 or ConstantFunction(0)
                fn_x = fn_y = envelope
            elif rwa:
                fn_x, fn_y = _modulate_rwa(envelope, frequency, frame_frequency)
            else:
                fn_x, fn_y = _modulate(envelope, frequency, frame_frequency)

            tlist.append(time)
            xlist.append(fn_x)
            ylist.append(fn_y)

        if len(tlist) == 1:
            raise ValueError('No drive amplitude specified')

        if frame_frequency != 0. or rwa:
            flists = [xlist, ylist]
        else:
            # ylist is a list of Nones
            flists = [xlist]

        fns = []
        for flist in flists:
            if len(tlist) == 2:
                fns.append(flist[0])
            elif all(isinstance(func, TimeFunction) for func in flist[:-1]):
                if (all(isinstance(func, ConstantFunction) for func in flist[:-1])
                    and np.allclose(list(func.value for func in flist[:-1]), flist[0].value)):
                    fn = ConstantFunction(flist[0].value)
                else:
                    fn = PiecewiseFunction(tlist, flist)

                fns.append(fn)
            elif all(isinstance(func, np.ndarray) for func in flist[:-1]):
                fns.append(np.concatenate(flist[:-1]))
            else:
                raise ValueError('Cannot generate a Hamiltonian coefficient from amplitude types'
                                 f' {list(type(func) for func in flist[:-1])}')

        if len(fns) == 1:
            fns.append(None)

        return tuple(fns)

    def _make_instlist(self):
        instlist = list()

        frequency = None
        phase_offset = 0.
        time = 0.

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
                instlist.append((time, frequency, phase_offset, 0.))
                time += inst.value
            else:
                if frequency is None:
                    raise RuntimeError('Pulse called before SetFrequency')

                if isinstance(inst, Pulse):
                    instlist.append((time, frequency, phase_offset, inst.shift(time)))
                    time += inst.duration
                else:
                    if time != 0. and isinstance(inst, str) and \
                        re.search('[^a-zA-Z_]t[^a-zA-Z0-9_]', inst):
                        warnings.warn('Possibly time-dependent string amplitude in a sequence'
                                      ' detected; this is not supported.', UserWarning)

                    instlist.append((time, frequency, phase_offset, inst))
                    # Indefinite drive
                    time = np.inf
                    break

        instlist.append((time, None, 0., 0.))

        return instlist


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
        if inst == 0.:
            envelope = 0.
        else:
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


def _modulate_rwa(envelope, frequency, frame_frequency):
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
                    f' + ({envelope}).imag * sin({detuning} * t)',
                    f'({envelope}).imag * cos({detuning} * t)'
                    f' - ({envelope}).real * sin({detuning} * t)')


def _modulate(envelope, frequency, frame_frequency):
    if isinstance(frequency, ParameterExpression) or isinstance(envelope, (np.ndarray, Expression)):
        envelope *= 2.
        labframe_fn = (ExpFunction(-frequency) * envelope).real

        if frame_frequency == 0.:
            return labframe_fn, None
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
            return labframe_fn, None
        else:
            return (f'{labframe_fn} * cos({frame_frequency} * t)',
                    f'{labframe_fn} * sin({frame_frequency} * t)')
