r"""
==========================================
Drive Hamiltonian (:mod:`qudit_sim.drive`)
==========================================

.. currentmodule:: qudit_sim.drive

See :ref:`drive-hamiltonian` for theoretical background.
"""

from typing import Callable, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
import numpy as np

from .time_function import ParameterExpression, Parameter, TimeFunction
from .config import config

HamiltonianCoefficient = Union[str, ArrayType, TimeFunction]

class DriveTerm:
    r"""Data class representing a drive.

    Args:
        frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
            that starts with SetFrequency.
        amplitude: Function :math:`r(t)`.
        constant_phase: The phase value of `amplitude` when it is a str or a callable and is known to have
            a constant phase. None otherwise.
        sequence: Drive sequence. If this argument is present and not None, all other arguments are ignored.
    """
    def __init__(
        self,
        frequency: Optional[Union[float, Parameter]] = None,
        amplitude: Union[float, complex, str, np.ndarray, Parameter, Callable] = 1.+0.j,
        constant_phase: Optional[Union[float, Parameter]] = None,
        sequence: Optional[List[Any]] = None
    ):
        if sequence is None:
            self._sequence = [SetFrequency(frequency), amplitude]
            self.constant_phase = constant_phase
        else:
            self._sequence = sequence

    def generate_fn(
        self,
        frame_frequency: float,
        drive_base: complex,
        rwa: bool
    ) -> Tuple[HamiltonianCoefficient, Union[HamiltonianCoefficient, None]]:
        r"""Generate the coefficients for X and Y drives.

        Args:
            frame_frequency: Frame frequency :math:`\xi_k^{l}`.
            drive_base: Factor :math:`\alpha_{jk} e^{i \rho_{jk}} \frac{\Omega_j}{2}`.
            rwa: If True, returns the RWA coefficients.

        Returns:
            X and Y coefficient functions.
        """
        funclist = list()

        frequency = None
        phase_offset = 0.
        time = 0.

        if rwa:
            generate_single = _generate_single_rwa
        else:
            generate_single = _generate_single_full

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

                if isinstance(inst, str):
                    # If this is actually a static expression, convert to complex
                    try:
                        inst = complex(eval(inst))
                    except:
                        pass

                fn_x, fn_y = generate_single(inst, frequency, frame_frequency, drive_base * np.exp(-1.j * phase_offset),
                                             time, self.constant_phase)
                funclist.append((time, fn_x, fn_y))

                if isinstance(inst, Pulse):
                    time += inst.duration
                else:
                    # Indefinite drive
                    time = np.inf
                    break

        timelist.append((time, None, None))

        if len(funclist) == 1:
            raise ValueError('No drive amplitude specified')

        elif len(funclist) == 2:
            fn_x = funclist[0][1]
            fn_y = funclist[0][2]

        elif all(isinstance(func, TimeFunction) for _, func, _ in funclist[:-1]):
            timelist = list(t for t, _, _ in funclist)
            xlist = list(x for _, x, _ in funclist[:-1])
            ylist = list(y for _, _, y in funclist[:-1])

            npmod = config.npmod

            def fn_x(t, args):
                result = 0.
                for time, func, _ in funclist[:-1]:
                    result = npmod.where(t > time, func(t, args), result)
                result = npmod.where(t > timelist[-1], 0., result)
                return result

            def fn_y(t, args):
                result = 0.
                for time, _, func in funclist[:-1]:
                    result = npmod.where(t > time, func(t, args), result)
                result = npmod.where(t > timelist[-1], 0., result)
                return result

        elif all(isinstance(func, np.ndarray) for _, func, _ in funclist[:-1]):
            fn_x = np.concatenate(list(x for _, x, _ in funclist[:-1]))
            fn_y = np.concatenate(list(y for _, _, y in funclist[:-1]))

        else:
            raise ValueError('Cannot generate a Hamiltonian coefficient from amplitude types'
                             f' {list(type(func) for _, func, _ in funclist[:-1])}')

        return fn_x, fn_y


def _generate_single_rwa(self, amplitude, frequency, frame_frequency, drive_base, tzero, constant_phase=None):
    detuning = frequency - frame_frequency

    if isinstance(frequency, Parameter):
        is_resonant = False
    else:
        is_resonant = np.isclose(detuning, 0.)

    if isinstance(amplitude, (float, complex, Parameter)):
        # static envelope
        envelope = amplitude * drive_base

        if is_resonant:
            if isinstance(envelope, Parameter):
                envelope = ConstantFunction(envelope)

            return envelope.real, envelope.imag

        elif (isinstance(amplitude, Parameter) or isinstance(frequency, Parameter)
              or config.pulse_sim_solver == 'jax'):
            fun = ExpFunction(-detuning) * envelope
            return fun.real, fun.imag

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

    elif isinstance(amplitude, str):
        if tzero != 0. and 't' in amplitude:
            warnings.warn('Possibly time-dependent string amplitude in a sequence detected; this is not supported.',
                          UserWarning)

        envelope = f'({drive_base}) * ({amplitude})'

        if is_resonant:
            return f'({envelope}).real', f'({envelope}).imag'

        elif constant_phase is None:
            return (f'({envelope}).real * cos({detuning} * t) + ({envelope}).imag * sin({detuning} * t)',
                    f'({envelope}).imag * cos({detuning} * t) - ({envelope}).real * sin({detuning} * t)')

        else:
            phase = np.angle(drive_base) + constant_phase
            return (f'abs({envelope}) * cos({phase} - ({detuning} * t))',
                    f'abs({envelope}) * sin({phase} - ({detuning} * t))')

    elif isinstance(amplitude, np.ndarray):
        envelope = amplitude * drive_base

        if is_resonant:
            return envelope.real, envelope.imag

        else:
            fun = ExpFunction(-detuning) * envelope
            return fun.real, fun.imag

    elif callable(amplitude):
        if not isinstance(amplitude, TimeFunction):
            amplitude = TimeFunction(amplitude)

        if tzero != 0.:
            envelope = copy.copy(amplitude * drive_base)
            envelope.tzero = tzero

        if is_resonant:
            return envelope.real, envelope.imag

        elif constant_phase is None:
            fun = envelope * ExpFunction(-detuning)
            return fun.real, fun.imag

        else:
            phase = constant_phase + np.angle(drive_base)
            absf = abs(envelope)
            return absf * CosFunction(-detuning, phase), absf * SinFunction(-detuning, phase)

    else:
        raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')


def _generate_single_full(self, amplitude, frequency, frame_frequency, drive_base, tzero, constant_phase=None):
    if isinstance(amplitude, (float, complex, Parameter)):
        # static envelope
        double_envelope = amplitude * 2. * drive_base

        if (isinstance(amplitude, Parameter) or isinstance(frequency, Parameter)
            or config.pulse_sim_solver == 'jax'):
            labframe_fn = (double_envelope.real * CosFunction(frequency)
                           + double_envelope.imag * SinFunction(frequency))
        else:
            labframe_fn_terms = []
            if double_envelope.real != 0.:
                labframe_fn_terms.append(f'({double_envelope.real} * cos({frequency} * t))')
            if double_envelope.imag != 0.:
                labframe_fn_terms.append(f'({double_envelope.imag} * sin({frequency} * t))')

            labframe_fn = ' + '.join(labframe_fn_terms)
            if len(labframe_fn_terms) > 1:
                labframe_fn = f'({labframe_fn})'

    elif isinstance(amplitude, str):
        if tzero != 0. and 't' in amplitude:
            warnings.warn('Possibly time-dependent string amplitude in a sequence detected; this is not supported.',
                          UserWarning)

        double_envelope = f'({2. * drive_base}) * ({amplitude})'

        if constant_phase is None:
            labframe_fn = f'({double_envelope} * (cos({frequency} * t) - 1.j * sin({frequency} * t))).real'

        else:
            phase = constant_phase + np.angle(drive_base)
            if isinstance(constant_phase, Parameter):
                labframe_fn = CosFunction(-frequency, phase) * abs(double_envelope)
            else:
                labframe_fn = f'abs({double_envelope}) * cos({phase} - ({frequency} * t))'

    elif isinstance(amplitude, np.ndarray):
        double_envelope = amplitude * 2. * drive_base

        labframe_fn = (double_envelope * ExpFunction(-frequency)).real

    elif callable(amplitude):
        if not isinstance(amplitude, TimeFunction):
            amplitude = TimeFunction(amplitude)

        double_envelope = amplitude * 2. * drive_base

        if tzero != 0.:
            double_envelope = copy.copy(double_envelope)
            double_envelope.tzero = tzero

        if constant_phase is None:
            labframe_fn = (double_envelope * ExpFunction(-frequency)).real

        else:
            phase = constant_phase + np.angle(drive_phase)
            absf = abs(double_envelope)
            labframe_fn = absf * CosFunction(-frequency, phase)

    else:
        raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

    if isinstance(labframe_fn, str):
        if frame_frequency == 0.:
            return labframe_fn, ConstantFunction(0.)
        else:
            return (f'{labframe_fn} * cos({frame_frequency} * t)',
                    f'{labframe_fn} * sin({frame_frequency} * t)')

    else:
        if frame_frequency == 0.:
            return labframe_fn, ConstantFunction(0.)
        else:
            return labframe_fn * CosFunction(frame_frequency), labframe_fn * SinFunction(frame_frequency)


class ConstantFunction(TimeFunction):
    def __init__(
        self,
        value: ParameterExpression
    ):
        def fn(t, args):
            return value.evaluate(args)

        super().__init__(fn, value.parameters)


class OscillationFunction(TimeFunction):
    def __init__(
        self,
        op: Callable,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        if isinstance(frequency, ParameterExpression):
            if isinstance(phase, ParameterExpression):
                freq_n_params = len(frequency.parameters)
                def fn(t, args):
                    return op(frequency.evaluate(args[:freq_n_params]) * t
                              + phase.evaluate(args[freq_n_params:]))

                super().__init__(fn, frequency.parameters + phase.parameters)
            else:
                def fn(t, args):
                    return op(frequency.evaluate(args) * t + phase)

                super().__init__(fn, frequency.parameters)
        else:
            if isinstance(phase, ParameterExpression):
                def fn(t, args):
                    return op(frequency * t + phase.evaluate(args))

                super().__init__(fn, phase.parameters)
            else:
                def fn(t, args=()):
                    return op(frequency * t + phase)

                super().__init__(fn, ())

def CosFunction(OscillationFunction):
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(config.npmod.cos, frequency, phase)

def SinFunction(OscillationFunction):
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(config.npmod.cos, frequency, phase)

def ExpFunction(OscillationFunction):
    @staticmethod
    def _op(x):
        return config.npmod.cos(x) + 1.j * config.npmod.sin(x)

    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        phase: Union[float, ParameterExpression] = 0.
    ):
        super().__init__(self._op, frequency, phase)

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
