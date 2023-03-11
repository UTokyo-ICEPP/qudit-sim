r"""
==========================================
Drive Hamiltonian (:mod:`qudit_sim.drive`)
==========================================

.. currentmodule:: qudit_sim.drive

See :ref:`drive-hamiltonian` for theoretical background.
"""

from typing import Callable, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from .time_function import Parameter, ParameterFn, TimeFunction
from .config import config

HamiltonianCoefficient = Union[str, ArrayType, TimeFunction]

@dataclass(frozen=True)
class DriveTerm:
    r"""Data class representing a drive.

    Args:
        frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
            that starts with SetFrequency.
        amplitude: Function :math:`r(t)`.
        constant_phase: The phase value of `amplitude` when it is a str or a callable and is known to have
            a constant phase. None otherwise.
    """
    frequency: Optional[Union[float, Parameter]] = None
    amplitude: Union[float, complex, str, np.ndarray, Parameter, Callable] = 1.+0.j
    constant_phase: Optional[Union[float, Parameter]] = None

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
        amplitude = self.amplitude
        if isinstance(amplitude, str):
            # If this is actually a static expression, convert to complex
            try:
                amplitude = eval(amplitude)
            except:
                pass

        if rwa:
            fn_x, fn_y = self._generate_fn_rwa(amplitude, frame_frequency, drive_base)
        else:
            fn_x, fn_y = self._generate_fn_full(amplitude, frame_frequency, drive_base)

        return fn_x, fn_y

    def _generate_fn_rwa(self, amplitude, frame_frequency, drive_base):
        if isinstance(self.frequency, Parameter):
            detuning = ParameterFn(lambda args: args[0] - frame_frequency, (self.frequency.name,))
            is_resonant = False
        else:
            detuning = self.frequency - frame_frequency
            is_resonant = np.isclose(detuning, 0.)

        if isinstance(amplitude, (float, complex, Parameter)):
            # static envelope
            envelope = amplitude * drive_base

            if is_resonant:
                if isinstance(amplitude, Parameter):
                    envelope = envelope.to_timefn()

                return envelope.real, envelope.imag

            elif (isinstance(detuning, ParameterFn) or isinstance(amplitude, Parameter)
                  or config.pulse_sim_solver == 'jax'):
                fun = envelope * exp_freq(-detuning)
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
            envelope = f'{drive_base} * ({amplitude})'

            if is_resonant:
                return f'({envelope}).real', f'({envelope}).imag'

            elif self.constant_phase is None:
                return (f'({envelope}).real * cos({detuning} * t) + ({envelope}).imag * sin({detuning} * t)',
                        f'({envelope}).imag * cos({detuning} * t) - ({envelope}).real * sin({detuning} * t)')

            else:
                phase = np.angle(drive_base) + self.constant_phase
                return (f'abs({envelope}) * cos({phase} - ({detuning} * t))',
                        f'abs({envelope}) * sin({phase} - ({detuning} * t))')

        elif isinstance(amplitude, np.ndarray):
            envelope = amplitude * drive_base

            if is_resonant:
                return envelope.real, envelope.imag

            else:
                fun = envelope * exp_freq(-detuning)
                return fun.real, fun.imag

        elif callable(amplitude):
            if not isinstance(amplitude, TimeFunction):
                amplitude = TimeFunction(amplitude)

            envelope = drive_base * amplitude

            if is_resonant:
                return envelope.real, envelope.imag

            elif self.constant_phase is None:
                fun = envelope * exp_freq(-detuning)
                return fun.real, fun.imag

            else:
                drive_base_phase = np.angle(drive_base)
                if isinstance(self.constant_phase, Parameter):
                    phase = ParameterFn(lambda args: drive_base_phase + args[0], (self.constant_phase.name,))
                else:
                    phase = drive_base_phase + self.constant_phase

                absf = abs(envelope)
                return absf * cos_freq(-detuning, phase), absf * sin_freq(-detuning, phase)

        else:
            raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

    def _generate_fn_full(self, amplitude, frame_frequency, drive_base):
        lab_frame = (frame_frequency == 0.)

        if isinstance(amplitude, (float, complex, Parameter)):
            # static envelope
            double_envelope = 2.* drive_base * amplitude

            if (isinstance(amplitude, Parameter) or isinstance(self.frequency, Parameter)
                or config.pulse_sim_solver == 'jax'):
                prefactor = (double_envelope.real * cos_freq(self.frequency)
                             + double_envelope.imag * sin_freq(self.frequency))
            else:
                prefactor_terms = []
                if double_envelope.real != 0.:
                    prefactor_terms.append(f'({double_envelope.real} * cos({self.frequency} * t))')
                if double_envelope.imag != 0.:
                    prefactor_terms.append(f'({double_envelope.imag} * sin({self.frequency} * t))')

                prefactor = ' + '.join(prefactor_terms)
                if len(prefactor_terms) > 1:
                    prefactor = f'({prefactor})'

        elif isinstance(amplitude, str):
            double_envelope = f'{2. * drive_base} * ({amplitude})'

            if self.constant_phase is None:
                prefactor = f'({double_envelope} * (cos({self.frequency} * t) - 1.j * sin({self.frequency} * t))).real'

            else:
                phase = np.angle(drive_base) + self.constant_phase
                prefactor = f'abs({double_envelope}) * cos({phase} - ({self.frequency} * t))'

        elif isinstance(amplitude, np.ndarray):
            double_envelope = 2. * drive_base * amplitude

            prefactor = (double_envelope * exp_freq(-self.frequency)).real

        elif callable(amplitude):
            if not isinstance(amplitude, TimeFunction):
                amplitude = TimeFunction(amplitude)

            double_envelope = 2. * drive_base * amplitude

            if self.constant_phase is None:
                prefactor = (double_envelope * exp_freq(-self.frequency)).real

            else:
                drive_base_phase = np.angle(drive_phase)
                if isinstance(self.constant_phase, Parameter):
                    phase = ParameterFn(lambda args: drive_base_phase + args[0], (self.constant_phase.name,))
                else:
                    phase = drive_base_phase + self.constant_phase

                absf = abs(double_envelope)
                prefactor = absf * cos_freq(negative_frequency, phase)

        else:
            raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

        if isinstance(prefactor, str):
            if lab_frame:
                return prefactor, ''
            else:
                return (f'{prefactor} * cos({frame_frequency} * t)',
                        f'{prefactor} * sin({frame_frequency} * t)')

        else:
            if lab_frame:
                return prefactor, None
            else:
                return prefactor * cos_freq(frame_frequency), prefactor * sin_freq(frame_frequency)


def cos_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> TimeFunction:
    """`cos(freq * t + phase)`"""
    return _mod_freq(freq, phase, config.npmod.cos)

def sin_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> TimeFunction:
    """`sin(freq * t + phase)`"""
    return _mod_freq(freq, phase, config.npmod.sin)

def exp_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> TimeFunction:
    """`cos(freq * t + phase) + 1.j * sin(freq * t + phase)`"""
    npmod = config.npmod
    return _mod_freq(freq, phase, lambda x: npmod.cos(x) + 1.j * npmod.sin(x))

def _mod_freq(_freq, _phase, _op):
    if isinstance(_freq, Parameter):
        if isinstance(_phase, Parameter):
            fn = TimeFunction(lambda t, args: _op(args[0] * t + args[1]),
                                     (_freq.name, _phase.name))
        elif isinstance(_phase, ParameterFn):
            fn = TimeFunction(lambda t, args: _op(args[0] * t + _phase(args[1:])),
                                     (_freq.name,) + _phase.parameters)
        else:
            fn = TimeFunction(lambda t, args: _op(args[0] * t + _phase),
                                     (_freq.name,))
    elif isinstance(_freq, ParameterFn):
        nparam = len(_freq.parameters)

        if isinstance(_phase, Parameter):
            fn = TimeFunction(lambda t, args: _op(_freq(args[0:nparam]) * t + args[nparam])
                                     _freq.parameters + (_phase.name,))
        elif isinstance(_phase, ParameterFn):
            fn = TimeFunction(lambda t, args: _op(_freq(args[0:nparam]) * t
                                                              + _phase(args[nparam:]))
                                     _freq.parameters + _phase.parameters)
        else:
            fn = TimeFunction(lambda t, args: _op(_freq(args) * t + _phase),
                                     _freq.parameters)
    else:
        if isinstance(_phase, Parameter):
            fn = TimeFunction(lambda t, args: _op(_freq * t + args[0]),
                                     (_phase.name,))
        elif isinstance(_phase, ParameterFn):
            fn = TimeFunction(lambda t, args: _op(_freq * t + _phase(args)),
                                     _phase.parameters)
        else:
            fn = TimeFunction(lambda t, args: _op(_freq * t + _phase))

    return fn
