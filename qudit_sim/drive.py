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

from .time_function import ParameterExpression, Parameter, TimeFunction
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
        detuning = self.frequency - frame_frequency

        if isinstance(self.frequency, Parameter):
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

            elif (isinstance(amplitude, Parameter) or isinstance(self.frequency, Parameter)
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
                fun = ExpFunction(-detuning) * envelope
                return fun.real, fun.imag

        elif callable(amplitude):
            if not isinstance(amplitude, TimeFunction):
                amplitude = TimeFunction(amplitude)

            envelope = amplitude * drive_base

            if is_resonant:
                return envelope.real, envelope.imag

            elif self.constant_phase is None:
                fun = envelope * ExpFunction(-detuning)
                return fun.real, fun.imag

            else:
                phase = self.constant_phase + np.angle(drive_base)
                absf = abs(envelope)
                return absf * CosFunction(-detuning, phase), absf * SinFunction(-detuning, phase)

        else:
            raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

    def _generate_fn_full(self, amplitude, frame_frequency, drive_base):
        lab_frame = (frame_frequency == 0.)

        if isinstance(amplitude, (float, complex, Parameter)):
            # static envelope
            double_envelope = amplitude * 2.* drive_base

            if (isinstance(amplitude, Parameter) or isinstance(self.frequency, Parameter)
                or config.pulse_sim_solver == 'jax'):
                prefactor = (double_envelope.real * CosFunction(self.frequency)
                             + double_envelope.imag * SinFunction(self.frequency))
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
            double_envelope = amplitude * 2. * drive_base

            if self.constant_phase is None:
                prefactor = f'(({double_envelope}) * (cos({self.frequency} * t) - 1.j * sin({self.frequency} * t))).real'

            else:
                phase = self.constant_phase + np.angle(drive_base)
                if isinstance(self.constant_phase, Parameter):
                    prefactor = CosFunction(-self.frequency, phase) * abs(double_envelope)
                else:
                    prefactor = f'abs({double_envelope}) * cos({phase} - ({self.frequency} * t))'

        elif isinstance(amplitude, np.ndarray):
            double_envelope = 2. * drive_base * amplitude

            prefactor = (double_envelope * ExpFunction(-self.frequency)).real

        elif callable(amplitude):
            if not isinstance(amplitude, TimeFunction):
                amplitude = TimeFunction(amplitude)

            double_envelope = amplitude * 2. * drive_base

            if self.constant_phase is None:
                prefactor = (double_envelope * ExpFunction(-self.frequency)).real

            else:
                phase = self.constant_phase + np.angle(drive_phase)
                absf = abs(double_envelope)
                prefactor = absf * CosFunction(-self.frequency, phase)

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
                return prefactor * CosFunction(frame_frequency), prefactor * SinFunction(frame_frequency)


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
