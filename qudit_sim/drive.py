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
import jax.numpy as jnp

from .config import config

ArrayType = Union[np.ndarray, jnp.ndarray]
# Type for the callable time-dependent Hamiltonian coefficient
TimeType = Union[float, ArrayType]
ArgsType = Union[Dict[str, Any], Tuple[Any, ...]]
ReturnType = Union[complex, ArrayType]

@dataclass(frozen=True)
class Parameter:
    name: str

@dataclass
class ParameterFn:
    fn: Callable[[Tuple[Any, ...]], ReturnType]
    parameters: Tuple[str, ...]

    def __call__(self, args: ArgsType) -> ReturnType:
        if isinstance(args, dict):
            args = tuple(args[key] for key in self.parameters)

        return self.fn(args)

@dataclass
class CallableCoefficient:
    fn: Callable[[TimeType, Tuple[Any, ...]], ReturnType]
    parameters: Tuple[str, ...] = ()

    def __call__(self, t: TimeType, args: ArgsType) -> ReturnType:
        if isinstance(args, dict):
            args = tuple(args[key] for key in self.parameters)

        return self.fn(t, args)

HamiltonianCoefficient = Union[str, ArrayType, CallableCoefficient]


def cos_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> CallableCoefficient:
    """`cos(freq * t + phase)`"""
    return _mod_freq(freq, phase, config.npmod.cos)

def sin_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> CallableCoefficient:
    """`sin(freq * t + phase)`"""
    return _mod_freq(freq, phase, config.npmod.sin)

def exp_freq(
    freq: Union[float, Parameter, ParameterFn],
    phase: Union[float, Parameter, ParameterFn] = 0.
) -> CallableCoefficient:
    """`cos(freq * t + phase) + 1.j * sin(freq * t + phase)`"""
    npmod = config.npmod
    return _mod_freq(freq, phase, lambda x: npmod.cos(x) + 1.j * npmod.sin(x))

def _mod_freq(_freq, _phase, _op):
    if isinstance(_freq, Parameter):
        if isinstance(_phase, Parameter):
            fn = CallableCoefficient(lambda t, args: _op(args[0] * t + args[1]),
                                     (_freq.name, _phase.name))
        elif isinstance(_phase, ParameterFn):
            fn = CallableCoefficient(lambda t, args: _op(args[0] * t + _phase(args[1:])),
                                     (_freq.name,) + _phase.parameters)
        else:
            fn = CallableCoefficient(lambda t, args: _op(args[0] * t + _phase),
                                     (_freq.name,))
    elif isinstance(_freq, ParameterFn):
        nparam = len(_freq.parameters)

        if isinstance(_phase, Parameter):
            fn = CallableCoefficient(lambda t, args: _op(_freq(args[0:nparam]) * t + args[nparam])
                                     _freq.parameters + (_phase.name,))
        elif isinstance(_phase, ParameterFn):
            fn = CallableCoefficient(lambda t, args: _op(_freq(args[0:nparam]) * t
                                                              + _phase(args[nparam:]))
                                     _freq.parameters + _phase.parameters)
        else:
            fn = CallableCoefficient(lambda t, args: _op(_freq(args) * t + _phase),
                                     _freq.parameters)
    else:
        if isinstance(_phase, Parameter):
            fn = CallableCoefficient(lambda t, args: _op(_freq * t + args[0]),
                                     (_phase.name,))
        elif isinstance(_phase, ParameterFn):
            fn = CallableCoefficient(lambda t, args: _op(_freq * t + _phase(args)),
                                     _phase.parameters)
        else:
            fn = CallableCoefficient(lambda t, args: _op(_freq * t + _phase))

    return fn

def constant_function(
    value: Union[float, complex, Parameter, ParameterFn]
) -> CallableCoefficient:
    """`value`"""
    if isinstance(value, Parameter):
        fn = CallableCoefficient(lambda t, args: args[0], (value.name,))

    elif isinstance(value, ParameterFn):
        nparams = len(value.parameters)
        fn = CallableCoefficient(lambda t, args: value(args), value.parameters)

    else:
        fn = CallableCoefficient(lambda t, args: value)

    return fn

def scaled_function(
    scale: Union[float, complex, Parameter],
    fn: CallableCoefficient
) -> CallableCoefficient:
    """`scale * fn(t, args)`"""
    if isinstance(scale, Parameter):
        scaled_fn = CallableCoefficient(lambda t, args: args[0] * fn(t, args[1:]),
                                        (scale.name,) + fn.parameters)
    else:
        scaled_fn = CallableCoefficient(lambda t, args: scale * fn(t, args), fn.parameters)

    return scaled_fn

def prod_function(
    fn1: CallableCoefficient,
    fn2: CallableCoefficient
) -> CallableCoefficient:
    """`fn1(t, args) * fn2(t, args)`"""
    return _binary_function(fn1, fn2, config.npmod.multiply)

def sum_function(
    fn1: CallableCoefficient,
    fn2: CallableCoefficient
) -> CallableCoefficient:
    """`fn1(t, args) + fn2(t, args)`"""
    return _binary_function(fn1, fn2, config.npmod.add)

def diff_function(
    fn1: CallableCoefficient,
    fn2: CallableCoefficient
) -> CallableCoefficient:
    """`fn1(t, args) - fn2(t, args)`"""
    return _binary_function(fn1, fn2, config.npmod.subtract)

def _binary_function(_fn1, _fn2, _op):
    nparam1 = len(_fn1.parameters)
    return CallableCoefficient(lambda t, args: _op(_fn1(t, args[:nparam1]), _fn2(t, args[nparam1:])),
                               _fn1.parameters + _fn2.parameters)

def conj_function(
    fn: CallableCoefficient
) -> CallableCoefficient:
    """`fn(t, args).conjugate()`"""
    return _unary_function(fn, config.npmod.conjugate)

def real_function(fn):
    """`fn(t, args).real`"""
    return _unary_function(fn, config.npmod.real)

def imag_function(fn):
    """`fn(t, args).imag`"""
    return _unary_function(fn, config.npmod.imag)

def abs_function(fn):
    """`abs(fn(t, args))`"""
    return _unary_function(fn, config.npmod.abs)

def _unary_function(_fn, _op):
    """`fn(t, args).conjugate()`"""
    return CallableCoefficient(lambda t, args: _op(_fn(t, args)), _fn.parameters)


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
            negative_detuning = ParameterFn(lambda args: frame_frequency - args[0], (self.frequency.name,))

            is_resonant = False
        else:
            detuning = self.frequency - frame_frequency
            negative_detuning = -detuning
            is_resonant = np.isclose(detuning, 0.)

        if isinstance(amplitude, (float, complex, Parameter)):
            # static envelope
            if is_resonant:
                if isinstance(amplitude, Parameter):
                    envelope = scaled_function(drive_base, constant_function(amplitude))
                    return real_function(envelope), imag_function(envelope)
                else:
                    envelope = amplitude * drive_base
                    return envelope.real, envelope.imag

            elif (isinstance(detuning, ParameterFn) or isinstance(amplitude, Parameter)
                  or config.pulse_sim_solver == 'jax'):
                cos_function = scaled_function(drive_base,
                                               scaled_function(amplitude,
                                                               cos_freq(detuning)))
                sin_function = scaled_function(drive_base,
                                               scaled_function(amplitude,
                                                               sin_freq(detuning)))
                fn_x = sum_function(
                    real_function(cos_function),
                    imag_function(sin_function)
                )
                fn_y = diff_function(
                    imag_function(cos_function),
                    real_function(sin_function)
                )

                return fn_x, fn_y

            else:
                envelope = amplitude * drive_base
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
                fun = scaled_function(envelope, exp_freq(negative_detuning))
                return real_function(fun), imag_function(fun)

        elif callable(amplitude):
            envelope = scaled_function(drive_base, amplitude)

            if is_resonant:
                return real_function(envelope), imag_function(envelope)

            elif self.constant_phase is None:
                cos_function = prod_function(envelope, cos_freq(detuning))
                sin_function = prod_function(envelope, sin_freq(detuning))

                fn_x = sum_function(
                    real_function(cos_function),
                    imag_function(sin_function)
                )
                fn_y = diff_function(
                    imag_function(cos_function),
                    real_function(sin_function)
                )

                return fn_x, fn_y

            else:
                drive_base_phase = np.angle(drive_base)
                if isinstance(self.constant_phase, Parameter):
                    phase = ParameterFn(lambda args: drive_base_phase + args[0], (self.constant_phase.name,))
                else:
                    phase = drive_base_phase + self.constant_phase

                absf = abs_function(envelope)
                return (prod_function(absf, cos_freq(negative_detuning, phase)),
                        prod_function(absf, sin_freq(negative_detuning, phase)))

        else:
            raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

    def _generate_fn_full(self, amplitude, frame_frequency, drive_base):
        lab_frame = (frame_frequency == 0.)

        if isinstance(self.frequency, Parameter):
            negative_frequency = ParameterFn(lambda args: -args[0], (self.frequency.name,))
        else:
            negative_frequency = -self.frequency

        if isinstance(amplitude, (float, complex, Parameter)):
            # static envelope
            if isinstance(amplitude, Parameter):
                double_envelope = scaled_function(2. * drive_base, constant_function(amplitude))
                real_envelope = real_function(double_envelope)
                imag_envelope = imag_function(double_envelope)
            else:
                double_envelope = 2. * amplitude * drive_base
                real_envelope = double_envelope.real
                imag_envelope = double_envelope.imag

            if (isinstance(amplitude, Parameter) or isinstance(self.frequency, Parameter)
                or config.pulse_sim_solver == 'jax'):
                prefactor = sum_function(scaled_function(real_envelope, cos_freq(self.frequency)),
                                         scaled_function(imag_envelope, sin_freq(self.frequency)))
            else:
                prefactor_terms = []
                if real_envelope != 0.:
                    prefactor_terms.append(f'({real_envelope} * cos({self.frequency} * t))')
                if imag_envelope != 0.:
                    prefactor_terms.append(f'({imag_envelope} * sin({self.frequency} * t))')

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
            double_envelope = 2. * amplitude * drive_base

            prefactor = real_function(scaled_function(double_envelope, exp_freq(negative_frequency)))

        elif callable(amplitude):
            double_envelope = scaled_function(2. * drive_base, amplitude)

            if self.constant_phase is None:
                prefactor = real_function(prod_function(double_envelope, exp_freq(negative_frequency)))

            else:
                drive_base_phase = np.angle(drive_phase)
                if isinstance(self.constant_phase, Parameter):
                    phase = ParameterFn(lambda args: drive_base_phase + args[0], (self.constant_phase.name,))
                else:
                    phase = drive_base_phase + self.constant_phase

                absf = abs_function(double_envelope)
                prefactor = prod_function(absf, cos_freq(negative_frequency, phase))

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
                return (prod_function(prefactor, cos_freq(frame_frequency)),
                        prod_function(prefactor, sin_freq(frame_frequency)))
