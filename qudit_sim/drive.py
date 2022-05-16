r"""
======================
Hamiltonian drive term
======================

The drive Hamiltonian for a given channel :math:`j`, qudit :math:`k`, level :math:`l` is

.. math::

    \tilde{H}_{\mathrm{d}}|_{jk}^{l} = \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i (\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \left( e^{i (\xi_k^{l} t + \phi_k^{l})} \sqrt{l+1} |l + 1 \rangle_k \langle l |_k  + \mathrm{h.c.} \right).

Let

.. math::

    R_{jk}(t) = \alpha_{jk} e^{i\rho_{jk}} \frac{\Omega_j}{2} r_j(t)

and

.. math::

    A^{l}_{k} & = e^{-i \phi^{l}_{k}} \sqrt{l + 1} | l \rangle_k \langle l + 1 |_k, \\
    X^{l}_{k} & = A^{l\dagger}_{k} + A^{l}_{k} \\
    Y^{l}_{k} & = i(A^{l\dagger}_{k} - A^{l}_{k}).

Then the drive term above is

.. math::

    \tilde{H}_{\mathrm{d}}|_{jk}^{l} = & R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} A^{l\dagger}_{k} + R^{*}_{jk}(t) e^{i (\nu_j - \xi_k^{l}) t} A^{l}_{k}
                                       + R^{*}_{jk}(t) e^{i (\nu_j + \xi_k^{l}) t} A^{l\dagger}_{k} + R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t} A^{l}_{k} \\
                                     = & \mathrm{Re}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} + R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t}] X^{l}_{k}
                                       + \mathrm{Im}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} - R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t}] Y^{l}_{k} \\
                                     = & 2 \mathrm{Re}[R_{jk}(t) e^{-i \nu_j t}] [\cos(\xi_k^{l} t) X^{l}_{k} + \sin(\xi_k^{l} t) Y^{l}_{k}].

With the rotating-wave approximation we instead have

.. math::

    \bar{H}_{\mathrm{d}}|_{jk}^{l} = & R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} A^{l\dagger}_{k} + R^{*}_{jk}(t) e^{i (\nu_j - \xi_k^{l}) t} A^{l}_{k} \\
                                     = & \mathrm{Re}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t}] X^{l}_{k} + \mathrm{Im}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t}] Y^{l}_{k}.

The representation in terms of :math:`X^{l}_{k}` and :math:`Y^{l}_{k}` operators has several advantages over using :math:`A^{l}_{k}` and :math:`A^{l\dagger}_{k}`:

- When :math:`r_j(t)` is a callable, QuTiP `sesolve` seems to run slightly faster when :math:`X` and :math:`Y` with real coefficients are passed as Hamiltonian terms.
- The Hamiltonian is manifestly Hermitian.
- For a pure real or imaginary :math:`R_{jk}(t)`, on-resonant (:math:`\nu_j = \xi_k^{l}`) drive, the RWA Hamiltonian reduces to a single term.
"""

from typing import Callable, Optional, Union
from dataclasses import dataclass
import numpy as np

from .pulse import PulseSequence

REL_FREQUENCY_EPSILON = 1.e-7

def cos_freq(freq, phase=0.):
    """`cos(freq * t + phase)`"""
    return lambda t, args: np.cos(freq * t + phase)

def sin_freq(freq, phase=0.):
    """`sin(freq * t + phase)`"""
    return lambda t, args: np.sin(freq * t + phase)

def exp_freq(freq, phase=0.):
    """`cos(freq * t + phase) + 1.j * sin(freq * t + phase)`"""
    def fun(t, args):
        exponent = t * freq + phase
        return np.cos(exponent) + 1.j * np.sin(exponent)

    return fun

def scaled_function(scale, fun):
    """`scale * fun(t, args)`"""
    return lambda t, args: scale * fun(t, args)

def prod_function(fun1, fun2):
    """`fun1(t, args) * fun2(t, args)`"""
    return lambda t, args: fun1(t, args) * fun2(t, args)

def sum_function(fun1, fun2):
    """`fun1(t, args) + fun2(t, args)`"""
    return lambda t, args: fun1(t, args) + fun2(t, args)

def diff_function(fun1, fun2):
    """`fun1(t, args) - fun2(t, args)`"""
    return lambda t, args: fun1(t, args) - fun2(t, args)

def conj_function(fun):
    """`fun(t, args).conjugate()`"""
    return lambda t, args: fun(t, args).conjugate()

def real_function(fun):
    """`fun(t, args).real`"""
    return lambda t, args: fun(t, args).real

def imag_function(fun):
    """`fun(t, args).imag`"""
    return lambda t, args: fun(t, args).imag

def abs_function(fun):
    """`abs(fun(t, args))`"""
    return lambda t, args: np.abs(fun(t, args))


@dataclass(frozen=True)
class DriveTerm:
    r"""Data class representing a drive.

    Args:
        frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
            that starts with SetFrequency.
        amplitude: Function :math:`r(t)`.
        phase: The phase value of `amplitude` when it is a str or a callable and is known to have
            a constant phase. None otherwise.
    """
    frequency: Optional[float] = None
    amplitude: Union[float, complex, str, np.ndarray, PulseSequence, Callable] = 1.+0.j
    phase: Optional[float] = None

    def generate_fn(
        self,
        frame_frequency: float,
        drive_base: complex,
        rwa: bool
    ) -> tuple:
        r"""Generate the coefficients for X and Y drives.

        Args:
            frame_frequency: Frame frequency :math:`\xi_k^{l}`.
            drive_base: Factor :math:`\alpha_{jk} e^{i \rho_{jk}} \frac{\Omega_j}{2}`.
            rwa: If True, returns the RWA coefficients.

        Returns:
            A 3-tuple of X and Y coefficients and the maximum frequency appearing in the term.
        """
        if isinstance(self.amplitude, PulseSequence):
            return self.amplitude.generate_fn(frame_frequency, drive_base, rwa, initial_frequency=self.frequency)

        if self.frequency is None:
            raise RuntimeError('Drive frequency not set')

        amplitude = self.amplitude
        if isinstance(amplitude, str):
            # If this is actually a static expression, convert to complex
            try:
                amplitude = eval(amplitude)
            except:
                pass

        if rwa:
            fn_x, fn_y = self._generate_fn_rwa(amplitude, frame_frequency, drive_base)
            max_freq = abs(self.frequency - frame_frequency)
        else:
            fn_x, fn_y = self._generate_fn_full(amplitude, frame_frequency, drive_base)
            max_freq = self.frequency + frame_frequency

        return fn_x, fn_y, max_freq

    def _generate_fn_rwa(self, amplitude, frame_frequency, drive_base):
        detuning = self.frequency - frame_frequency
        is_resonant = (abs(detuning) < REL_FREQUENCY_EPSILON * frame_frequency)

        if isinstance(amplitude, (float, complex)):
            # static envelope
            envelope = amplitude * drive_base

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

        elif isinstance(amplitude, str):
            envelope = f'{drive_base} * ({amplitude})'

            if is_resonant:
                return f'({envelope}).real', f'({envelope}).imag'

            elif self.phase is None:
                return (f'({envelope}).real * cos({detuning} * t) + ({envelope}).imag * sin({detuning} * t)',
                        f'({envelope}).imag * cos({detuning} * t) - ({envelope}).real * sin({detuning} * t)')

            else:
                phase = np.angle(drive_base) + self.phase
                return (f'abs({envelope}) * cos({phase} - ({detuning} * t))',
                        f'abs({envelope}) * sin({phase} - ({detuning} * t))')

        elif isinstance(amplitude, np.ndarray):
            envelope = amplitude * drive_base

            if is_resonant:
                return envelope.real, envelope.imag

            else:
                fun = scaled_function(envelope, exp_freq(-detuning))
                return real_function(fun), imag_function(fun)

        elif callable(amplitude):
            envelope = scaled_function(drive_base, amplitude)

            if is_resonant:
                return real_function(envelope), imag_function(envelope)

            elif self.phase is None:
                cos = cos_freq(detuning)
                sin = sin_freq(detuning)
                real = real_function(envelope)
                imag = imag_function(envelope)
                return (sum_function(prod_function(real, cos), prod_function(imag, sin)),
                        diff_function(prod_function(imag, cos), prod_function(real, sin)))

            else:
                phase = np.angle(drive_base) + self.phase
                absf = abs_function(envelope)
                return (prod_function(absf, cos_freq(-detuning, phase)),
                        prod_function(absf, sin_freq(-detuning, phase)))

        else:
            raise TypeError(f'Unsupported amplitude type f{type(amplitude)}')

    def _generate_fn_full(self, amplitude, frame_frequency, drive_base):
        lab_frame = (frame_frequency == 0.)

        if isinstance(amplitude, (float, complex)):
            # static envelope
            double_envelope = 2. * amplitude * drive_base

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

            if self.phase is None:
                prefactor = f'({double_envelope} * (cos({self.frequency} * t) - 1.j * sin({self.frequency} * t))).real'

            else:
                phase = np.angle(drive_base) + self.phase
                prefactor = f'abs({double_envelope}) * cos({phase} - ({self.frequency} * t))'

        elif isinstance(amplitude, np.ndarray):
            double_envelope = 2. * amplitude * drive_base

            prefactor = real_function(scaled_function(double_envelope, exp_freq(-self.frequency)))

        elif callable(amplitude):
            double_envelope = scaled_function(2. * drive_base, amplitude)

            if self.phase is None:
                prefactor = real_function(prod_function(double_envelope, exp_freq(-self.frequency)))

            else:
                phase = np.angle(drive_base) + self.phase
                absf = abs_function(double_envelope)
                prefactor = prod_function(absf, cos_freq(-self.frequency, phase))

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
