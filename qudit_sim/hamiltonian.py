from typing import Any, Dict, Sequence, List, Tuple, Callable, Optional, Union
import copy
import numpy as np
import qutip as qtp

from .hamiltonian_utils import (ScaledExpression, ComplexExpression, ComplexFunction)

DriveAmplitude = Union[Callable, ScaledExpression.InitArg]

REL_FREQUENCY_EPSILON = 1.e-7
    
def cos_freq(freq):
    return lambda t, args: np.cos(freq * t)

def sin_freq(freq):
    return lambda t, args: np.sin(freq * t)

    
class RWAHamiltonianGenerator:
    r"""Rotating-wave approximation Hamiltonian in the qudit frame.

    **Full Hamiltonian:**
    
    The full Hamiltonian of the :math:`n`-qudit system is
    
    .. math::
    
        H = H_0 + H_{\mathrm{int}} + H_{\mathrm{d}},
        
    where
    
    .. math::
    
        H_0 & = \sum_{j=0}^{n} \left[ \omega_j b_j^{\dagger} b_j + \frac{\Delta_j}{2} b_j^{\dagger} b_j (b_j^{\dagger} b_j - 1) \right] \\
        & = \sum_{j=0}^{n} \left[ \left( \omega_j - \frac{\Delta_j}{2} \right) N_j + \frac{\Delta_j}{2} N_j^2 \right],
        
        H_{\mathrm{int}} = \sum_{jk} J_{jk} \left( b_j^{\dagger} b_k + b_j b_k^{\dagger} \right),
        
        H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \left( p_j(t) \cos (\nu_j t) + q_j(t) \sin (\nu_j t) \right) \left( e^{i\phi_{jk}} b_k^{\dagger} + e^{-i\phi_{jk}} b_k \right),
        
    with
    
    - :math:`\omega_j`: Frequency of the :math:`j`th qubit
    - :math:`\Delta_j`: Anharmonicity of the :math:`j`th qudit
    - :math:`J_{jk}`: Coupling between qudits :math:`j` and :math:`k`
    - :math:`\alpha_{jk}`: Crosstalk attenuation factor of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\phi_{jk}`: Crosstalk phase shift of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\Omega_j`: Base amplitude of drive in channel :math:`j`
    - :math:`p_j (t), q_j (t)`: I and Q components of the pulse envelope of drive in channel :math:`j`
    - :math:`\nu_j`: Local oscillator frequency of drive in channel :math:`j`.
    
    When considering more than a single drive frequency per channel, it can be more convenient to express
    the drive Hamiltonian in the frequency domain:
    
    .. math::
    
        H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \int d\nu \left( \tilde{p}_j(\nu) \cos (\nu t) + \tilde{q}_j(\nu) \sin (\nu t) \right) \left( e^{i\phi_{jk}} b_k^{\dagger} + e^{-i\phi_{jk}} b_k \right)
    
    **Qudit-frame Hamiltonian with Rotating-wave approximation:**
    
    We move to the qudit frame through a transformation with :math:`U_q := e^{i H_0 t}`:
    
    .. math::
    
        \tilde{H} & := U_q H U_q^{\dagger} + i \dot{U_q} U_q^{\dagger} \\
        & = U_q (H_{\mathrm{int}} + H_{\mathrm{d}}) U_q^{\dagger} =: \tilde{H}_{\mathrm{int}} + \tilde{H}_{\mathrm{d}}.
    
    Using :math:`b N = (N + 1) b` and :math:`b^{\dagger}N = (N - 1) b^{\dagger}`, we have

    .. math::
    
        b_j H_0^n = & \left\{ \sum_{k \neq j} \left[ \left( \omega_k - \frac{\Delta_k}{2} \right) N_k + \frac{\Delta_k}{2} N_k^2 \right] \right. \\
        + \left. \left( \omega_j - \frac{\Delta_j}{2} \right) (N_j + 1) + \frac{\Delta_j}{2} (N_j + 1)^2 \right\}^n b_j \\
        = & \left[ H_0 + \omega_j + \Delta_j N_j \right]^n b_j,

        b_j^{\dagger} H_0^n = & \left\{ \sum_{k \neq j} \left[ \left( \omega_k - \frac{\Delta_k}{2} \right) N_k + \frac{\Delta_k}{2} N_k^2 \right] \right. \\
        + \left. \left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - 1) + \frac{\Delta_j}{2} (N_j - 1)^2 \right\}^n b_j^{\dagger} \\
        = & \left[ H_0 - \omega_j - \Delta_j (N_j - 1) \right]^n b_j^{\dagger}.
              
    The interaction Hamiltonian in the qudit frame is therefore
    
    .. math::
    
        \tilde{H}_{\mathrm{int}} = \sum_{jk} J_{jk} \left( e^{i \delta_{jk} t} e^{i [\Delta_j (N_j - 1) - \Delta_k N_k] t} b_j^{\dagger} b_k \right. \\
        \left. + e^{-i \delta_{jk} t} e^{-i [\Delta_j N_j - \Delta_k (N_k - 1)] t} b_j b_k^{\dagger} \right).

    The drive Hamiltonian is
    
    .. math::
    
        \tilde{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \left( p_j(t) \cos (\nu_j t) + q_j(t) \sin (\nu_j t) \right) \left( e^{i (\omega_k t + \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + e^{-i (\omega_k t + \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right),
    
    and with the rotating wave approximation (RWA)
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left[ r_j(t) e^{-i (\epsilon_{jk} t - \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + r^{*}_j(t) e^{i (\epsilon_{jk} t - \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right],
        
    where :math:`\epsilon_{jk} := \nu_j - \omega_k` and :math:`r_j(t) := p_j(t) + i q_j(t)`.
    
    The RWA drive Hamiltonian in the frequency domain is (assuming :math:`\tilde{p}_j` and :math:`\tilde{q}_j` have
    support only around the qudit frequencies)
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left[ \tilde{r}_j(\nu) e^{-i [(\nu - \omega_k) t - \phi_{jk}]} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + \tilde{r}^{*}_j(\nu) e^{i [(\nu - \omega_k) t - \phi_{jk}]} e^{-i \Delta_k N_k t} b_k \right].
    
    **QuTiP implementation:**
    
    Time-dependent Hamiltonian in QuTiP is represented by a two-tuple `(H, c(t))` where `H` is a static Qobj and `c(t)`
    is the time-dependent coefficient of `H`. This function returns two lists of tuples, corresponding to the 
    interaction and drive Hamiltonians, with the total list length corresponding to the number of distinct
    time dependencies in the RWA Hamiltonian :math:`\tilde{H}_{\mathrm{int}} + \bar{H}_{\mathrm{d}}`. Because
    `c(t)` must be a real function, each returned tuple contains two Qobjs corresponding to the "X"
    (:math:`\propto b^{\dagger} + b`) and "Y" (:math:`\propto i(b^{\dagger} - b)`) parts of the Hamiltonian.
        
    **TODO: Qubit optimization:**
    
    When only considering qubits, anharmonicity terms are dropped, making the formulae above somewhat simpler. In particular,
    when one or more drives are resonant with qubit frequencies, some terms in the interaction and drive Hamiltonians
    will have common frequencies (i.e. :math:`\delta_{jk}` and `\epsilon_{kj}` coincide). We should look into how to detect
    and exploit such cases in the future.
    
    Attributes:
        h_int (list): Interaction term components. List of n_coupling elements. Each element is a three-tuple
            (freq, H_x, H_y), where freq is the frequency for the given term. H_x and H_y are the coupling
            Hamiltonians proportional to the "X" and "Y" parts of the oscillating coefficients.
        h_drive (list): Drive term components. List of n_qubits elements. Each element is a three-tuple
            (expr_gen, H_x, H_y), where expr_gen is an instance of DriveExprGen.
    """
    def __init__(
        self,
        qubits: Sequence[int],
        params: Dict[str, Any],
        num_levels: int = 2,
        compile_hint: bool = True
    ) -> None:
        r"""
        Args:
            qubits: List of :math:`n` qudits to include in the Hamiltonian.
            params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally
                augmented with `'crosstalk'`, which should be a `dict` of form `{(i, j): z}` specifying the crosstalk
                factor `z` (complex corresponding to :math:`\alpha_{jk} e^{i\phi_{jk}}`) of drive on qubit `i` seen
                by qubit `j`. `i` and `j` are qubit ids given in `qubits`.
            num_levels: Number of oscillator levels to consider.
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
        """
        ## Validate and format the input

        if isinstance(qubits, int):
            qubits = (qubits,)

        num_qubits = len(qubits)

        self.qubit_index_mapping = dict((q, iq) for iq, q in enumerate(qubits))
        
        self.num_levels = num_levels
        
        ## Parse the parameters
        
        self.qubit_frequencies = np.array([params[f'wq{q}'] for q in qubits])
        self.anharmonicities = np.array([params[f'delta{q}'] for q in qubits])
        
        # Element [j,k]: \alpha_{jk} \exp (i \phi_{jk})
        crosstalk_matrix = np.eye(num_qubits, dtype=np.complex128)
        if 'crosstalk' in params:
            for (qsrc, qtrg), factor in params['crosstalk'].items():
                isrc = self.qubit_index_mapping[qsrc]
                itrg = self.qubit_index_mapping[qtrg]
                crosstalk_matrix[isrc, itrg] = factor

        amps = np.array(list(params[f'omegad{ch}'] for ch in qubits)) / 2.
        # Element [j,k]: \alpha_{jk} \frac{\Omega}{2} \exp (i \phi_{jk})
        self.drive_base = amps[:, None] * crosstalk_matrix
        
        self._max_frequency_int = 0.
        self._max_frequency_drive = 0.
        
        ## Lists of Hamiltonian terms
        
        self.hdrive = list()
        self.hint = list()
        self.hstatic = qtp.Qobj()

        # Does this represent an array-based Hamiltonian?
        self._need_tlist = False
        
        ## Compute the interaction term components
        
        for q1 in qubits:
            iq1 = self.qubit_index_mapping[q1]

            for q2 in qubits[iq1 + 1:]:
                iq2 = self.qubit_index_mapping[q2]

                try:
                    J = params[f'jq{min(q1, q2)}q{max(q1, q2)}']
                except KeyError:
                    # no coupling between the qudits
                    continue

                freq_diff = (self.qubit_frequencies[iq1] - self.qubit_frequencies[iq2])

                D1 = self.anharmonicities[iq1]
                D2 = self.anharmonicities[iq2]
                
                # Define one Hamiltonian term per level pair
                # cf. Hint = sum_{jk} J_{jk} exp(i d_{jk} t) exp(i D_j (N_j - 1) t) exp(-i D_k N_k t) b_j+ b_k + H.C.

                for l1 in range(num_levels - 1):
                    for l2 in range(num_levels - 1):
                        c1 = np.sqrt(l1 + 1) * qtp.basis(num_levels, l1) * qtp.basis(num_levels, l1 + 1).dag()
                        c2 = np.sqrt(l2 + 1) * qtp.basis(num_levels, l2) * qtp.basis(num_levels, l2 + 1).dag()
                        op = [qtp.qeye(num_levels)] * num_qubits
                        op[iq1] = c1
                        op[iq2] = c2.dag()
                        h_x = J * (qtp.tensor(op).dag() + qtp.tensor(op))
                        h_y = J * 1.j * (qtp.tensor(op).dag() - qtp.tensor(op))
                        freq = freq_diff + D1 * l1 - D2 * l2

                        if compile_hint:
                            self.hint.append([h_x, f'cos({freq}*t)'])
                            self.hint.append([h_y, f'sin({freq}*t)'])
                        else:
                            self.hint.append([h_x, cos_freq(freq)])
                            self.hint.append([h_y, sin_freq(freq)])
                        
                        self._max_frequency_int = max(self._max_frequency_int, abs(freq))
                        
    @property
    def max_frequency(self) -> float:
        """Return the maximum frequency appearing in this Hamiltonian."""
        
        return max(self._max_frequency_int, self._max_frequency_drive)
    
    @property
    def need_tlist(self) -> bool:
        return self._need_tlist

    def add_drive(
        self,
        qubit: int,
        frequency: float,
        amplitude: Optional[Union[complex, np.ndarray, DriveAmplitude, Tuple[DriveAmplitude, DriveAmplitude]]] = None
    ) -> None:
        """Add a drive term.
        
        Args:
            qubit: ID of the qubit to apply the drive to.
            frequency: Carrier frequency of the drive.
            amplitude: A constant drive amplitude or an envelope function.
        """
        ich = self.qubit_index_mapping[qubit]
        num_qubits = len(self.qubit_index_mapping)
        
        if amplitude is None:
            drive_amp = ComplexExpression(1., 0.)
        else:
            if isinstance(amplitude, complex):
                drive_amp = ComplexExpression(amplitude.real, amplitude.imag)
            elif isinstance(amplitude, np.ndarray):
                drive_amp = amplitude
                self._need_tlist = True
            else:
                if not isinstance(amplitude, tuple):
                    amplitude = (amplitude, 0.)
                    
                if callable(amplitude[0]):
                    drive_amp = ComplexFunction(amplitude[0], amplitude[1])
                else:
                    drive_amp = ComplexExpression(amplitude[0], amplitude[1])
            
        for iq in range(num_qubits):
            if self.drive_base[ich, iq] == 0.:
                continue

            # Amplitude expressions for X (even) and Y (odd) terms
            # Corresponds to R_j(t) := \alpha_{jk} \frac{\Omega_j}{2} r_j(t) \exp(i \phi_{jk})
            envelope = drive_amp * self.drive_base[ich, iq]
            if isinstance(envelope, ComplexExpression):
                try:
                    radial, phase = envelope.polar()
                except ValueError:
                    radial = None
            
            # Define one Hamiltonian term per level
            for level in range(self.num_levels - 1):
                # \omega_k + n \Delta_k
                level_frequency = self.qubit_frequencies[iq] + level * self.anharmonicities[iq]
                # \nu_j - \omega_k - n \Delta_k
                detuning = frequency - level_frequency
                
                if abs(detuning) < REL_FREQUENCY_EPSILON * self.qubit_frequencies[iq]:
                    level_drive = copy.deepcopy(envelope)
                    
                elif isinstance(envelope, ComplexExpression):
                    if radial is not None:
                        carrier_phase = f'{-detuning}*t{phase:+}'
                        level_drive = ComplexExpression(f'cos({carrier_phase})', f'sin({carrier_phase})')
                        level_drive *= radial
                    else:
                        carrier = ComplexExpression(f'cos({-detuning}*t)', f'sin({-detuning}*t)')
                        level_drive = carrier * envelope
                        
                elif isinstance(envelope, (ComplexFunction, np.ndarray)):
                    carrier = ComplexFunction(cos_freq(-detuning), sin_freq(-detuning))
                    level_drive = carrier * envelope
                    
                qudit_ops = [qtp.qeye(self.num_levels)] * num_qubits
                qudit_ops[iq] = (np.sqrt(level + 1) * qtp.basis(self.num_levels, level)
                    * qtp.basis(self.num_levels, level + 1).dag())
                annihilator = qtp.tensor(qudit_ops)
                h_x = annihilator.dag() + annihilator
                h_y = 1.j * (annihilator.dag() - annihilator)
                
                for h, amp in zip((h_x, h_y), level_drive):
                    if isinstance(level_drive, ComplexExpression):
                        if amp.is_zero():
                            continue

                        if amp.expression is None:
                            self.hstatic += h * amp.scale
                        else:
                            self.hdrive.append([h * amp.scale, amp.expression])

                    else:
                        if amp:
                            self.hdrive.append([h, amp])
                        
                self._max_frequency_drive = max(self._max_frequency_drive, abs(detuning))
                
    def clear_drive(self) -> None:
        """Reset all drive-related attributes."""
        self.hdrive = list()
        self._max_frequency_drive = 0.
        self._need_tlist = False
        
    def generate(self) -> List:
        """Return the list of Hamiltonian terms passable to qutip.sesolve."""
        
        if self._need_tlist:
            raise RuntimeError('This Hamiltonian must be instantiated with array_hamiltonian()')
        
        if self.hstatic == qtp.Qobj():
            return self.hint + self.hdrive
        else:
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            return [self.hstatic] + self.hint + self.hdrive

    def array_generate(
        self,
        tlist: np.ndarray,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Return a list of Hamiltonian terms passable to qutip.sesolve.
        
        When at least one drive term is given in terms of an ndarray, the concrete Hamiltonian must be generated
        through this function. When all time-dependent terms are string-based, the output is identical to what is
        obtained from `generate()`.
        """
        hamiltonian = []
        
        if self.hstatic != qtp.Qobj():
            hamiltonian.append(self.hstatic)
            
        for h, f in self.hint + self.hdrive:
            if callable(f):
                hamiltonian.append([h, f(tlist, args)])
            else:
                hamiltonian.append([h, f])
        
        return hamiltonian

    def make_tlist(
        self,
        points_per_cycle: int,
        num_cycles: int
    ) -> np.ndarray:
        """Generate a list of time points using the maximum frequency in the Hamiltonian.
        
        When the maximum frequency is 0 (i.e. single-qubit simulation with resonant drive), return the time
        range from 0 to 2pi/sqrt(tr(Hstat . Hstat) / 2^nq).
    
        Args:
            points_per_cycle: Number of points per cycle at the highest frequency.
            num_cycles: Number of overall cycles.

        Returns:
            Array of time points.
        """
        if self.max_frequency == 0.:
            hstat = self.hstatic.full()
            amp2 = np.trace(hstat @ hstat).real / (2 ** len(self.qubit_index_mapping))
            return np.linspace(0., 2. * np.pi / np.sqrt(amp2) * num_cycles, points_per_cycle * num_cycles)
        else:
            return np.linspace(0., 2. * np.pi / self.max_frequency * num_cycles, points_per_cycle * num_cycles)
