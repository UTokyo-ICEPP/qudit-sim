from typing import Any, Dict, Sequence, List, Tuple, Callable, Optional, Union
import copy
from dataclasses import dataclass
import numpy as np
import qutip as qtp

from .pulse import PulseSequence

def cos_freq(freq):
    return lambda t, args: np.cos(freq * t)

def sin_freq(freq):
    return lambda t, args: np.sin(freq * t)

def exp_freq(freq):
    def fun(t, args):
        exponent = t * freq
        return np.cos(exponent) + 1.j * np.sin(exponent)
    
    return fun

def scaled_function(fun, scale):
    return lambda t, args: scale * fun(t, args)

def prod_function(fun1, fun2):
    return lambda t, args: fun1(t, args) * fun2(t, args)

def conj_function(fun):
    return lambda t, args: fun(t, args).conjugate()


REL_FREQUENCY_EPSILON = 1.e-7

@dataclass
class QuditParams:
    qid: int
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float
    frame: Optional[np.ndarray] = None
    frame_phase: Optional[np.ndarray] = None

@dataclass
class DriveData:
    frequency: float
    amplitude: Union[float, complex, str, np.ndarray, Callable, None] = 1.+0.j
    sequence: Optional[PulseSequence] = None
    is_real: bool = False

    
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
    
    - :math:`\omega_j`: Frequency of the :math:`j`th qudit
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
    """
    def __init__(
        self,
        num_levels: int = 2,
        qudits: Optional[Union[int, Sequence[int]]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        r"""
        Args:
            num_levels: Number of oscillator levels to consider.
            qudits: List of :math:`n` qudits to include in the Hamiltonian.
            params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally
                augmented with `'crosstalk'`, which should be a `dict` of form `{(i, j): z}` specifying the crosstalk
                factor `z` (complex corresponding to :math:`\alpha_{jk} e^{i\phi_{jk}}`) of drive on qudit `i` seen
                by qudit `j`. `i` and `j` are qudit ids given in `qudits`.
        """
        
        self.num_levels = num_levels
        
        # Makes use of dict order guarantee from python 3.7
        self.qudit_params = dict()
        self.coupling = dict()
        self.crosstalk = dict()
        self.drive = dict()
        
        self._max_frequency_int = None
        self._max_frequency_drive = None
        self._need_tlist = False
        
        if qudits is None:
            return
        
        if isinstance(qudits, int):
            qudits = (qudits,)

        for q in qudits:
            self.add_qudit(q, params[f'wq{q}'], params[f'delta{q}'], params[f'omegad{q}'])

        for q1, q2 in zip(qudits[:-1], qudits[1:]):
            try:
                coupling = params[f'jq{min(q1, q2)}q{max(q1, q2)}']
            except KeyError:
                continue

            self.add_coupling(q1, q2, coupling)
            
        if 'crosstalk' in params:
            for (qsrc, qtrg), factor in params['crosstalk'].items():
                self.add_crosstalk(qsrc, qtrg, factor)
            
    def add_qudit(
        self,
        qudit: int,
        qubit_frequency: float,
        anharmonicity: float,
        drive_amplitude: float,
        index: Optional[int] = None,
        frame: Optional[np.ndarray] = None,
        frame_phase: Optional[np.ndarray] = None,
    ) -> None:
        """Add a qudit to the system.
        """
        params = QuditParams(qid=qudit, qubit_frequency=qubit_frequency, anharmonicity=anharmonicity,
                             drive_amplitude=drive_amplitude, frame=frame, frame_phase=frame_phase)

        if index is None:
            self.qudit_params[qudit] = params
        elif index > len(self.qudit_params):
            raise IndexError(f'Index {index} greater than number of existing parameters')
        else:
            qudit_params = dict(list(self.qudit_params.items())[:index])
            qudit_params[qudit] = params
            qudit_params.update(list(self.qudit_params.items())[index:])

            self.qudit_params.clear()
            self.qudit_params.update(qudit_params)
        
    def set_frame(
        self,
        qudit: int,
        frame: Optional[np.ndarray] = None,
        phase: Optional[np.ndarray] = None
    ) -> None:
        """Set the rotating frame for the qudit."""
        if frame is not None:
            self.qudit_params[qudit].frame = frame.copy()
        if phase is not None:
            self.qudit_params[qudit].frame_phase = phase.copy()
            
    def add_coupling(self, q1: int, q2: int, value: float) -> None:
        self.coupling[frozenset({self.qudit_params[q1], self.qudit_params[q2]})] = value
            
    def add_crosstalk(self, source: int, target: int, factor: complex) -> None:
        self.crosstalk[(self.qudit_params[source], self.qudit_params[target])] = factor
        
    def add_drive(
        self,
        qudit: int,
        frequency: float,
        amplitude: Union[float, complex, str, np.ndarray, Callable, None] = 1.+0.j,
        sequence: Optional[PulseSequence] = None,
        is_real: bool = False
    ) -> None:
        """Add a drive term.
        
        Args:
            qudit: ID of the qudit to apply the drive to.
            frequency: Carrier frequency of the drive.
            amplitude: A constant drive amplitude or an envelope function.
            sequence: The full pulse sequence. If not None, `amplitude` is ignored.
            is_real: Set to True if `amplitude` is a str or a callable and is known to take
                only real values.
        """
        self.drive[self.qudit_params[qudit]] = DriveData(frequency=frequency, amplitude=amplitude,
                                                         sequence=sequence, is_real=is_real)
        
    @property
    def max_frequency(self) -> float:
        """Return the maximum frequency appearing in this Hamiltonian."""
        
        return max(self._max_frequency_int, self._max_frequency_drive)
    
    @property
    def need_tlist(self) -> bool:
        return self._need_tlist

    def generate(
        self,
        compile_hint: bool = True
    ) -> List:
        """Return the list of Hamiltonian terms passable to qutip.sesolve.
        
        Args:
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
        
        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        
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
    
    def _generate_hdiag(self) -> qtp.Qobj:
        hdiag = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        
        for iq, params in enumerate(self.qudit_params.values()):
            if params.frame is None:
                # qudit frame -> free Hamiltonian is null
                continue
                
            qudit_hfree = np.arange(self.num_levels) * (params.qubit_frequency - params.anharmonicity / 2.)
            qudit_hfree += np.square(np.arange(self.num_levels)) * params.anharmonicity / 2.
            energy_offset = qudit_hfree - np.cumsum(np.concatenate((np.zeros(1), params.frame)))
            qudit_op = qtp.Qobj(inpt=np.diag(energy_offset))

            ops = [qtp.qeye(self.num_levels)] * num_qudits
            ops[iq] = qudit_op
            
            hdiag += qtp.tensor(ops)
            
        return hdiag
    
    def _generate_hint(self, compile_hint: bool = True) -> List:
        """Generate the interaction Hamiltonian."""
        self._max_frequency_int = 0.

        hint = list()
        hstatic = qtp.Qobj()

        num_qudits = len(self.qudit_params)
        params_list = list(self.qudit_params.values())
        
        for iq1, iq2 in np.ndindex((num_qudits, num_qudits)):
            if iq2 <= iq1:
                continue
                
            p1 = params_list[iq1]
            p2 = params_list[iq2]

            try:
                coupling = self.coupling[frozenset({p1, p2})]
            except KeyError:
                # no coupling between the qudits
                continue

            for l1, l2 in np.ndindex((self.num_levels - 1, self.num_levels - 1)):
                # exp(i (xi_{p1,l1} - xi_{p2,l2}) t) sqrt(l1 + 1) sqrt(l2 + 1) |l1+1>|l2><l1|<l2+1| + h.c.
                
                # Annihilator terms for this level combination
                ann1 = np.sqrt(l1 + 1) * qtp.basis(self.num_levels, l1) * qtp.basis(self.num_levels, l1 + 1).dag()
                ann2 = np.sqrt(l2 + 1) * qtp.basis(self.num_levels, l2) * qtp.basis(self.num_levels, l2 + 1).dag()
                
                ops = [qtp.qeye(self.num_levels)] * num_qudits
                ops[iq1] = ann1.dag()
                ops[iq2] = ann2
                
                op = qtp.tensor(ops)
                
                if p1.frame is None:
                    # qudit frame
                    frequency = p1.qubit_frequency + l1 * p1.anharmonicity
                else:
                    frequency = p1.frame[l1]
                    
                if p2.frame is None:
                    frequency -= p2.qubit_frequency + l2 * p2.anharmonicity
                else:
                    frequency -= p2.frame[l2]
                    
                if abs(frequency) < REL_FREQUENCY_EPSILON * (p1.qubit_frequency + p2.qubit_frequency) * 0.5:
                    hstatic += op + op.dag()

                else:
                    h_x = coupling * (op + op.dag())
                    h_y = coupling * 1.j * (op - op.dag())
                
                    if compile_hint:
                        hint.append([h_x, f'cos({frequency}*t)'])
                        hint.append([h_y, f'sin({frequency}*t)'])
                    else:
                        hint.append([h_x, cos_freq(frequency)])
                        hint.append([h_y, sin_freq(frequency)])

                    self._max_frequency_int = max(self._max_frequency_int, abs(frequency))
                    
        if hstatic != qtp.Qobj():
            hint.insert(0, hstatic)

        return hint

    def _generate_hdrive(self) -> List:
        self._max_frequency_drive = 0.
        self._need_tlist = False
        
        hdrive = list()
        hstatic = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        params_list = list(self.qudit_params.values())
        
        # Element [j,k]: \alpha_{jk} \exp (i \phi_{jk})
        crosstalk_matrix = np.eye(num_qudits, dtype=np.complex128)
        
        for isource, psource in enumerate(self.qudit_params.values()):
            for itarget, ptarget in enumerate(self.qudit_params.values()):
                try:
                    factor = self.crosstalk[(psource, ptarget)]
                except KeyError:
                    pass
                
                crosstalk_matrix[isource, itarget] = factor
            
        amps = np.array(list(params[f'omegad{ch}'] for ch in qubits), dtype=np.complex128) / 2.
        # Element [j,k]: \alpha_{jk} \frac{\Omega}{2} \exp (i \phi_{jk})
        self.drive_base = amps[:, None] * crosstalk_matrix

        # Construct the Qobj for each qudit/level first
        qops = list()

        for iq, params in enumerate(self.qudit_params.values()):
            for level in range(self.num_levels - 1):
                cre = np.sqrt(level + 1) * qtp.basis(self.num_levels, level + 1) * qtp.basis(self.num_levels, level).dag()

                ops = [qtp.qeye(self.num_levels)] * num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)
                
                if params.frame is None:
                    frame_frequency = params.qubit_frequency + level * params.anharmonicity
                else:
                    frame_frequency = params.frame[level]
                
                qops.append((params, op, frame_frequency))

        # Loop over the drive channels
        for ch_params, drive in self.drive.items():
            if isinstance(drive.amplitude, np.ndarray):
                self._need_tlist = True
            
            ich = params_list.index(ch_params)

            drive_base = crosstalk_matrix[ich] * ch_params.drive_amplitude / 2.
            
            # Loop over the driven operators
            for params, creation_op, frame_frequency in qops:
                iq = params_list.index(params)
                
                if drive_base[iq] == 0.:
                    continue
                    
                # Possible time dependence of the envelope implies that the Hamiltonian cannot be
                # split into h_x and h_y -> need to give complex drives to the creation and annihilation
                # operators instead
                
                is_static = False
                is_real = False
                    
                if drive.sequence is None:
                    detuning = drive.frequency - frame_frequency

                    on_resonance = (abs(detuning) < REL_FREQUENCY_EPSILON * params.qubit_frequency)

                    if isinstance(drive.amplitude, (float, complex)):
                        envelope = drive.amplitude * drive_base[iq]

                        if on_resonance:
                            cr_drive = envelope
                            an_drive = envelope.conjugate()
                            is_real = envelope.imag == 0.
                            is_static = True
                        else:
                            cr_drive = f'{envelope} * (cos({detuning} * t) - 1.j * sin({detuning} * t))'
                            an_drive = f'{envelope.conjugate()} * (cos({detuning} * t) + 1.j * sin({detuning} * t))'
                            
                    elif isinstance(drive.amplitude, np.ndarray):
                        envelope = drive.amplitude * drive_base[iq]
                        
                        if on_resonance:
                            cr_drive = envelope
                            an_drive = envelope.conjugate()
                            is_real = np.all(envelope.imag == 0.)
                        else:
                            cr_drive = scaled_function(exp_freq(-detuning), envelope)
                            an_drive = scaled_function(exp_freq(detuning), envelope.conjugate())

                    elif isinstance(drive.amplitude, str):
                        envelope = f'({drive_base[iq]} * ({drive.amplitude}))'
                        
                        if on_resonance:
                            cr_drive = envelope
                            an_drive = f'{envelope}.conjugate()'
                            is_real = drive.is_real
                            try:
                                cr = eval(cr_drive)
                                an = eval(an_drive)
                            except:
                                pass
                            else:
                                cr_drive = cr
                                an_drive = an
                                is_static = True
                        else:
                            cr_drive = f'{envelope} * (cos({detuning} * t) - 1.j * sin({detuning} * t))'
                            an_drive = f'{envelope}.conjugate() * (cos({detuning} * t) + 1.j * sin({detuning} * t))'
                            
                    elif callable(drive.amplitude):
                        envelope = scaled_function(drive.amplitude, drive_base[iq])
                        
                        if on_resonance:
                            cr_drive = envelope
                            an_drive = conj_function(envelope)
                            is_real = drive.is_real
                        else:
                            cr_drive = prod_function(exp_freq(-detuning), envelope)
                            an_drive = prod_function(exp_freq(detuning), conj_function(envelope))
                            
                else:
                    cr_drive = drive.sequence.generate_fn(drive.frequency, frame_frequency)
                    an_drive = conj_function(cr_drive)
                    on_resonance = False

                if is_static:
                    hstatic += cr_drive * creation_op + an_drive * creation_op.dag()

                elif is_real:
                    self.hdrive.append([creation_op + creation_op.dag(), cr_drive])
                else:
                    self.hdrive.append([creation_op, cr_drive])
                    self.hdrive.append([creation_op.dag(), an_drive])
                        
                self._max_frequency_drive = max(self._max_frequency_drive, abs(drive.frequency - frame_frequency))
                
                        
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
