from typing import Any, Dict, Sequence, List, Tuple, Callable, Optional, Union
import copy
from dataclasses import dataclass
import numpy as np
import qutip as qtp

def cos_freq(freq):
    return lambda t, args: np.cos(freq * t)

def sin_freq(freq):
    return lambda t, args: np.sin(freq * t)

def exp_freq(freq):
    def fun(t, args):
        exponent = t * freq
        return np.cos(exponent) + 1.j * np.sin(exponent)
    
    return fun

def scaled_function(scale, fun):
    return lambda t, args: scale * fun(t, args)

def prod_function(fun1, fun2):
    return lambda t, args: fun1(t, args) * fun2(t, args)

def conj_function(fun):
    return lambda t, args: fun(t, args).conjugate()


REL_FREQUENCY_EPSILON = 1.e-7

@dataclass
class Frame:
    frequency: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None

@dataclass
class QuditParams:
    qid: int
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float
    frame: Frame

    
@dataclass
class DriveData:
    frequency: float
    amplitude: Union[float, complex, str, np.ndarray, Callable, None] = 1.+0.j
    sequence: Optional[PulseSequence] = None
    is_real: bool = False
    
    def generate_fn(frame, drive_base):
        detuning = self.frequency - frame.frequency

        is_resonant = (abs(detuning) < REL_FREQUENCY_EPSILON * frame.frequency)
        
        amplitude = self.amplitude
        if isinstance(amplitude, str):
            # If this is actually a static expression, convert to complex
            try:
                amplitude = eval(amplitude)
            except:
                pass
            
        if isinstance(amplitude, (float, complex)):
            # static envelope
            
            envelope = amplitude * drive_base
            
            is_real = envelope.imag == 0.

            if is_resonant:
                fn_1 = envelope.real
                if is_real:
                    fn_2 = None
                else:
                    fn_2 = -envelope.imag
            else:
                fn_1 = f'{envelope.real} * cos({detuning} * t)'
                fn_2 = f'{envelope.real} * sin({detuning} * t)'
                if not is_real:
                    fn_1 += f' + {envelope.imag} * sin({detuning} * t)'
                    fn_2 += f' - {envelope.imag} * cos({detuning} * t)'
                    
            split = 'xy'
            
        else:
            # dynamic envelope

            if isinstance(amplitude, str):
                envelope = f'({drive_base} * ({amplitude}))'

                is_real = self.is_real

                conj = f'{envelope}.conjugate()'
                cos = f'cos({detuning} * t)'
                sin = f'sin({detuning} * t)'
                exp_n = f'(cos({detuning} * t) - 1.j * sin({detuning} * t))'
                exp_p = f'(cos({detuning} * t) + 1.j * sin({detuning} * t))'
                prod = lambda env, f: f'{env} * {f}'            

            if isinstance(amplitude, np.ndarray):
                envelope = amplitude * drive_base

                is_real = np.all(envelope.imag == 0.)

                conj = envelope.conjugate()
                cos = cos_freq(detuning)
                sin = sin_freq(detuning)
                exp_n = exp_freq(-detuning)
                exp_p = exp_freq(detuning)
                prod = lambda env, f: scaled_function(env, f)

            elif callable(amplitude):
                envelope = scaled_function(drive_base, amplitude)

                is_real = self.is_real

                conj = conj_function(envelope)
                cos = cos_freq(detuning)
                sin = sin_freq(detuning)
                exp_n = exp_freq(-detuning)
                exp_p = exp_freq(detuning)
                prod = lambda env, f: prod_function(env, f)
                
            if is_real:
                if is_resonant:
                    fn_1 = envelope
                    fn_2 = None
                else:
                    fn_1 = prod(envelope, cos)
                    fn_2 = prod(envelope, sin)
                
                split = 'xy'
                
            else:
                if is_resonant:
                    fn_1 = envelope
                    fn_2 = conj
                else:
                    fn_1 = prod(envelope, exp_n)
                    fn_2 = prod(conj, exp_p)
                    
                split = 'ca'

        return split, fn_1, fn_2
    
    
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
        frame_frequency: Optional[np.ndarray] = None,
        frame_phase: Optional[np.ndarray] = None,
    ) -> None:
        """Add a qudit to the system.
        """
        params = QuditParams(qid=qudit, qubit_frequency=qubit_frequency, anharmonicity=anharmonicity,
                             drive_amplitude=drive_amplitude,
                             frame=Frame(frequency=frame_frequency, phase=frame_phase))

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
        frequency: Union[np.ndarray, None],
        phase: Union[np.ndarray, None]
    ) -> None:
        """Set the rotating frame for the qudit."""
        self.qudit_params[qudit].frame = Frame(frequency=frequency, phase=phase)
            
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
        rwa: bool = True,
        compile_hint: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Return the list of Hamiltonian terms passable to qutip.sesolve.
        
        Args:
            rwa: If True, apply the rotating-wave approximation.
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
            tlist: If not None, all callable Hamiltonian coefficients are called with (tlist, args) and the resulting
                arrays are instead passed to sesolve.
            args: Arguments to the callable coefficients.
        
        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        
        if tlist is None and self._need_tlist:
            raise RuntimeError('This Hamiltonian must be generated with a tlist')
        
        hstatic = self.generate_hdiag()
        hint = self.generate_hint(compile_hint=compile_hint)
        hdrive = self.generate_hdrive(rwa=rwa)
        
        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint.pop(0)
        if hdrive and isinstance(hdrive[0], qtp.Qobj):
            hstatic += hdrive.pop(0)

        hamiltonian = []
        
        if hstatic != qtp.Qobj():
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            hamiltonian.append(hstatic)
            
        if tlist is not None:
            for h, f in hint + hdrive:
                if callable(f):
                    hamiltonian.append([h, f(tlist, args)])
                else:
                    hamiltonian.append([h, f])
                    
        else:
            hamiltonian.extend(hint)
            hamiltonian.extend(hdrive)
            
        return hamiltonian
        
    def generate_hdiag(self) -> qtp.Qobj:
        hdiag = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        
        for iq, params in enumerate(self.qudit_params.values()):
            if params.frame.frequency is None:
                # qudit frame -> free Hamiltonian is null
                continue
                
            qudit_hfree = np.arange(self.num_levels) * (params.qubit_frequency - params.anharmonicity / 2.)
            qudit_hfree += np.square(np.arange(self.num_levels)) * params.anharmonicity / 2.
            energy_offset = qudit_hfree - np.cumsum(np.concatenate((np.zeros(1), params.frame.frequency)))
            qudit_op = qtp.Qobj(inpt=np.diag(energy_offset))

            ops = [qtp.qeye(self.num_levels)] * num_qudits
            ops[iq] = qudit_op
            
            hdiag += qtp.tensor(ops)
            
        return hdiag
    
    def generate_hint(self, compile_hint: bool = True) -> List:
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
                frequency = 0.
                
                for params, level, sign in [(p1, l1, 1.), (p2, l2, -1.)]:
                    if params.frame.phase is not None:
                        op *= np.exp(sign * 1.j * params.frame.phase[level])
                        
                    if params.frame.frequency is None:
                        frequency += sign * (params.qubit_frequency + level * params.anharmonicity)
                    else:
                        frequency += sign * params.frame.frequency[level]
                        
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

    def generate_hdrive(self, rwa: bool = True) -> List:
        self._max_frequency_drive = 0.
        self._need_tlist = False
        
        hdrive = list()
        hstatic = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        
        # Construct the Qobj for each qudit/level first
        qops = list()

        for iq, params in enumerate(self.qudit_params.values()):
            for level in range(self.num_levels - 1):
                cre = np.sqrt(level + 1) * qtp.basis(self.num_levels, level + 1) * qtp.basis(self.num_levels, level).dag()

                ops = [qtp.qeye(self.num_levels)] * num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)
                
                if params.frame.phase is not None:
                    op *= np.exp(1.j * params.frame.phase[level])
                
                if params.frame.frequency is None:
                    frame_frequency = params.qubit_frequency + level * params.anharmonicity
                else:
                    frame_frequency = params.frame.frequency[level]

                qops.append((params, op, frame_frequency))

        # Loop over the drive channels
        for ch_params, drive in self.drive.items():
            if isinstance(drive.amplitude, np.ndarray):
                self._need_tlist = True
            
            # Loop over the driven operators
            for params, creation_op, frame_frequency in qops:
                drive_base = ch_params.drive_amplitude / 2.
                
                if ch_params != params:
                    try:
                        drive_base *= self.crosstalk[(ch_params, params)]
                    except KeyError:
                        # This qudit gets no drive
                        continue
                        
                # Hamiltonian term can be split in xy (static and/or real envelope) or creation/annihilation (otherwise)
                    
                if drive.sequence is None:
                    split, fn_1, fn_2 = drive.generate_fn(frame_frequency, drive_base)
                    term_max_frequency = abs(drive.frequency - frame_frequency)
                            
                else:
                    split, fn_1, fn_2, term_max_frequency = drive.sequence.generate_fn(frame_frequency, drive_base,
                                                                                       initial_frequency=drive.frequency)
                    
                if split == 'xy':
                    h_x = creation_op.dag() + creation_op
                    h_y = 1.j * (creation_op.dag() - creation_op)

                    if isinstance(fn_1, (float, complex)):
                        hstatic += fn_1 * h_x
                        if fn_2 is not None:
                            hstatic += fn_2 * h_y

                    else:
                        hdrive.append([h_x, fn_1])
                        if fn_2 is not None:
                            hdrive.append([h_y, fn_2])
                            
                else:
                    hdrive.append([creation_op, fn_1])
                    hdrive.append([creation_op.dag(), fn_2])
                        
                self._max_frequency_drive = max(self._max_frequency_drive, term_max_frequency)
                
        if hstatic != qtp.Qobj():
            hdrive.insert(0, hstatic)
                
        return hdrive
                
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
            hamiltonian = self.generate()
            if not hamiltonian:
                raise RuntimeError('Cannot determine the tlist')

            hstat = hamiltonian[0]
            amp2 = np.trace(hstat @ hstat).real / (2 ** len(self.qudit_params))
            return np.linspace(0., 2. * np.pi / np.sqrt(amp2) * num_cycles, points_per_cycle * num_cycles)
        else:
            return np.linspace(0., 2. * np.pi / self.max_frequency * num_cycles, points_per_cycle * num_cycles)
