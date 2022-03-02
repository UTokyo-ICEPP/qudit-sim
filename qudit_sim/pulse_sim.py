from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import os
import copy
import numpy as np
import qutip as qtp

phase_epsilon = 1.e-9

class DriveExprGen:
    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels
        self.base_frequency = 0.
        self.base_amplitude = np.zeros(num_channels)
        self.phase_shift = np.zeros(num_channels)
        
    def __str__(self):
        return (f'DriveExprGen: num_channels = {self.num_channels}, base_frequency = {self.base_frequency},\n'
            f'base_amplitude = {self.base_amplitude},\nphase_shift = {self.phase_shift}')
    
    def __repr__(self):
        return self.__str__()
        
    def generate(
        self,
        drive_def: Dict[int, Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generate the time-dependent coefficient expression for H_x and H_y
        
        Args:
            drive_def: Drive definition. Outer dict maps a channel id to a dict of form
                {'frequency': freq_value, 'amplitude': amplitude}. Argument `'amplitude'` is optional
                (defaults to 1) and can be a float, a string, or a 2-tuple thereof. String arguments
                define C++ functions of t and will be multiplied to `self.base_amplitude[channel]`.
                If a singleton argument is given, it is used for the amplitude of the in-phase (cosine)
                drive with the quadratic (sine) component set to zero. If a 2-tuple is given, the
                elements correspond to the in-phase and quadratic amplitudes, respectively.
                
        Returns:
            coeff_x (str): Expression for H_x
            coeff_y (str): Expression for H_y
        """

        x_terms = []
        y_terms = []

        for ich, def_dict in drive_def.items():
            if self.base_amplitude[ich] == 0.:
                continue
            
            frequency = self.base_frequency - def_dict['frequency']
            
            amplitude = ['0.', '0.']
            
            try:
                amp_factor = def_dict["amplitude"]
            except KeyError:
                amplitude[0] = f'{self.base_amplitude[ich]}'
            else:
                if not isinstance(amp_factor, tuple):
                    amp_factor = (amp_factor, 0.)
                    
                for iq, factor in enumerate(amp_factor):
                    if isinstance(factor, str):
                        if 't' in factor:
                            amplitude[iq] = f'{self.base_amplitude[ich]} * ({factor})'
                        else:
                            amplitude[iq] = f'({self.base_amplitude[ich] * eval(factor)})'
                    else:
                        amplitude[iq] = f'({self.base_amplitude[ich] * factor})'

            xsub = []
            ysub = []
            
            if frequency == 0.:
                cos = np.cos(self.phase_shift[ich])
                sin = np.sin(self.phase_shift[ich])

                if 't' in amplitude[0]:
                    if cos != 0.:
                        xsub.append(f'({cos}) * {amplitude[0]}')
                    if sin != 0.:
                        ysub.append(f'({sin}) * {amplitude[0]}')
                elif eval(amplitude[0]) != 0.:
                    if cos != 0.:
                        xsub.append(f'({cos * eval(amplitude[0])})')
                    if sin != 0.:
                        ysub.append(f'({sin * eval(amplitude[0])})')

                if 't' in amplitude[1]:
                    if sin != 0.:
                        xsub.append(f'({-sin}) * {amplitude[1]}')
                    if cos != 0.:
                        ysub.append(f'({cos}) * {amplitude[1]}')
                elif eval(amplitude[1]) != 0.:
                    if sin != 0.:
                        xsub.append(f'({-sin * eval(amplitude[1])})')
                    if cos != 0.:                
                        ysub.append(f'({cos * eval(amplitude[1])})')
                
            else:
                if self.phase_shift[ich] == 0.:
                    phase_shift = ''
                elif self.phase_shift[ich] > 0.:
                    phase_shift = f' + {self.phase_shift[ich]}'
                else:
                    phase_shift = f' - {abs(self.phase_shift[ich])}'
                
                if 't' in amplitude[0] or eval(amplitude[0]) != 0.:
                    xsub.append(f'{amplitude[0]} * cos({frequency} * t{phase_shift})')
                    ysub.append(f'{amplitude[0]} * sin({frequency} * t{phase_shift})')

                if 't' in amplitude[1] or eval(amplitude[1]) != 0.:
                    xsub.append(f'(-{amplitude[1]}) * sin({frequency} * t{phase_shift})')
                    ysub.append(f'{amplitude[1]} * cos({frequency} * t{phase_shift})')
                    
            if xsub:
                x_terms.append(f'({" + ".join(xsub)})')
            if ysub:
                y_terms.append(f'({" + ".join(ysub)})')
            
        return ' + '.join(x_terms), ' + '.join(y_terms)
    
    def max_frequency(
        self,
        drive_def: Dict[int, Dict[str, Any]]
    ) -> float:
        """Return the maximum frequency for the drive def."""
        
        frequency = 0.
        
        for ich, def_dict in drive_def.items():
            if self.base_amplitude[ich] == 0.:
                continue
            
            frequency = max(frequency, abs(self.base_frequency - def_dict['frequency']))
            
        return frequency


def make_hamiltonian_components(
    qubits: Sequence[int],
    params: Dict[str, Any],
    num_levels: int = 2
) -> Tuple[list, list]:
    r"""Construct the rotating-wave approximation Hamiltonian in the qudit frame.

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
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left[ r_j(t) e^{i (\epsilon_{kj} t + \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + r^{*}_j(t) e^{-i (\epsilon_{kj} t + \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right],
        
    where :math:`\epsilon_{kj} = \omega_k - \nu_j` and :math:`r_j(t) = p_j(t) + i q_j(t)`.
    
    The RWA drive Hamiltonian in the frequency domain is (assuming :math:`\tilde{p}_j` and :math:`\tilde{q}_j` have
    support only around the qudit frequencies)
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left[ \tilde{r}_j(\nu) e^{i [(\omega_k - \nu) t + \phi_{jk}]} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + \tilde{r}^{*}_j(\nu) e^{-i [(\omega_k - \nu) t + \phi_{jk}]} e^{-i \Delta_k N_k t} b_k \right].
    
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
    
    Args:
        qubits: List of :math:`n` qudits to include in the Hamiltonian.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally augmented with
            `'crosstalk'` whose value is an `ndarray` with shape (:math:`n`, :math:`n`) and dtype `complex128`.
            Element `[i, j]` of the matrix corresponds to :math:`\alpha_{jk} e^{i\phi_{jk}}`.
        num_levels: Number of oscillator levels to consider.

    Returns:
        h_int (list): Interaction term components. List of n_coupling elements. Each element is a three-tuple
            (freq, H_x, H_y), where freq is the frequency for the given term. H_x and H_y are the coupling
            Hamiltonians proportional to the "X" and "Y" parts of the oscillating coefficients.
        h_drive (list): Drive term components. List of n_qubits elements. Each element is a three-tuple
            (expr_gen, H_x, H_y), where expr_gen is an instance of DriveExprGen.
    """
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    if 'crosstalk' in params:
        assert params['crosstalk'].shape == (num_qubits, num_qubits), 'Invalid shape of crosstalk matrix'
        assert np.all(np.diagonal(params['crosstalk']) == 1.+0.j), 'Diagonal of crosstalk matrix must be unity'
    else:
        params = copy.deepcopy(params)
        params['crosstalk'] = np.eye(num_qubits, dtype=np.complex128)

    ## Compute the interaction term components

    h_int = []
    
    for iq1, q1 in enumerate(qubits):
        for iq2 in range(iq1 + 1, num_qubits):
            q2 = qubits[iq2]
            try:
                J = params[f'jq{min(q1, q2)}q{max(q1, q2)}']
            except KeyError:
                # no coupling between the qudits
                continue
                
            delta = (params[f'wq{q1}'] - params[f'wq{q2}'])
            
            D1 = params[f'delta{q1}']
            D2 = params[f'delta{q2}']

            for l1 in range(num_levels - 1):
                for l2 in range(num_levels - 1):
                    c1 = np.sqrt(l1 + 1) * qtp.basis(num_levels, l1) * qtp.basis(num_levels, l1 + 1).dag()
                    c2 = np.sqrt(l2 + 1) * qtp.basis(num_levels, l2) * qtp.basis(num_levels, l2 + 1).dag()
                    op = [qtp.qeye(num_levels)] * num_qubits
                    op[iq1] = c1
                    op[iq2] = c2.dag()
                    h_x = J * (qtp.tensor(op).dag() + qtp.tensor(op))
                    h_y = J * 1.j * (qtp.tensor(op).dag() - qtp.tensor(op))
                    freq = delta + D1 * l1 - D2 * l2

                    h_int.append((freq, h_x, h_y))

    ## Compute the drive term components
                    
    h_drive = []
    
    omega_over_two = np.array(list(params[f'omegad{ch}'] for ch in qubits)) / 2.
    
    for iq, qubit in enumerate(qubits):
        for level in range(num_levels - 1):
            op = [qtp.qeye(num_levels)] * num_qubits
            op[iq] = np.sqrt(level + 1) * qtp.basis(num_levels, level) * qtp.basis(num_levels, level + 1).dag()
            h_x = qtp.tensor(op).dag() + qtp.tensor(op)
            h_y = 1.j * (qtp.tensor(op).dag() - qtp.tensor(op))

            expr_gen = DriveExprGen(num_qubits)
            expr_gen.base_frequency = (params[f'wq{qubit}'] + params[f'delta{qubit}'] * level)
            expr_gen.base_amplitude[:] = np.abs(params['crosstalk'][:, iq]) * omega_over_two
            expr_gen.phase_shift[:] = np.angle(params['crosstalk'][:, iq])
                
            h_drive.append((expr_gen, h_x, h_y))

    return h_int, h_drive


def build_pulse_hamiltonian(
    h_int: Sequence[Tuple[float, qtp.Qobj, qtp.Qobj]],
    h_drive: Sequence[Tuple[DriveExprGen, qtp.Qobj, qtp.Qobj]],
    drive_def: Dict[int, Dict[str, Any]]
) -> List[Union[qtp.Qobj, Tuple[qtp.Qobj, str]]]:
    """Build the list of Hamiltonian terms to be passed to the QuTiP solver from a drive pulse definition.
    
    Args:
        h_int: List of interaction term components returned by `make_hamiltonian_components`.
        h_drive: List of drive term components returned by `make_hamiltonian_components`.
        drive_def: Drive definition. See the docstring of DriveExprGen for details.
    
    Returns:
        List of Hamiltonian terms.
    """
    hamiltonian = []
    static_term = qtp.Qobj()
    
    for freq, h_x, h_y in h_int:
        hamiltonian.append((h_x, f'cos({freq} * t)'))
        hamiltonian.append((h_y, f'sin({freq} * t)'))
        
    for expr_gen, h_x, h_y in h_drive:
        coeff_x, coeff_y = expr_gen.generate(drive_def)
        
        if coeff_x:
            if 't' in coeff_x:
                hamiltonian.append((h_x, coeff_x))
            else:
                static_term += (eval(coeff_x) * h_x)
                
        if coeff_y:
            if 't' in coeff_y:
                hamiltonian.append((h_y, coeff_y))
            else:
                static_term += (eval(coeff_y) * h_y)
                
    if static_term.shape != (1, 1):
        # Static term doesn't really have to be at position 0 but QuTiP recommends doing so
        hamiltonian.insert(0, static_term)

    return hamiltonian


def make_tlist(
    h_int: Sequence[Tuple[float, qtp.Qobj, qtp.Qobj]],
    h_drive: Sequence[Tuple[DriveExprGen, qtp.Qobj, qtp.Qobj]],
    drive_def: Dict[int, Dict[str, Any]],
    points_per_cycle: int,
    num_cycles: int
) -> np.ndarray:
    """Generate a list of time points using the maximum frequency in the Hamiltonian.
    
    Args:
        h_int: List of interaction term components returned by `make_hamiltonian_components`.
        h_drive: List of drive term components returned by `make_hamiltonian_components`.
        drive_def: Drive definition. See the docstring of DriveExprGen for details.
        points_per_cycle: Number of points per cycle at the highest frequency.
        num_cycles: Number of overall cycles.
        
    Returns:
        Array of time points.
    """
    frequency_list = list(abs(freq) for freq, _, _ in h_int)
    frequency_list.extend(expr_gen.max_frequency(drive_def) for expr_gen, _, _ in h_drive)
    max_frequency = max(frequency_list)

    if max_frequency == 0.:
        # Single qubit resonant drive -> static Hamiltonian
        coeff_x, coeff_y = h_drive[0][0].generate(drive_def)
        amp2 = eval(coeff_x) ** 2. if coeff_x else 0.
        amp2 += eval(coeff_y) ** 2. if coeff_y else 0.
        tlist = np.linspace(0., 2. * np.pi / np.sqrt(amp2), points_per_cycle * num_cycles)
    else:
        tlist = np.linspace(0., 2. * np.pi / max_frequency * num_cycles, points_per_cycle * num_cycles)
        
    return tlist


def run_pulse_sim(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, Any]],
    psi0: qtp.Qobj = qtp.basis(2, 0),
    tlist: Union[np.ndarray, Tuple[int, int]] = (10, 100),
    e_ops: Optional[Sequence[Any]] = None,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None
) -> qtp.solver.Result:
    """Run a pulse simulation.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of make_hamiltonian_components for details.
        drive_def: Drive definition. See the docstring of DriveExprGen for details.
        psi0: Initial state Qobj.
        tlist: Time points to use in the simulation or a pair `(points_per_cycle, num_cycles)` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
        e_ops: List of observables passed to the QuTiP solver.
        options: QuTiP solver options.
        save_result_to: File name (without the extension) to save the simulation result to.

    Returns:
        Result of running `qutip.sesolve`.
    """

    if isinstance(qubits, int):
        qubits = (qubits,)
    
    # Make the Hamiltonian list
    
    num_sim_levels = psi0.dims[0][0]
    
    h_int, h_drive = make_hamiltonian_components(
        qubits, params,
        num_levels=num_sim_levels)
    
    hamiltonian = build_pulse_hamiltonian(h_int, h_drive, drive_def)
    
    if isinstance(tlist, tuple):
        # List of time slices not given; define using the highest frequency in the Hamiltonian.
        tlist = make_tlist(h_int, h_drive, drive_def, *tlist)

    cd = os.getcwd()
    os.chdir('/tmp')

    result = qtp.sesolve(hamiltonian, psi0, tlist, e_ops=e_ops, options=options)

    os.chdir(cd)
    
    if save_result_to:
        qtp.fileio.qsave(result, save_result_to)
        
    return result