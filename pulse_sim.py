from typing import Any, Dict, Tuple, Sequence
import copy
import string
import numpy as np
import qutip as qtp
import h5py
import scipy.optimize as sciopt

phase_epsilon = 1.e-9

class DriveExprGen:
    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels
        self.base_frequency = 0.
        self.base_amplitude = np.zeros(num_channels)
        self.base_phase = np.zeros(num_channels)
        
    def generate(
        self,
        drive_def: Dict[int, Dict[str, Any]]) -> Tuple[str, str]:
        """Generate the time-dependent coefficient expression for H_cos and H_sin
        
        Args:
            drive_def: Drive definition. Outer dict has the channel id as key and a dict of form
                {'frequency': freq_value, 'phase': phase_value, 'envelope': envelope_func_str}. Arguments
                phase and envelope are optional; if ommitted they default to 0 and '1'.
                
        Returns:
            coeff_cos (str): Expression for H_cos
            coeff_sin (str): Expression for H_sin
        """

        cos_terms = []
        sin_terms = []

        for ich, def_dict in drive_def.items():
            if self.base_amplitude[ich] == 0.:
                continue
            
            frequency = self.base_frequency - def_dict['frequency']
            
            try:
                envelope = def_dict["envelope"]
            except KeyError:
                amplitude = f'{self.base_amplitude[ich]}'
            else:
                if type(envelope) is str and 't' in envelope:
                    amplitude = f'{self.base_amplitude[ich]} * ({envelope})'
                else:
                    amplitude = f'({self.base_amplitude[ich] * envelope})'

            try:
                phase = self.base_phase[ich] - def_dict['phase']
            except KeyError:
                phase = self.base_phase[ich]
                
            while phase > 2. * np.pi:
                phase -= 2. * np.pi
            while phase < 0.:
                phase += 2. * np.pi
                
            if frequency == 0.:
                if np.abs(phase) < phase_epsilon:
                    cos_terms.append(f'{amplitude}')
                elif np.abs(phase - np.pi / 2.) < phase_epsilon:
                    sin_terms.append(f'{amplitude}')
                elif np.abs(phase - np.pi) < phase_epsilon:
                    cos_terms.append(f'(-{amplitude})')
                elif np.abs(phase - 3. * np.pi / 2.) < phase_epsilon:
                    sin_terms.append(f'(-{amplitude})')
                elif 't' in amplitude:
                    cos_terms.append(f'({np.cos(phase)}) * {amplitude}')
                    sin_terms.append(f'({np.sin(phase)}) * {amplitude}')
                else:
                    cos_terms.append(f'({np.cos(phase) * eval(amplitude)})')
                    sin_terms.append(f'({np.sin(phase) * eval(amplitude)})')
                
            else:
                if np.abs(phase) < phase_epsilon:
                    cos_terms.append(f'{amplitude} * cos({frequency} * t)')
                    sin_terms.append(f'{amplitude} * sin({frequency} * t)')
                elif np.abs(phase - np.pi / 2.) < phase_epsilon:
                    cos_terms.append(f'(-{amplitude}) * sin({frequency} * t)')
                    sin_terms.append(f'{amplitude} * cos({frequency} * t)')
                elif np.abs(phase - np.pi) < phase_epsilon:
                    cos_terms.append(f'(-{amplitude}) * cos({frequency} * t)')
                    sin_terms.append(f'(-{amplitude}) * sin({frequency} * t)')
                elif np.abs(phase - 3. * np.pi / 2.) < phase_epsilon:
                    cos_terms.append(f'{amplitude} * sin({frequency} * t)')
                    sin_terms.append(f'(-{amplitude}) * cos({frequency} * t)')
                else:
                    cos_terms.append(f'{amplitude} * cos({frequency} * t + {phase})')
                    sin_terms.append(f'{amplitude} * sin({frequency} * t + {phase})')
            
        return ' + '.join(cos_terms), ' + '.join(sin_terms)
            

def make_rwa_hamiltonian(
    qubits: Sequence[int],
    params: Dict[str, Any],
    crosstalk_matrix: np.ndarray,
    num_levels: int = 2) -> Tuple[list, list]:
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
        
        H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j s_j (t) \cos (\nu_j t + \psi_j) \left( e^{i\phi_{jk}} b_k^{\dagger} + e^{-i\phi_{jk}} b_k \right),
        
    with
    
    - :math:`\omega_j`: Frequency of the :math:`j`th qubit
    - :math:`\Delta_j`: Anharmonicity of the :math:`j`th qudit
    - :math:`J_{jk}`: Coupling between qudits :math:`j` and :math:`k`
    - :math:`\alpha_{jk}`: Crosstalk attenuation factor of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\phi_{jk}`: Crosstalk phase shift of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\Omega_j`: Base amplitude of drive in channel :math:`j`
    - :math:`\s_j (t)`: Pulse envelope of drive in channel :math:`j`
    - :math:`\nu_j`: Local oscillator frequency of drive in channel :math:`j`
    - :math:`\psi_j`: Local oscillator phase offset in channel :math:`j`
    
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
        \left. e^{-i \delta_{jk} t} e^{-i [\Delta_j N_j - \Delta_k (N_k - 1)] t} b_j b_k^{\dagger} \right).

    The drive Hamiltonian is
    
    .. math::
    
        \tilde{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j s_j (t) \cos (\nu_j t + \psi_j) \left( e^{i (\omega_k t + \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. e^{-i (\omega_k t + \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right),
    
    and with rotating wave approximation
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} s_j (t) \left( e^{i (\epsilon_{kj} t + \phi_{jk} - \psi_j)} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. e^{-i (\epsilon_{kj} t + \phi_{jk} - \psi_j)} e^{-i \Delta_k N_k t} b_k \right),
        
    where :math:`\epsilon_{kj} = \omega_k - \nu_j`.
    
    **QuTiP implementation:**
    
    Time-dependent Hamiltonian in QuTiP is represented by a two-tuple `(H, c(t))` where `H` is a static Qobj and `c(t)`
    is the time-dependent coefficient of `H`. This function returns two lists of tuples, corresponding to the interaction
    and drive Hamiltonians, with the total list length corresponding to the number of distinct frequencies in the RWA
    Hamiltonian :math:`\tilde{H}_{\mathrm{int}} + \bar{H}_{\mathrm{d}}`. Because `c(t)` must be a real function, each
    returned tuple contains two Qobjs corresponding to the "cosine" and "sine" parts of the Hamiltonian oscillating at
    the given frequency.
        
    **TODO: Qubit optimization:**
    
    When only considering qubits, anharmonicity terms are dropped, making the formulae above somewhat simpler. In particular,
    when one or more drives are resonant with qubit frequencies, some terms in the interaction and drive Hamiltonians
    will have common frequencies (i.e. :math:`\delta_{jk}` and `\epsilon_{kj}` coincide). We should look into how to detect
    and exploit such cases in the future.
    
    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`.
        crosstalk_matrix: Crosstalk factor matrix. Element `[i, j]` corresponds to :math:`\alpha_{jk} e^{i\phi_{jk}}`.
        num_levels: Number of oscillator levels to consider.

    Returns:
        h_int (list): Interaction term Hamiltonians. List of n_coupling elements. Each element is a list of three-tuples
            (freq, H_cos, H_sin), where freq is the frequency for the given term. The inner list runs over
            different combinations of creations and annihilations for various qudit levels. H_cos and H_sin
            are the coupling Hamiltonians proportional to the "cosine" and "sine" parts of the oscillating coefficients.
        h_drive (list): Drive term Hamiltonians. List of n_qubits elements. Each element is a list of three-tuples
            (expr_gen, H_cos, H_sin), where expr_gen is an instance of DriveExprGen.
    """
    
    if type(qubits) is int:
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    assert crosstalk_matrix.shape == (num_qubits, num_qubits)
    assert np.all(np.diagonal(crosstalk_matrix) == 1.+0.j)

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
                    c1 = np.sqrt(l1 + 1) * qtp.basis(num_levels, l1 + 1) * qtp.basis(num_levels, l1).dag()
                    c2 = np.sqrt(l2 + 1) * qtp.basis(num_levels, l2 + 1) * qtp.basis(num_levels, l2).dag()
                    op = [qtp.qeye(num_levels)] * num_qubits
                    op[iq1] = c1
                    op[iq2] = c2.dag()
                    h_cos = J * (qtp.tensor(op) + qtp.tensor(op).dag())
                    h_sin = J * 1.j * (qtp.tensor(op) - qtp.tensor(op).dag())
                    freq = delta + D1 * l1 - D2 * l2

                    h_int.append((freq, h_cos, h_sin))

    h_drive = []
    
    omega_over_two = np.array(list(params[f'omegad{ch}'] for ch in qubits)) / 2.
    
    for iq, qubit in enumerate(qubits):
        for level in range(num_levels - 1):
            op = [qtp.qeye(num_levels)] * num_qubits
            op[iq] = np.sqrt(level + 1) * qtp.basis(num_levels, level + 1) * qtp.basis(num_levels, level).dag()
            h_cos = qtp.tensor(op) + qtp.tensor(op).dag()
            h_sin = 1.j * (qtp.tensor(op) - qtp.tensor(op).dag())

            expr_gen = DriveExprGen(num_qubits)
            expr_gen.base_frequency = (params[f'wq{qubit}'] + params[f'delta{qubit}'] * level)
            expr_gen.base_amplitude[:] = np.abs(crosstalk_matrix[:, iq]) * omega_over_two
            expr_gen.base_phase[:] = np.angle(crosstalk_matrix[:, iq])
                
            h_drive.append((expr_gen, h_cos, h_sin))

    return h_int, h_drive


def run_pulse_sim(
    qubits: Sequence[int],
    params: Dict[str, Any],
    crosstalk_matrix: np.ndarray,
    drive_def: Dict[int, Dict[str, Any]],
    psi0: qtp.Qobj = qtp.basis(2, 0),
    tlist: Optional[np.ndarray] = None,
    save_result_to: Optional[str] = None) -> qtp.solver.Result:
    """Run a pulse simulation and return the final state Qobj.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`.
        crosstalk_matrix: Crosstalk factor matrix. Element `[i, j]` corresponds to :math:`\alpha_{jk} e^{i\phi_{jk}}`.
        drive_def: Drive definition. Outer dict has the channel id as key and a dict of form
                `{'frequency': freq_value, 'phase': phase_value, 'envelope': envelope_func_str}`. Arguments `phase` and
                `envelope` are optional; if ommitted they default to 0 and '1'.
        psi0: Initial state Qobj.
        tlist: Time points to use in the simulation.
        save_result_to: File name (without the extension) to save the simulation result to.

    Returns:
        Result of running `qutip.sesolve`.
    """

    if type(qubits) is int:
        qubits = (qubits,)
    
    # Make the Hamiltonian list
    
    num_sim_levels = psi0.dims[0][0]
    
    h_int, h_drive = make_rwa_hamiltonian(qubits,
                                          params,
                                          crosstalk_matrix,
                                          num_sim_levels)
    
    hamiltonian = []
    
    for freq, h_cos, h_sin in h_int:
        hamiltonian.append((h_cos, f'cos({freq} * t)'))
        hamiltonian.append((h_sin, f'sin({freq} * t)'))
        
    for expr_gen, h_cos, h_sin in h_drive:
        coeff_cos, coeff_sin = expr_gen.generate(drive_def)
        
        if coeff_cos:
            if 't' in coeff_cos:
                hamiltonian.append((h_cos, coeff_cos))
            else:
                hamiltonian.append(eval(coeff_cos) * h_cos)
                
        if coeff_sin:
            if 't' in coeff_sin:
                hamiltonian.append((h_sin, coeff_sin))
            else:
                hamiltonian.append(eval(coeff_sin) * h_sin)
            
    if tlist is None:
        # List of time slices not given; default to 100 cycles under the first frequency in h_int
        # and 10 time slices per cycle
        tlist = np.linspace(0., 2. * np.pi / h_int[0][0] * 100, 1000)
        
    result = qtp.sesolve(hamiltonian, psi0, tlist)
    
    if save_result_to:
        qtp.fileio.qsave(result, save_result_to)
        
    return result
    
    
def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    crosstalk_matrix: np.ndarray,
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None) -> np.ndarray:
    """Run a pulse simulation with constant envelopes and extract the Pauli components of the effective Hamiltonian.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`.
        crosstalk_matrix: Crosstalk factor matrix. Element `[i, j]` corresponds to :math:`\alpha_{jk} e^{i\phi_{jk}}`.
        drive_def: Drive definition. Outer dict has the channel id as key and a dict of form
                `{'frequency': freq_value, 'amplitude': amplitude_factor, 'phase': phase_value}`. Note the definition
                must contain an argument `amplitude` instead of `envelope`. Argument `phase` is optional and defaults
                to 0 if ommitted.
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
        :math:`\lambda_i \otimes \lambda_j \otimes \dots` of the effective Hamiltonian.
    """
    
    assert comp_dim <= num_sim_levels
    
    if type(qubits) is int:
        qubits = (qubits,)
        
    num_qubits = len(qubits)

    sim_drive_def = copy.deepcopy(drive_def)
    for key, value in drive_def.items():
        sim_drive_def[key]['envelope'] = f'{value["amplitude"]}'

    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)
    
    frequencies = []
    iq = 0
    while True:
        try:
            freq = params[f'wq{iq}']
        except KeyError:
            break
            
        frequencies.append(freq)
        iq += 1
        
    frequencies = np.array(frequencies)
    max_delta = np.amax(np.abs(np.add.outer(frequencies, -frequencies)))
            
    # 400 cycles with the maximum frequency difference; 10 points per cycle
    tlist = np.linspace(0., 2. * np.pi / max_delta * 400, 4000)

    result = run_pulse_sim(qubits,
                           params,
                           crosstalk_matrix,
                           sim_drive_def,
                           psi0,
                           tlist,
                           save_result_to)
    
    ## Take the log of the time evolution operator

    unitaries = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    eigvals, eigcols = np.linalg.eig(unitaries)
    eigrows = np.conjugate(np.transpose(eigcols, axes=(0, 2, 1)))
    
    omega_t = -np.angle(eigvals) # list of energy eigenvalues (mod 2pi) times t

    # Find the first t where an eigenvalue does a 2pi jump
    omega_min = np.amin(omega_t, axis=1)
    omega_max = np.amax(omega_t, axis=1)

    margin = 0.01

    min_hits_minus_pi = np.asarray(omega_min < -np.pi + margin).nonzero()[0]
    if len(min_hits_minus_pi) == 0:
        tmax_min = omega_t.shape[0]
    else:
        tmax_min = min_hits_minus_pi[0]
    
    max_hits_pi = np.asarray(omega_max > np.pi - margin).nonzero()[0]
    if len(max_hits_pi) == 0:
        tmax_max = omega_t.shape[0]
    else:
        tmax_max = max_hits_pi[0]
        
    tmax = min(tmax_min, tmax_max)
    
    # Will only consider t upto this discontinuity from now on
    
    heff_t = (eigcols[:tmax] * np.tile(np.expand_dims(omega_t[:tmax], axis=1), (1, omega_t.shape[1], 1))) @ eigrows[:tmax]
    
    ## Extract the (generalized) Pauli components
    
    num_paulis = comp_dim ** 2
    
    paulis = np.zeros((num_paulis, num_sim_levels, num_sim_levels), dtype=np.complex128)
    paulis[0, :comp_dim, :comp_dim] = np.diagflat(np.ones(comp_dim)) / np.sqrt(comp_dim)
    
    ip = 1
    unit = 1. / np.sqrt(2.)
    for idim in range(1, comp_dim):
        for irow in range(idim):
            paulis[ip, irow, idim] = unit
            paulis[ip, idim, irow] = unit
            ip += 1
            paulis[ip, irow, idim] = -unit * 1.j
            paulis[ip, idim, irow] = unit * 1.j
            ip += 1

        paulis[ip, :idim + 1, :idim + 1] = np.diagflat(np.array([1.] * idim + [-idim]))
        paulis[ip] /= np.sqrt(np.trace(paulis[ip] * paulis[ip]))
        ip += 1

    # Use of einsum implies that we can deal with at most 52 // 3 = 17 qubits
    al = string.ascii_letters
    indices_in = []
    indices_out = [''] * 3
    for il in range(0, num_qubits * 3, 3):
        indices_in.append(al[il:il + 3])
        indices_out[0] += al[il]
        indices_out[1] += al[il + 1]
        indices_out[2] += al[il + 2]

    indices = f'{",".join(indices_in)}->{"".join(indices_out)}'
    shape = [num_paulis] * num_qubits + [num_sim_levels ** num_qubits] * 2
    pauli_basis = np.einsum(indices, *([paulis] * num_qubits)).reshape(*shape)

    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    pauli_coeffs_t = np.einsum(f'txy,{al[:num_qubits]}yx->{al[:num_qubits]}t', heff_t[:tmax], pauli_basis)
    
    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('pauli_coeffs_t', data=pauli_coeffs_t)
    
    ## Do a linear fit for each component
    
    pauli_coeffs = np.zeros(num_paulis ** num_qubits)

    line = lambda a, x: a * x
    for ic, coeffs_t in enumerate(pauli_coeffs_t.reshape(num_paulis ** num_qubits, tmax)):
        popt, _ = sciopt.curve_fit(line, tlist[:tmax], coeffs_t)
        pauli_coeffs[ic] = popt[0]

    return pauli_coeffs.reshape(*([num_paulis] * num_qubits))
