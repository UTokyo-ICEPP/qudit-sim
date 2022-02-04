from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import sys
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
        
    def __str__(self):
        return (f'DriveExprGen: num_channels = {self.num_channels}, base_frequency = {self.base_frequency},\n'
            f'base_amplitude = {self.base_amplitude},\nbase_phase = {self.base_phase}')
    
    def __repr__(self):
        return self.__str__()
        
    def generate(
        self,
        drive_def: Dict[int, Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Generate the time-dependent coefficient expression for H_cos and H_sin
        
        Args:
            drive_def: Drive definition. Outer dict maps a channel id to a dict of form
                {'frequency': freq_value, 'phase': phase_value, 'amplitude': amplitude}. Argument
                `'amplitude'` can be a float or a string defining a C++ function of t and will be
                multiplied to `self.base_amplitude[channel]`. Arguments `'phase'` and `'amplitude'`
                are optional; if ommitted they default to 0 and 1.
                
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
                amp_factor = def_dict["amplitude"]
            except KeyError:
                amplitude = f'{self.base_amplitude[ich]}'
            else:
                if isinstance(amp_factor, str):
                    if 't' in amplitude:
                        amplitude = f'{self.base_amplitude[ich]} * ({amp_factor})'
                    else:
                        amplitude = f'({self.base_amplitude[ich] * eval(amp_factor)})'
                else:
                    amplitude = f'({self.base_amplitude[ich] * amp_factor})'

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
        \left. + e^{-i \delta_{jk} t} e^{-i [\Delta_j N_j - \Delta_k (N_k - 1)] t} b_j b_k^{\dagger} \right).

    The drive Hamiltonian is
    
    .. math::
    
        \tilde{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j s_j (t) \cos (\nu_j t + \psi_j) \left( e^{i (\omega_k t + \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + e^{-i (\omega_k t + \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right),
    
    and with rotating wave approximation
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} s_j (t) \left( e^{i (\epsilon_{kj} t + \phi_{jk} - \psi_j)} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + e^{-i (\epsilon_{kj} t + \phi_{jk} - \psi_j)} e^{-i \Delta_k N_k t} b_k \right),
        
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
        qubits: List of :math:`n` qudits to include in the Hamiltonian.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally augmented with
            `'crosstalk'` whose value is an `ndarray` with shape (:math:`n`, :math:`n`) and dtype `complex128`.
            Element `[i, j]` of the matrix corresponds to :math:`\alpha_{jk} e^{i\phi_{jk}}`.
        num_levels: Number of oscillator levels to consider.

    Returns:
        h_int (list): Interaction term components. List of n_coupling elements. Each element is a three-tuple
            (freq, H_cos, H_sin), where freq is the frequency for the given term. H_cos and H_sin
            are the coupling Hamiltonians proportional to the "cosine" and "sine" parts of the oscillating coefficients.
        h_drive (list): Drive term components. List of n_qubits elements. Each element is a three-tuple
            (expr_gen, H_cos, H_sin), where expr_gen is an instance of DriveExprGen.
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
                    c1 = np.sqrt(l1 + 1) * qtp.basis(num_levels, l1 + 1) * qtp.basis(num_levels, l1).dag()
                    c2 = np.sqrt(l2 + 1) * qtp.basis(num_levels, l2 + 1) * qtp.basis(num_levels, l2).dag()
                    op = [qtp.qeye(num_levels)] * num_qubits
                    op[iq1] = c1
                    op[iq2] = c2.dag()
                    h_cos = J * (qtp.tensor(op) + qtp.tensor(op).dag())
                    h_sin = J * 1.j * (qtp.tensor(op) - qtp.tensor(op).dag())
                    freq = delta + D1 * l1 - D2 * l2

                    h_int.append((freq, h_cos, h_sin))

    ## Compute the drive term components
                    
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
            expr_gen.base_amplitude[:] = np.abs(params['crosstalk'][:, iq]) * omega_over_two
            expr_gen.base_phase[:] = np.angle(params['crosstalk'][:, iq])
                
            h_drive.append((expr_gen, h_cos, h_sin))

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
    
    for freq, h_cos, h_sin in h_int:
        hamiltonian.append((h_cos, f'cos({freq} * t)'))
        hamiltonian.append((h_sin, f'sin({freq} * t)'))
        
    for expr_gen, h_cos, h_sin in h_drive:
        coeff_cos, coeff_sin = expr_gen.generate(drive_def)
        
        if coeff_cos:
            if 't' in coeff_cos:
                hamiltonian.append((h_cos, coeff_cos))
            else:
                static_term += (eval(coeff_cos) * h_cos)
                
        if coeff_sin:
            if 't' in coeff_sin:
                hamiltonian.append((h_sin, coeff_sin))
            else:
                static_term += (eval(coeff_sin) * h_sin)
                
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
        coeff_cos, coeff_sin = h_drive[0][0].generate(drive_def)
        amp2 = eval(coeff_cos) ** 2. if coeff_cos else 0.
        amp2 += eval(coeff_sin) ** 2. if coeff_sin else 0.
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
    
    result = qtp.sesolve(hamiltonian, psi0, tlist, e_ops=e_ops, options=options)
    
    if save_result_to:
        qtp.fileio.qsave(result, save_result_to)
        
    return result


def make_generalized_paulis(
    comp_dim: int = 2,
    matrix_dim: Optional[int] = None
) -> np.ndarray:
    """Return a list of normalized generalized Pauli matrices of given dimension as a numpy array.
    
    Args:
        comp_dim: Dimension of the Pauli matrices.
        matrix_dim: Dimension of the containing matrix, which may be greater than `comp_dim`.
        
    Returns:
        The full list of Pauli matrices as an array of dtype `complex128` and shape
            `(comp_dim ** 2, matrix_dim, matrix_dim)`.
    """
    
    if matrix_dim is None:
        matrix_dim = comp_dim
        
    assert matrix_dim >= comp_dim, 'Matrix dimension cannot be smaller than Pauli dimension'
    
    num_paulis = comp_dim ** 2
    
    paulis = np.zeros((num_paulis, matrix_dim, matrix_dim), dtype=np.complex128)
    paulis[0, :comp_dim, :comp_dim] = np.diagflat(np.ones(comp_dim)) / np.sqrt(comp_dim)
    ip = 1
    for idim in range(1, comp_dim):
        for irow in range(idim):
            paulis[ip, irow, idim] = 1. / np.sqrt(2.)
            paulis[ip, idim, irow] = 1. / np.sqrt(2.)
            ip += 1
            paulis[ip, irow, idim] = -1.j / np.sqrt(2.)
            paulis[ip, idim, irow] = 1.j / np.sqrt(2.)
            ip += 1

        paulis[ip, :idim + 1, :idim + 1] = np.diagflat(np.array([1.] * idim + [-idim]))
        paulis[ip] /= np.sqrt(np.trace(paulis[ip] * paulis[ip]))
        ip += 1
        
    return paulis


def make_prod_basis(
    basis: np.ndarray,
    num_qubits: int
) -> np.ndarray:
    """Return a list of basis matrices of multi-qubit operators.
    
    Args:
        basis: Basis of single-qubit operators. Array with shape
            `(num_basis, matrix_dim, matrix_dim)`.
        num_qubits: Number of qubits.
        
    Returns:
        The full list of basis matrices as a single `num_qubits + 2`-dimensional array.
            The first `num_qubits` dimensions are size `num_basis`, and the last two are
            size `matrix_dim ** num_qubits`.
    """
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
    
    shape = [basis.shape[0]] * num_qubits + [basis.shape[1] ** num_qubits] * 2
    
    return np.einsum(indices, *([basis] * num_qubits)).reshape(*shape)


def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    fit_tol: float = 0.001,
    save_result_to: Optional[str] = None
) -> np.ndarray:
    """Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of make_hamiltonian_components for details.
        drive_def: Drive definition. See the docstring of DriveExprGen for details. Argument `'amplitude'` for
            each channel must be a constant expression (float or string).
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
        :math:`\lambda_i \otimes \lambda_j \otimes \dots` of the effective Hamiltonian.
    """
    
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    for key, value in drive_def.items():
        try:
            amp_factor = value['amplitude']
        except KeyError:
            raise RuntimeError(f'Missing amplitude specification for drive on channel {key}')

        if isinstance(amp_factor, str) and 't' in amp_factor:
            raise RuntimeError(f'Cannot use time-dependent amplitude (found in channel {key})')
            
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)

    result = run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=(10, 400),
        save_result_to=save_result_to)
    
    ## Take the log of the time evolution operator

    unitaries = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    eigvals, eigcols = np.linalg.eig(unitaries)
    eigrows = np.conjugate(np.transpose(eigcols, axes=(0, 2, 1)))
    
    omega_t = -np.angle(eigvals) # list of energy eigenvalues (mod 2pi) times t

    # Find the first t where an eigenvalue does a 2pi jump
    omega_min = np.amin(omega_t, axis=1)
    omega_max = np.amax(omega_t, axis=1)

    margin = 0.1

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
    
    # Will only consider t up to this discontinuity from now on
    
    heff_t = (eigcols[:tmax] * np.tile(np.expand_dims(omega_t[:tmax], axis=1), (1, omega_t.shape[1], 1))) @ eigrows[:tmax]
    
    ## Extract the (generalized) Pauli components
    
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)
    prod_basis = make_prod_basis(paulis, num_qubits)

    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    pauli_coeffs_t = np.einsum(f'txy,{qubit_indices}yx->t{qubit_indices}', heff_t[:tmax], prod_basis).real

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('pauli_coeffs_t', data=pauli_coeffs_t)
    
    ## Do a linear fit to each component
    
    num_paulis = paulis.shape[0]
    
    pauli_coeffs = np.zeros(num_paulis ** num_qubits)

    line = lambda a, x: a * x
    for ic, coeffs_t in enumerate(pauli_coeffs_t.reshape(num_paulis ** num_qubits, tmax)):
        # Iteratively determine the interval that yields a fit within tolerance
        xdata = result.times[:tmax]
        ydata = coeffs_t
        while True:
            popt, _ = sciopt.curve_fit(line, xdata, ydata)
            if abs(np.sum(ydata - popt[0] * xdata) / np.sum(ydata)) < fit_tol:
                break
                
            start = int(xdata.shape[0] * 0.1)
            end = int(xdata.shape[0] * 0.9)
            xdata = xdata[start:end]
            ydata = ydata[start:end]
            if xdata.shape[0] <= 10:
                sys.stderr.write(f'Linear fit for {ic}th pauli coefficient did not yield a reliable result.'
                                'Run the function again with the save_result_to option and check the raw output.\n')
                popt = np.array([0.])
                break
                
        pauli_coeffs[ic] = popt[0]

    return pauli_coeffs.reshape(*([num_paulis] * num_qubits))


def find_gate(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    tlist: Union[np.ndarray, Tuple[int, int]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None
) -> np.ndarray:
    """Run a pulse simulation and return the log of the resulting unitary.
    
    This function computes the time evolution operator :math:`U_{\mathrm{pulse}}` effected by the drive pulse
    and returns :math:`i \log U_{\mathrm{pulse}}`, projected onto the computational space if the simulation is
    performed with more levels than computational dimension. The returned value is given as an array of Pauli
    coefficients.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of make_hamiltonian_components for details.
        drive_def: Drive definition. See the docstring of DriveExprGen for details.
        tlist: Time points to use in the simulation. See the docstring of run_pulse_sim for details.
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
        :math:`\lambda_i \otimes \lambda_j \otimes \dots` in :math:`i log U_{\mathrm{pulse}}`.
    """
    
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    for key, value in drive_def.items():
        try:
            amp_factor = value['amplitude']
        except KeyError:
            raise RuntimeError(f'Missing amplitude specification for drive on channel {key}')

        if isinstance(amp_factor, str) and 't' in amp_factor:
            raise RuntimeError(f'Cannot use time-dependent amplitude (found in channel {key})')
            
    ## Evolve the identity operator to obtain the evolution operator corresponding to the pulse
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)

    result = run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=(10, 400),
        save_result_to=save_result_to)
    
    ## Take the log of the evolution operator

    # Apparently sesolve always store the states for all time points regardless of the options..
    unitary = result.states[-1]
    
    eigvals, eigcols = np.linalg.eig(unitary)
    eigrows = np.conjugate(np.transpose(eigcols))
    
    ilog_diagonal = np.diag(-np.angle(eigvals))

    ilog_u = eigcols @ ilog_diagonal @ eigrows
    
    ## Extract the (generalized) Pauli components
    
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)
    prod_basis = make_prod_basis(paulis, num_qubits)
    
    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    pauli_coeffs = np.einsum(f'xy,{qubit_indices}yx->{qubit_indices}', ilog_u, prod_basis).real
    
    return pauli_coeffs
