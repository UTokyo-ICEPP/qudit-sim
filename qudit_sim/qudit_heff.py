from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import string
import sys
import numpy as np
import h5py
import scipy.optimize as sciopt
import qutip as qtp

from .paulis import make_generalized_paulis, make_prod_basis
from .pulse_sim import run_pulse_sim

def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    fit_tol: float = 0.01,
    save_result_to: Optional[str] = None,
    warn_fit_failure: bool = True
) -> np.ndarray:
    """Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.
    
    QuTiP `sesolve` applied to the identity matrix will give the time evolution operator :math:`U_H(t)` according
    to the rotating-wave Hamiltonian :math:`H` at each time point. If an effective Hamiltonian 
    :math:`H_{\mathrm{eff}}` is to be found, the evolution should be approximatable with
    :math:`\exp(-i H_{\mathrm{eff}} t)`. This function takes the matrix-log of calculated :math:`U_H(t)`, extracts
    the Pauli coefficients at each time point, and performs a linear fit to each coefficient as a function of time.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of `pulse_sim.make_hamiltonian_components` for details.
        drive_def: Drive definition. See the docstring of `pulse_sim.DriveExprGen` for details. Argument `'amplitude'` for
            each channel must be a constant expression (float or string).
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        fit_tol: Tolerance factor for the linear fit. The function tries to iteratively find a time interval
            where the best fit line `f(t)` satisfies `abs(sum(U(t) - f(t)) / sum(U(t))) < fit_tol`.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
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
    
    heff_t = (eigcols * np.tile(omega_t[:, np.newaxis], (1, omega_t.shape[1], 1))) @ eigrows
    
    ## Extract the (generalized) Pauli components
    
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)
    prod_basis = make_prod_basis(paulis, num_qubits)

    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    pauli_coeffs_t = np.einsum(f'txy,{qubit_indices}yx->t{qubit_indices}', heff_t, prod_basis).real
    # Divide the trace by two to account for the normalization of the generalized Paulis
    pauli_coeffs_t /= 2.

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('omega', data=omega_t)
            out.create_dataset('eigcols', data=eigcols)
            out.create_dataset('pauli_coeffs', data=pauli_coeffs_t)
            out.create_dataset('tlist', data=result.times)
            out.create_dataset('tmax', data=np.array([tmax]))
            out.create_dataset('fit_success', shape=pauli_coeffs_t.shape[1:], dtype='i')
            out.create_dataset('fit_range', shape=(pauli_coeffs_t.shape[1:] + (2,)), dtype='i')
            out.create_dataset('fit_residual', shape=pauli_coeffs_t.shape[1:], dtype='f8')
    
    ## Do a linear fit to each component
    
    num_paulis = paulis.shape[0]
    
    pauli_coeffs = np.zeros(pauli_coeffs_t.shape[1:])
    
    # This is probably not the most numpythonic way of indexing the array..
    time_series_list = pauli_coeffs_t.reshape(pauli_coeffs_t.shape[0], np.prod(pauli_coeffs.shape)).T

    line = lambda a, x: a * x
    for ic, coeffs_t in enumerate(time_series_list):
        icm = np.unravel_index(ic, pauli_coeffs.shape)
        
        # Iteratively determine the interval that yields a fit within tolerance
        start = 0
        end = tmax
        min_residual = None
        while True:
            xdata = result.times[start:end]
            ydata = coeffs_t[start:end]
            
            if xdata.shape[0] <= 10:
                if warn_fit_failure:
                    sys.stderr.write(f'Linear fit for coefficient of {icm} did not yield a'
                                     f' reliable result (minimum residual = {min_residual}).\n')
                
                if save_result_to:
                    with h5py.File(f'{save_result_to}.h5', 'a') as out:
                        out['fit_success'][icm] = 0
                        out['fit_range'][icm] = [start, end]
                        out['fit_residual'][icm] = residual
                elif warn_fit_failure:
                    sys.stderr.write(' Run the function with the save_result_to option and'
                                     ' check the raw output.\n')
                popt = np.array([0.])
                break
            
            popt, _ = sciopt.curve_fit(line, xdata, ydata)
            
            residual = abs(np.sum(ydata - popt[0] * xdata) / np.sum(ydata))
            if min_residual is None or residual < min_residual:
                min_residual = residual
                
            if residual < fit_tol:
                if save_result_to:
                    with h5py.File(f'{save_result_to}.h5', 'a') as out:
                        out['fit_success'][icm] = 1
                        out['fit_range'][icm] = [start, end]
                        out['fit_residual'][icm] = residual
                    
                break
                
            start += int(xdata.shape[0] * 0.1)
            end -= int(xdata.shape[0] * 0.1)
                
        pauli_coeffs[icm] = popt[0]

    return pauli_coeffs


def find_gate(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    tlist: np.ndarray,
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
        params: Hamiltonian parameters. See the docstring of `pulse_sim.make_hamiltonian_components` for details.
        drive_def: Drive definition. See the docstring of `pulse_sim.DriveExprGen` for details.
        tlist: Time points to use in the simulation.
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

    ## Evolve the identity operator to obtain the evolution operator corresponding to the pulse
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)

    result = run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=tlist,
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
    # Divide the trace by two to account for the normalization of the generalized Paulis
    pauli_coeffs /= 2.
    
    return pauli_coeffs
