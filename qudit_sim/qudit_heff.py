from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable
import os
import sys
import time
import string
import collections
import logging
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy
import h5py
import qutip as qtp

from .paulis import (get_num_paulis, make_generalized_paulis, make_prod_basis,
                    unravel_basis_index, get_l0_projection)
from .pulse_sim import run_pulse_sim

def matrix_ufunc(
    op: Callable[[np.ndarray], np.ndarray],
    mat: np.ndarray,
    with_diagonals: bool = False
) -> np.ndarray:
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.
    
    The argument `mat` must be an array of normal matrices (in the last two dimensions). This function
    unitary-diagonalizes the matrices, applies `op` to the diagonals, and inverts the diagonalization.
    
    Args:
        op: Unary operator to be applied to the diagonals of `mat`.
        mat: Array of normal matrices (shape (..., n, n)). No check on normality is performed.
        with_diagonals: If True, also return the array `op(eigenvalues)`.

    Returns:
        An array corresponding to `op(mat)`. If `diagonals==True`, another array corresponding to `op(eigvals)`.
    """
    eigvals, eigcols = np.linalg.eig(mat)
    eigrows = np.conjugate(np.moveaxis(eigcols, -2, -1))

    op_eigvals = op(eigvals)
    
    op_mat = np.matmul(eigcols * op_eigvals[..., None, :], eigrows)

    if with_diagonals:
        return op_mat, op_eigvals
    else:
        return op_mat


def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    num_cycles: int = 400,
    max_com: float = 20.,
    min_coeff_ratio: float = 0.005,
    num_update: int = 0,
    max_iterations: int = 100,
    save_result_to: Optional[str] = None,
    save_iterations: bool = False,
    log_level: int = logging.WARNING
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
        num_cycles: Duration of the square pulse, in units of cycles in the highest frequency appearing in the Hamiltonian.
        max_com: Convergence condition.
        min_coeff_ratio: Convergence condition.
        num_update: Number of coefficients to update per iteration. If <= 0, all candidate coefficients are updated.
        max_iterations: Maximum number of fit & subtract iterations.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        save_iterations: If True, save detailed intermediate data from fit iterations.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
    """
    if isinstance(qubits, int):
        qubits = (qubits,)
    
    result = run_pulse_sim_for_heff(
        qubits,
        params,
        drive_def,
        num_sim_levels,
        num_cycles,
        save_result_to,
        log_level)
    
    heff_coeffs = find_heff_from(
        result,
        len(qubits),
        comp_dim,
        max_com,
        min_coeff_ratio,
        num_update,
        max_iterations,
        save_result_to,
        save_iterations,
        log_level)
    
    return heff_coeffs


def run_pulse_sim_for_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    num_cycles: int = 400,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING    
):
    logging.getLogger().setLevel(log_level)
    
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    for key, value in drive_def.items():
        try:
            amp_factor = value['amplitude']
        except KeyError:
            raise RuntimeError(f'Missing amplitude specification for drive on channel {key}')

        if isinstance(amp_factor, str) and 't' in amp_factor:
            raise RuntimeError(f'Cannot use time-dependent amplitude (found in channel {key})')
            
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)
    
    logging.info('Running a square pulse simulation for %d cycles', num_cycles)

    return run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=(10, num_cycles),
        save_result_to=save_result_to)


def find_heff_from(
    result: qtp.solver.Result,
    num_qubits: int = 1,
    comp_dim: int = 2,
    max_com: float = 20.,
    min_coeff_ratio: float = 0.005,
    num_update: int = 0,
    max_iterations: int = 100,
    save_result_to: Optional[str] = None,
    save_iterations: bool = False,
    log_level: int = logging.WARNING
) -> np.ndarray:
    logging.getLogger().setLevel(log_level)
    
    num_sim_levels = result.states[0].dims[0][0]
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    tlist = result.times

    ## Set up the Pauli product basis of the space of Hermitian operators
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    # Flattened list of basis operators excluding the identity operator
    basis_list = basis.reshape(-1, *basis.shape[-2:])[1:]
    basis_size = basis_list.shape[0]
    # Basis array multiplied with time - concatenate with coeffs to produce Heff*t
    basis_t = tlist[:, None, None, None] * np.repeat(basis_list[None, ...], tlist.shape[0], axis=0) / (2 ** (num_qubits - 1))

    ## Time evolution unitaries
    time_evolution = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    
    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('num_qubits', data=num_qubits)
            out.create_dataset('num_sim_levels', data=num_sim_levels)
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('time_evolution', data=time_evolution)
            out.create_dataset('tlist', data=result.times)
        
        if save_iterations:
            with h5py.File(f'{save_result_to}_iter.h5', 'w') as out:
                out.create_dataset('max_com', data=max_com)
                out.create_dataset('min_coeff_ratio', data=min_coeff_ratio)
                out.create_dataset('num_update', data=num_update)
                out.create_dataset('omegas', shape=(max_iterations, tlist.shape[0], time_evolution.shape[-1]), dtype='f')
                out.create_dataset('ilogu_coeffs', shape=(max_iterations, tlist.shape[0], basis_size), dtype='f')
                out.create_dataset('tmax', shape=(max_iterations,), dtype='i')
                out.create_dataset('heff_coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('fit_success', shape=(max_iterations, basis_size), dtype='i')
                out.create_dataset('com', shape=(max_iterations, basis_size), dtype='f')
        else:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('omegas', shape=(tlist.shape[0], time_evolution.shape[-1]), dtype='f')
                out.create_dataset('ilogu_coeffs', shape=(tlist.shape[0], basis_size), dtype='f')
            
    ## Output array

    heff_coeffs = np.zeros(basis_size)

    ## Iteratively fit and subtract            
            
    def fit_fn(t, a):
        return t * a

    unitaries = time_evolution

    for iloop in range(max_iterations):
        logging.info('Fit-and-subtract iteration %d', iloop)
        
        ## Compute ilog(U(t))

        ilogus, omegas = matrix_ufunc(lambda u: -np.angle(u), unitaries, with_diagonals=True)
    
        ## Extract the Pauli components

        # Divide the trace by two to account for the normalization of the generalized Paulis
        ilogu_coeffs = np.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.
    
        ## Find the first t where an eigenvalue does a 2pi jump
        tmax = tlist.shape[0]
        for omega_ext in [np.amin(omegas, axis=1), -np.amax(omegas, axis=1)]:
            margin = 0.1
            hits_minus_pi = np.asarray(omega_ext < -np.pi + margin).nonzero()[0]
            if len(hits_minus_pi) != 0:
                tmax = min(tmax, hits_minus_pi[0])
                
        ## Do a linear fit to each component

        xdata = tlist[:tmax] / tlist[-1]
        
        @partial(np.vectorize, otypes=[float, bool, float], signature='(t)->(),(),()')
        def get_slope(ydata):
            try:
                popt, pcov = scipy.optimize.curve_fit(fit_fn, xdata, ydata, p0=(1.,))
            except:
                slope = 0.
                success = False
                com = -1.
            else:
                slope = popt[0]
                success = True
                curve_opt = fit_fn(xdata, *popt)
                cumul = np.cumsum(ydata - curve_opt)
                com = np.amax(np.abs(cumul)) / np.amax(np.abs(ydata))
            
            return slope, success, com
        
        slope, success, com = get_slope(ilogu_coeffs[:tmax].T)

        ## Coeffs from slopes (slopes obtained from a normalized time series)
    
        coeffs = slope / tlist[-1]

        ## Update Heff with the best-fit coefficients
        
        is_update_candidate = success & (com < max_com)
            
        # If we have successfully made ilogu small that the fit range is the entire tlist,
        # also cut on the size of the coefficient (to avoid updating Heff with ~0 indefinitely)
        if iloop != 0 and tmax == tlist.shape[0]:
            max_coeff = np.amax(np.abs(heff_coeffs))
            is_update_candidate &= (np.abs(coeffs) / max_coeff > min_coeff_ratio)
            
        num_candidates = is_update_candidate.nonzero()[0].shape[0]
        if num_update > 0:
            num_candidates = min(num_update, num_candidates)

        # Take the first n largest-coefficient candidates
        update_indices = np.argsort(np.where(is_update_candidate, -np.abs(coeffs), 0.))
        update_indices = update_indices[:num_candidates]
        
        ## Update the output array taking the coefficient of the best-fit component
         
        heff_coeffs[update_indices] += coeffs[update_indices]
        
        ## Save the computation results
        
        if save_result_to:
            if save_iterations:
                with h5py.File(f'{save_result_to}_iter.h5', 'a') as out:
                    out['omegas'][iloop] = omegas
                    out['ilogu_coeffs'][iloop] = ilogu_coeffs
                    out['tmax'][iloop] = tmax
                    out['coeffs'][iloop] = coeffs
                    out['fit_success'][iloop] = success
                    out['com'][iloop] = com
                    out['heff_coeffs'][iloop] = heff_coeffs
                    
            elif iloop == 0:
                with h5py.File(f'{save_result_to}.h5', 'a') as out:
                    out['omegas'][:] = omegas
                    out['ilogu_coeffs'][:] = ilogu_coeffs

        ## Break if there were no updates in this iteration
        
        if num_candidates == 0:
            break
        
        ## Unitarily subtract the current Heff from the time evolution
        
        if iloop != max_iterations - 1:
            heff_t = np.tensordot(basis_t, heff_coeffs, (1, 0))
            exp_iheffs = matrix_ufunc(np.exp, 1.j * heff_t)
            unitaries = np.matmul(time_evolution, exp_iheffs)
            
    if save_result_to and save_iterations:
        with h5py.File(f'{save_result_to}_iter.h5', 'r') as source:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                for key in source.keys():
                    data = source[key]
                    if len(data.shape) > 0:
                        out.create_dataset(key, data=data[:iloop + 1])
                    else:
                        out.create_dataset(key, data=data)

        os.unlink(f'{save_result_to}_iter.h5')

    heff_coeffs = np.concatenate(([0.], heff_coeffs)).reshape(basis.shape[:-2])
    
    if comp_dim < num_sim_levels:
        ## Truncate the hamiltonian to the computational subspace
        l0_projection = get_l0_projection(comp_dim, num_sim_levels)

        for iq in range(num_qubits):
            indices = [slice(None)] * num_qubits
            indices[iq] = 0
            indices = tuple(indices)
            
            heff_coeffs[indices] = np.tensordot(heff_coeffs, l0_projection, (iq, 0))
            
        num_comp_paulis = get_num_paulis(comp_dim)
        heff_coeffs = heff_coeffs[(slice(num_comp_paulis),) * num_qubits]

    return heff_coeffs


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

    ilogu = eigcols @ ilog_diagonal @ eigrows
    
    ## Extract the (generalized) Pauli components
    
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)
    prod_basis = make_prod_basis(paulis, num_qubits)
    
    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    ilogu_coeffs = np.einsum(f'xy,{qubit_indices}yx->{qubit_indices}', ilogu, prod_basis).real
    # Divide the trace by two to account for the normalization of the generalized Paulis
    ilogu_coeffs /= 2.
    
    return ilogu_coeffs


