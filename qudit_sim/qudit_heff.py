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

from .heff import iterative_fit

def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    num_cycles: int = 400,
    comp_dim: int = 2,
    method: Callable = iterative_fit,
    extraction_params: Dict,
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

    heff_coeffs = method(
        result,
        comp_dim,
        save_result_to,
        log_level,
        **extraction_params)
    
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


