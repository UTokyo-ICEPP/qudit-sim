from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable
import os
import sys
import time
import string
import collections
import logging
import multiprocessing
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy
import h5py
import qutip as qtp
try:
    import jax
    num_jax_devices = jax.local_device_count()
except ImportError:
    num_jax_devices = 1

from .paulis import (get_num_paulis, make_generalized_paulis, make_prod_basis,
                    unravel_basis_index, get_l0_projection)
from .pulse_sim import run_pulse_sim, DriveDef

from .heff import iterative_fit, maximize_fidelity

def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Union[List[DriveDef], DriveDef],
    num_sim_levels: int = 2,
    num_cycles: int = 400,
    comp_dim: int = 2,
    method: str = 'maximize_fidelity',
    extraction_params: Optional[Dict] = None,
    save_result_to: Optional[str] = None,
    num_cpus: int = None,
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
        params: Hamiltonian parameters. See the docstring of `hamiltonian.RWAHamiltonianGenerator` for details.
        drive_def: Drive definition or a list thereof. See the docstring of `hamiltonian.RWAHamiltonianGenerator`
            for details. Argument `'amplitude'` for each channel must be a float or a constant expression string.
        num_sim_levels: Number of oscillator levels in the simulation.
        num_cycles: Duration of the square pulse, in units of cycles in the highest frequency appearing in the Hamiltonian.
        comp_dim: Dimensionality of the computational space.
        method: Name of the function to use for Pauli coefficient extraction.
        extraction_params: Optional keyword arguments to pass to the extraction function.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
            Simulation result will not be saved when a list is passed as `drive_def`.
        sim_num_cpus: Number of threads to use for the pulse simulation when a list is passed as `drive_def`.
            If <=0, set to `qutip.settings.num_cpus`.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
    """
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    ## Extraction function and parameters
    
    if method == 'iterative_fit':
        extraction_fn = iterative_fit
    elif method == 'maximize_fidelity':
        extraction_fn = maximize_fidelity
        
    if extraction_params is None:
        extraction_params = dict()
    
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)
    
    logging.info('Running a square pulse simulation for %d cycles', num_cycles)

    tlist_tuple = (10, num_cycles)
    
    if isinstance(drive_def, list):
        with multiprocessing.Pool(processes=num_cpus) as pool:
            async_results = list()
            for ddef in drive_def:
                args = (qubits, params, ddef, psi0, tlist_tuple)
                res = pool.apply_async(run_pulse_sim, args)
                async_res.append(res)

            results = list(res.get() for res in async_results)

            async_results = list()
            jax_device_id = 0
            for result in results:
                time_evolution = np.stack(list(state.full() for state in result.states))
                tlist = result.times

                args = (time_evolution, tlist)
                kwargs = {
                    'num_qubits': num_qubits,
                    'num_sim_levels': num_sim_levels,
                    'comp_dim': comp_dim,
                    'log_level': log_level,
                    'jax_device_id': jax_device_id
                }
                kwargs.update(extraction_params)

                jax_device_id += 1
                jax_device_id %= num_jax_devices

                res = pool.apply_async(extraction_fn, args, kwargs)
                async_res.append(res)

            heff_coeffs = list(res.get() for res in async_results)
        
        return np.stack(heff_coeffs)
    
    else:
        result = run_pulse_sim(drive_def, qubits, params, psi0, tlist_tuple, save_result_to=save_result_to)
        
        time_evolution = np.stack(list(state.full() for state in result.states))
        tlist = result.times
        
        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'w') as out:
                out.create_dataset('num_qubits', data=num_qubits)
                out.create_dataset('num_sim_levels', data=num_sim_levels)
                out.create_dataset('comp_dim', data=comp_dim)
                out.create_dataset('time_evolution', data=time_evolution)
                out.create_dataset('tlist', data=tlist)

        return extraction_fn(
            time_evolution,
            tlist,
            num_qubits=num_qubits,
            num_sim_levels=num_sim_levels,
            comp_dim=comp_dim,
            save_result_to=save_result_to,
            log_level=log_level,
            **extraction_params)
