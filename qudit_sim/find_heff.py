from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable
import os
import sys
import time
import string
import collections
import logging
from multiprocessing import Process, cpu_count
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy
import h5py
import qutip as qtp

from .paulis import (get_num_paulis, make_generalized_paulis, make_prod_basis,
                    unravel_basis_index, get_l0_projection)
from .pulse_sim import run_pulse_sim, DriveDef
from .parallel import parallel_map

logger = logging.getLogger(__name__)

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
    sim_num_cpus: int = 0,
    ext_num_cpus: int = 0,
    jax_devices: Optional[List] = None,
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
            If <=0, set to `multiprocessing.cpu_count()`.
        ext_num_cpus: Number of threads to use for Pauli coefficient extraction when a list is passed as `drive_def`.
            If <=0, set to `multiprocessing.cpu_count()`. For extraction methods that use GPUs, the combination of
            `jax_devices` and this parameter controls how many processes will be run on each device.
        jax_devices: List of GPU ids (integers starting at 0) to use.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    ## Extraction function and parameters
    
    if method == 'iterative_fit':
        from .heff import iterative_fit
        extraction_fn = iterative_fit
    elif method == 'maximize_fidelity':
        from .heff import maximize_fidelity
        extraction_fn = maximize_fidelity
        
    if extraction_params is None:
        extraction_params = dict()
    
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)
    
    logger.info('Running a square pulse simulation for %d cycles', num_cycles)

    tlist_tuple = (10, num_cycles)
    
    if isinstance(drive_def, list):
        results = parallel_map(
            run_pulse_sim,
            mapped_args=drive_def,
            arg_position=2,
            common_args=(qubits, params),
            common_kwargs={'psi0': psi0, 'tlist': tlist_tuple, 'log_level': log_level},
            num_cpus=sim_num_cpus,
            log_level=log_level
        )
        
        logger.info('Passing the simulation results to %s', extraction_fn.__name__)
        
        if jax_devices is None:
            try:
                import jax
                num_jax_devices = jax.local_device_count()
            except ImportError:
                num_jax_devices = 1

            jax_devices = list(range(num_jax_devices))

        jax_device_iter = iter(jax_devices)
            
        mapped_args = []
        mapped_kwargs = []
        for time_evolution, tlist in results:
            try:
                jax_device_id = next(jax_device_iter)
            except StopIteration:
                jax_device_iter = iter(jax_devices)
                jax_device_id = next(jax_device_iter)
            
            mapped_args.append((time_evolution, tlist))
            mapped_kwargs.append({'jax_device_id': jax_device_id})
            
        common_kwargs = {
            'num_qubits': num_qubits,
            'num_sim_levels': num_sim_levels,
            'comp_dim': comp_dim,
            'log_level': log_level
        }
        common_kwargs.update(extraction_params)
        
        heff_coeffs = parallel_map(
            extraction_fn,
            mapped_args=mapped_args,
            mapped_kwargs=mapped_kwargs,
            common_kwargs=common_kwargs,
            num_cpus=ext_num_cpus,
            log_level=log_level,
            thread_based=True
        )
        
        heff_coeffs = np.stack(heff_coeffs)
    
    else:
        time_evolution, tlist = run_pulse_sim(
            qubits,
            params,
            drive_def,
            psi0=psi0,
            tlist=tlist_tuple,
            save_result_to=save_result_to,
            log_level=log_level)
        
        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'w') as out:
                out.create_dataset('num_qubits', data=num_qubits)
                out.create_dataset('num_sim_levels', data=num_sim_levels)
                out.create_dataset('comp_dim', data=comp_dim)
                out.create_dataset('time_evolution', data=time_evolution)
                out.create_dataset('tlist', data=tlist)

        heff_coeffs = extraction_fn(
            time_evolution,
            tlist,
            num_qubits=num_qubits,
            num_sim_levels=num_sim_levels,
            comp_dim=comp_dim,
            save_result_to=save_result_to,
            log_level=log_level,
            **extraction_params)

    logger.setLevel(original_log_level)
    
    return heff_coeffs
