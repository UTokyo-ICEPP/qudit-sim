r"""
================================
Effective Hamiltonian extraction
================================

The full time evolution operator :math:`U_{H}(t) = T\left[\exp(-i \int_0^t dt' H(t'))\right]` of a driven qudit
system is time-dependent and highly nontrivial. However, when the drive amplitude is a constant, at a longer time
scale, it should be approximatable with a time evolution by a constant Hamiltonian (= effective Hamiltonian)
:math:`U_{\mathrm{eff}}(t) = \exp(-i H_{\mathrm{eff}} t)`.

Identification of this :math:`H_{\mathrm{eff}}` is essentially a linear fit to the time evolution of Pauli
components of :math:`i \mathrm{log} (U_{H}(t))`. In qudit-sim we have two implementations of this fit:

- `"fidelity"` finds the effective Pauli components that maximize
  :math:`\sum_{i} \big| \mathrm{tr} \left[ U(t_i)\, \exp \left(i H_{\mathrm{eff}} t_i \right)\right]} \big|^2`.
- `"leastsq"` performs a least-squares fit to individual components of :math:`i \mathrm{log} (U_{H}(t))`.

Fidelity method is usually more robust, but the least squares method allows better "fine-tuning". A combined
method is also available.
"""

from typing import Any, Dict, List, Sequence, Optional, Union
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py

import rqutils.paulis as paulis

from .parallel import parallel_map

logger = logging.getLogger(__name__)

def find_heff(
    sim_result: Union['PulseSimResult', List['PulseSimResult']],
    comp_dim: int = 2,
    method: str = 'fidelity',
    method_params: Optional[Dict] = None,
    save_result_to: Optional[str] = None,
    num_cpus: int = 0,
    jax_devices: Optional[List[int]] = None,
    log_level: int = logging.WARNING
) -> np.ndarray:
    r"""Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.
    
    QuTiP `sesolve` applied to the identity matrix will give the time evolution operator :math:`U_H(t)` according
    to the rotating-wave Hamiltonian :math:`H` at each time point. This function then finds the Pauli components
    of :math:`H_{\mathrm{eff}}` whose time evolution :math:`\exp(-i H_{\mathrm{eff}} t)` approximates :math:`U_H(t)`.

    Args:
        hgen: Hamiltonian generator or a list thereof.
        num_cycles: Duration of the square pulse, in units of cycles in the highest frequency appearing in the
            Hamiltonian.
        comp_dim: Dimensionality of the computational space.
        method: Name of the function to use for Pauli component extraction. Currently possible values are
            'fidelity' and 'leastsq'.
        extraction_params: Optional keyword arguments to pass to the extraction function.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
            Simulation result will not be saved when a list is passed as `drive_def`.
        sim_num_cpus: Number of threads to use for the pulse simulation when a list is passed as `drive_def`.
            If <=0, set to `multiprocessing.cpu_count()`.
        ext_num_cpus: Number of threads to use for Pauli component extraction when a list is passed as `drive_def`.
            If <=0, set to `multiprocessing.cpu_count()`. For extraction methods that use GPUs, the combination of
            `jax_devices` and this parameter controls how many processes will be run on each device.
        jax_devices: List of GPU ids (integers starting at 0) to use.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the component of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    ## Extraction function and parameters
    
    if method == 'leastsq':
        from .heff import leastsq_minimization
        extraction_fn = leastsq_minimization
    elif method == 'fidelity':
        from .heff import fidelity_maximization
        extraction_fn = fidelity_maximization
        
    if method_params is None:
        method_params = dict()
        
    def save_result(filename, sim_result, compos, compos_original):
        with h5py.File(f'{filename}.h5', 'w') as out:
            out.create_dataset('num_qudits', data=len(sim_result.dim))
            out.create_dataset('num_sim_levels', data=sim_result.dim[0])
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('time_evolution', data=sim_result.states)
            out.create_dataset('tlist', data=sim_result.times)
            out.create_dataset('heff_compos', data=compos)
            if compos_original is not None:
                out.create_dataset('heff_compos_original', data=compos_original)
                
            with h5py.File(f'{filename}_ext.h5', 'r') as source:
                for key in source.keys():
                    out.create_dataset(key, data=source[key])

            os.unlink(f'{filename}_ext.h5')
        
    if isinstance(sim_result, list):
        num_tasks = len(sim_result)
        
        if jax_devices is None:
            try:
                import jax
                num_jax_devices = jax.local_device_count()
            except ImportError:
                num_jax_devices = 1

            jax_devices = list(range(num_jax_devices))

        jax_device_iter = iter(jax_devices)
            
        args = []
        kwarg_keys = ('jax_device_id',)
        kwarg_values = []
        for result in sim_result:
            try:
                jax_device_id = next(jax_device_iter)
            except StopIteration:
                jax_device_iter = iter(jax_devices)
                jax_device_id = next(jax_device_iter)
            
            args.append((result.states, result.times, result.dim))
            kwarg_values.append((jax_device_id,))
        
        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)
            
            kwarg_keys += ('save_result_to',)
            for itask in range(num_tasks):
                kwarg_values[itask] += (filename,)
                
        common_kwargs = {
            'log_level': log_level
        }
        common_kwargs.update(method_params)
        
        heff_compos_list = parallel_map(
            extraction_fn,
            args=args,
            kwarg_keys=kwarg_keys,
            kwarg_values=kwarg_values,
            common_kwargs=common_kwargs,
            num_cpus=num_cpus,
            log_level=log_level,
            thread_based=True)
        
        heff_compos = np.stack(heff_compos_list)

        if result.dim[0] != comp_dim:
            heff_compos_original = heff_compos
            heff_compos = paulis.truncate(heff_compos, (comp_dim,) * len(result.dim))
        else:
            heff_compos_original = None
        
        if save_result_to:
            for itask, result in enumerate(sim_result):
                filename = os.path.join(save_result_to, f'heff_{itask}')
                save_result(filename, result, heff_compos, heff_compos_original)
    
    else:        
        heff_compos = extraction_fn(
            sim_result.states,
            sim_result.times,
            sim_result.dim,
            save_result_to=save_result_to,
            log_level=log_level,
            **method_params)
        
        if sim_result.dim[0] != comp_dim:
            heff_compos_original = heff_compos
            heff_compos = paulis.truncate(heff_compos, (comp_dim,) * len(sim_result.dim))
        else:
            heff_compos_original = None
        
        if save_result_to:
            save_result(save_result_to, sim_result, heff_compos, heff_compos_original)
                
    logger.setLevel(original_log_level)
    
    return heff_compos
