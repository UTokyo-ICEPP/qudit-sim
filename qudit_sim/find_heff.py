"""Effective Hamiltonian extraction frontend."""

from typing import Any, Dict, List, Sequence, Optional, Union
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py

import rqutils.paulis as paulis

from .util import PulseSimResult
from .parallel import parallel_map

logger = logging.getLogger(__name__)

def find_heff(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: int = 2,
    method: str = 'fidelity',
    method_params: Optional[Dict] = None,
    save_result_to: Optional[str] = None,
    num_cpus: int = 0,
    jax_devices: Optional[List[int]] = None,
    log_level: int = logging.WARNING
) -> Union[np.ndarray, List[np.ndarray]]:
    r"""Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.

    The full time evolution operator :math:`U_{H}(t) = T\left[\exp(-i \int_0^t dt' H(t'))\right]` of a driven qudit
    system is time-dependent and highly nontrivial. However, when the drive amplitude is a constant, at a longer time
    scale, it should be approximatable with a time evolution by a constant Hamiltonian (= effective Hamiltonian)
    :math:`U_{\mathrm{eff}}(t) = \exp(-i H_{\mathrm{eff}} t)`.

    Identification of this :math:`H_{\mathrm{eff}}` is essentially a linear fit to the time evolution of Pauli
    components of :math:`i \mathrm{log} (U_{H}(t))`. In qudit-sim we have two implementations of this fit:

    - `"fidelity"` finds the effective Pauli components that maximize
      :math:`\sum_{i} \big| \mathrm{tr} \left[ U(t_i)\, \exp \left(i H_{\mathrm{eff}} t_i \right)\right] \big|^2`.
    - `"leastsq"` performs a least-squares fit to individual components of :math:`i \mathrm{log} (U_{H}(t))`.

    The fidelity method is usually more robust, but the least squares method allows better "fine-tuning". A combined
    method is also available.

    Args:
        sim_result: Result from pulse_sim.
        comp_dim: Dimensionality of the computational space.
        method: Name of the function to use for Pauli component extraction. Currently possible values are
            'fidelity' and 'leastsq'.
        method_params: Optional keyword arguments to pass to the extraction function.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
            Simulation result will not be saved when a list is passed as `drive_def`.
        num_cpus: Number of threads to use for Pauli component extraction when a list is passed as `sim_result`.
            If <=0, set to `multiprocessing.cpu_count()`. For extraction methods that use GPUs, the combination of
            `jax_devices` and this parameter controls how many processes will be run on each device.
        jax_devices: List of GPU ids (integers starting at 0) to use.

    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the component of
        :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian. If a list is passed
        as `sim_result`, returns a list of arrays.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    ## Extraction function parameters
    if method_params is None:
        method_params = dict()

    if isinstance(sim_result, list):
        common_kwargs = {'method': method, 'comp_dim': comp_dim, 'method_params': method_params,
                         'log_level': log_level}

        num_tasks = len(sim_result)

        if jax_devices is None:
            try:
                import jax
                num_jax_devices = jax.local_device_count()
            except ImportError:
                num_jax_devices = 1

            jax_devices = list(range(num_jax_devices))

        args = list()
        kwarg_keys = ('jax_device_id', 'logger_name',)
        kwarg_values = list()

        for itask, result in enumerate(sim_result):
            args.append((result.states, result.times, result.dim))
            kwarg_values.append((itask % len(jax_devices), f'{__name__}.{itask}'))

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            kwarg_keys += ('save_result_to',)
            for itask in range(num_tasks):
                kwarg_values[itask] += (os.path.join(save_result_to, f'heff_{itask}'),)

        heff_compos = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys,
                                   kwarg_values=kwarg_values, common_kwargs=common_kwargs,
                                   num_cpus=num_cpus, log_level=log_level, thread_based=True)

    else:
        if jax_devices is None:
            jax_device_id = 0
        else:
            jax_device_id = jax_devices[0]

        heff_compos = _run_single(sim_result.states, sim_result.times, sim_result.dim,
                                  method=method, comp_dim=comp_dim, method_params=method_params,
                                  jax_device_id=jax_device_id, save_result_to=save_result_to)

    logger.setLevel(original_log_level)

    return heff_compos


def _run_single(
    states: np.ndarray,
    tlist: np.ndarray,
    dim: tuple,
    method: str,
    comp_dim: int,
    method_params: dict,
    jax_device_id: int = 0,
    save_result_to: Optional[str] = None,
    logger_name: str = __name__,
    log_level: int = logging.WARNING
):
    logger = logging.getLogger(logger_name)

    if method == 'leastsq':
        from .heff import leastsq_minimization
        extraction_fn = leastsq_minimization
    elif method == 'fidelity':
        from .heff import fidelity_maximization
        extraction_fn = fidelity_maximization

    heff_compos = extraction_fn(states, tlist, dim, save_result_to=save_result_to,
                                log_level=log_level, jax_device_id=jax_device_id,
                                **method_params)

    if dim[0] != comp_dim:
        heff_compos_original = heff_compos
        heff_compos = paulis.truncate(heff_compos, (comp_dim,) * len(dim))
    else:
        heff_compos_original = None

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('num_qudits', data=len(dim))
            out.create_dataset('num_sim_levels', data=dim[0])
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('time_evolution', data=states)
            out.create_dataset('tlist', data=tlist)
            out.create_dataset('heff_compos', data=heff_compos)
            if heff_compos_original is not None:
                out.create_dataset('heff_compos_original', data=heff_compos_original)

            with h5py.File(f'{save_result_to}_ext.h5', 'r') as source:
                for key in source.keys():
                    out.create_dataset(key, data=source[key])

            os.unlink(f'{save_result_to}_ext.h5')

    return heff_compos
