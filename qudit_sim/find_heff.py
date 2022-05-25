"""
=============================================================
Effective Hamiltonian extraction (:mod:`qudit_sim.find_heff`)
=============================================================

.. currentmodule:: qudit_sim.find_heff

See :doc:`/heff` for theoretical background.
"""

from typing import Any, Dict, List, Sequence, Optional, Union
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py

import rqutils.paulis as paulis

from .util import PulseSimResult
from .parallel import parallel_map
from .heff.common import heff_fidelity

logger = logging.getLogger(__name__)

def find_heff(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: int = 2,
    method: str = 'fidelity',
    method_params: Optional[Dict] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union[np.ndarray, List[np.ndarray]]:
    r"""Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.

    Args:
        sim_result: Result from pulse_sim.
        comp_dim: Dimensionality of the computational space.
        method: Name of the function to use for Pauli component extraction. Currently possible values are
            'fidelity' and 'leastsq'.
        method_params: Optional keyword arguments to pass to the extraction function.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
            Simulation result will not be saved when a list is passed as `drive_def`.

    Returns:
        An array of Pauli components or a list thereof (if a list is passed to `sim_result`).
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

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            save_result_path = lambda itask: os.path.join(save_result_to, f'heff_{itask}')
        else:
            save_result_path = lambda itask: None

        args = list()
        kwarg_keys = ('logger_name', 'save_result_to')
        kwarg_values = list()

        for itask, result in enumerate(sim_result):
            args.append((result.states, result.times, result.dim))
            kwarg_values.append((
                f'{__name__}.{itask}',
                save_result_path(itask)))

        heff_compos = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys,
                                   kwarg_values=kwarg_values, common_kwargs=common_kwargs,
                                   log_level=log_level, thread_based=True)

    else:
        heff_compos = _run_single(sim_result.states, sim_result.times, sim_result.dim,
                                  method=method, comp_dim=comp_dim, method_params=method_params,
                                  save_result_to=save_result_to, log_level=log_level)

    logger.setLevel(original_log_level)

    return heff_compos


def _run_single(
    states: np.ndarray,
    tlist: np.ndarray,
    dim: tuple,
    method: str,
    comp_dim: int,
    method_params: dict,
    save_result_to: Union[str, None],
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
                                log_level=log_level, **method_params)

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
