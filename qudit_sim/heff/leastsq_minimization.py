from typing import Optional, Tuple
import os
import logging
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy.optimize
import h5py

try:
    import jax
except ImportError:
    HAS_JAX = False
else:
    HAS_JAX = True
    import jax.numpy as jnp
    import jax.scipy.optimize

import rqutils.paulis as paulis

from .common import get_ilogus_and_valid_it, compose_ueff

def leastsq_minimization(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    jax_device_id: Optional[int] = None,
    max_com: float = 20.,
    min_compo_ratio: float = 0.005,
    num_update_per_iteration: int = 0,
    max_iterations: int = 100,
    save_iterations: bool = False
) -> np.ndarray:
    """Determine the effective Hamiltonian from the simulation result using the iterative fit method.

    Args:
        time_evolution: Time evolution operator (shape (T, d1*d2*..., d1*d2*...))
        tlist: Time points (shape (T,)).
        dim: Subsystem dimensions.
        save_result_to: File name (without an extension) to save the intermediate results to.
        log_level: Log level.
        jax_device_id: If not None, use JAX on the specified device ID.
        max_com: Convergence condition.
        min_compo_ratio: Convergence condition.
        num_update_per_iteration: Number of components to update per iteration. If <= 0, all candidate components are updated.
        max_iterations: Maximum number of fit & subtract iterations.
        save_iterations: If True, save the results of individual fit iterations to `save_result_to`.

    Returns:
        Pauli components of the effective Hamiltonian.
    """
    original_log_level = logging.getLogger().level
    logging.getLogger().setLevel(log_level)

    matrix_dim = np.prod(dim)
    assert time_evolution.shape == (tlist.shape[0], matrix_dim, matrix_dim), 'Inconsistent input shape'

    if jax_device_id is not None:
        assert HAS_JAX, 'JAX is not installed'

    tsize = tlist.shape[0]
    tend = tlist[-1]

    ## Set up the Pauli product basis of the space of Hermitian operators
    basis = paulis.paulis(dim)
    # Flattened list of basis operators excluding the identity operator
    basis_list = basis.reshape(-1, *basis.shape[-2:])[1:]
    basis_size = basis_list.shape[0]

    if num_update_per_iteration <= 0:
        num_update_per_iteration = basis_size

    if save_result_to and save_iterations:
        with h5py.File(f'{save_result_to}_iter.h5', 'w') as out:
            out.create_dataset('max_com', data=max_com)
            out.create_dataset('min_compo_ratio', data=min_compo_ratio)
            out.create_dataset('num_update_per_iteration', data=num_update_per_iteration)
            out.create_dataset('ilogvs', shape=(max_iterations, tsize, time_evolution.shape[-1]), dtype='f')
            out.create_dataset('ilogu_compos', shape=(max_iterations, tsize, basis_size), dtype='f')
            out.create_dataset('last_valid_it', shape=(max_iterations,), dtype='i')
            out.create_dataset('iter_heff', shape=(max_iterations, basis_size), dtype='f')
            out.create_dataset('compos', shape=(max_iterations, basis_size), dtype='f')
            out.create_dataset('fit_success', shape=(max_iterations, basis_size), dtype='i')
            out.create_dataset('com', shape=(max_iterations, basis_size), dtype='f')

    unitaries = time_evolution
    # Normalized tlist
    tlist_norm = tlist / tend

    if jax_device_id is not None:
        npmod = jnp
        sciopt = jax.scipy.optimize
    else:
        npmod = np
        sciopt = scipy.optimize

    def fun(params, ydata, mask):
        residual_masked = (tlist_norm * params[0] - ydata) * mask
        return npmod.sum(npmod.square(residual_masked))

    @partial(npmod.vectorize, signature='(t),(),(t)->(),()')
    def fit_compos(ydata, x0, mask):
        # Using minimize instead of curve_fit because the latter is not available in jax
        res = sciopt.minimize(fun, npmod.array([x0]), args=(ydata, mask), method='BFGS')
        return res.x[0] / tend, res.success

    def update_heff(ilogus, last_valid_it, heff_compos):
        # Divide the trace by two to account for the normalization of the generalized Paulis
        ilogu_compos = npmod.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.

        ## Do a linear fit to each component
        mask = (np.arange(tsize) < last_valid_it).astype(float)
        compos, success = fit_compos(ilogu_compos.T, ilogu_compos.T[:, last_valid_it - 1], mask[None, :])

        cumul = npmod.cumsum((tlist[:, None] * compos[None, :] - ilogu_compos) * mask[:, None], axis=0)
        maxabs = npmod.amax(npmod.abs(ilogu_compos) * mask[:, None], axis=0)
        com = npmod.amax(npmod.abs(cumul), axis=0) / maxabs

        ## Update Heff with the best-fit components
        is_update_candidate = success & (com < max_com)
        # If we have successfully made ilogu small that the fit range is the entire tlist,
        # also cut on the size of the component (to avoid updating Heff with ~0 indefinitely)
        compo_nonnegligible = (npmod.abs(compos) > min_compo_ratio * npmod.amax(npmod.abs(heff_compos)))
        is_update_candidate &= compo_nonnegligible | (last_valid_it != tsize)

        num_candidates = npmod.min(npmod.array([num_update_per_iteration, npmod.count_nonzero(is_update_candidate)]))

        ## Take the first n largest-component candidates
        update_indices = npmod.argsort(npmod.where(is_update_candidate, -npmod.abs(compos), 0.))

        ## Update the output array taking the component of the best-fit component
        mask = (npmod.arange(compos.shape[0]) < num_candidates).astype(float)
        compos_update = compos[update_indices] * mask
        compos_update = compos_update[npmod.argsort(update_indices)]

        if save_result_to:
            return heff_compos + compos_update, num_candidates > 0, ilogu_compos, compos, success, com
        else:
            return heff_compos + compos_update, num_candidates > 0

    def update_unitaries(heff_compos, time_evolution):
        ueff_dagger = compose_ueff(heff_compos, basis_list, tlist, phase_factor=1., npmod=npmod)
        return npmod.matmul(time_evolution, ueff_dagger)

    ## Output array
    heff_compos = np.zeros(basis_size)

    if jax_device_id is not None:
        jax_device = jax.devices()[jax_device_id]

        time_evolution = jax.device_put(time_evolution, device=jax_device)
        basis_list = jnp.device_put(basis_list, device=jax_device)

        tlist_norm = jnp.device_put(tlist_norm, device=jax_device)

        heff_compos = jax.device_put(heff_compos, device=jax_device)

        update_heff = jax.jit(update_heff, device=jax_device)
        update_unitaries = jax.jit(update_unitaries, device=jax_device)

    ## Iterative fit & subtract loop
    for iloop in range(max_iterations):
        logging.info('Fit-and-subtract iteration %d', iloop)

        ilogus, ilogvs, last_valid_it = get_ilogus_and_valid_it(unitaries)

        update_result = update_heff(ilogus, last_valid_it, heff_compos)

        heff_compos, updated = update_result[:2]

        ## Save the computation results
        if save_result_to and save_iterations:
            ilogu_compos, compos, success, com = update_result[2:]

            with h5py.File(f'{save_result_to}_iter.h5', 'a') as out:
                out['ilogvs'][iloop] = ilogvs
                out['last_valid_it'][iloop] = last_valid_it
                out['ilogu_compos'][iloop] = ilogu_compos
                out['compos'][iloop] = compos
                out['fit_success'][iloop] = success
                out['com'][iloop] = com
                out['iter_heff'][iloop] = heff_compos

        ## Break if there were no updates in this iteration
        if not updated:
            break

        ## Unitarily subtract the current Heff from the time evolution
        if iloop != max_iterations - 1:
            unitaries = update_unitaries(heff_compos, time_evolution)

    if save_result_to and save_iterations:
        with h5py.File(f'{save_result_to}_iter.h5', 'r') as source:
            with h5py.File(f'{save_result_to}_ext.h5', 'w') as out:
                for key in source.keys():
                    data = source[key]
                    if len(data.shape) > 0:
                        out.create_dataset(key, data=data[:iloop + 1])
                    else:
                        out.create_dataset(key, data=data)

        os.unlink(f'{save_result_to}_iter.h5')

    heff_compos = np.concatenate(([0.], heff_compos)).reshape(basis.shape[:-2])

    logging.getLogger().setLevel(original_log_level)

    return heff_compos
