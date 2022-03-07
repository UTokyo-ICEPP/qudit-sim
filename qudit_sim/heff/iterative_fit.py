from typing import Optional
import os
import logging
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import scipy.optimize
import h5py
import qutip as qtp

try:
    import jax
except ImportError:
    HAS_JAX = False
else:
    HAS_JAX = True
    import jax.numpy as jnp
    import jax.scipy.optimize

from ..paulis import make_generalized_paulis, make_prod_basis
from ..utils import matrix_ufunc, make_ueff
from .common import get_ilogus_and_valid_it, truncate_heff

def iterative_fit(
    result: qtp.solver.Result,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    max_com: float = 20.,
    min_coeff_ratio: float = 0.005,
    num_update_per_iteration: int = 0,
    max_iterations: int = 100,
    save_iterations: bool = False,
    use_jax: bool = HAS_JAX
) -> np.ndarray:
    original_log_level = logging.getLogger().level
    logging.getLogger().setLevel(log_level)
    
    if use_jax:
        assert HAS_JAX, 'JAX is not installed'
    
    num_qubits = len(result.states[0].dims[0])
    num_sim_levels = result.states[0].dims[0][0]
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    tlist = result.times
    tsize = tlist.shape[0]
    tend = tlist[-1]

    ## Set up the Pauli product basis of the space of Hermitian operators
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    del paulis
    # Flattened list of basis operators excluding the identity operator
    basis_list = basis.reshape(-1, *basis.shape[-2:])[1:]
    basis_size = basis_list.shape[0]

    if num_update_per_iteration <= 0:
        num_update_per_iteration = basis_size

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
                out.create_dataset('num_update_per_iteration', data=num_update_per_iteration)
                out.create_dataset('ilogvs', shape=(max_iterations, tsize, time_evolution.shape[-1]), dtype='f')
                out.create_dataset('ilogu_coeffs', shape=(max_iterations, tsize, basis_size), dtype='f')
                out.create_dataset('last_valid_it', shape=(max_iterations,), dtype='i')
                out.create_dataset('heff_coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('fit_success', shape=(max_iterations, basis_size), dtype='i')
                out.create_dataset('com', shape=(max_iterations, basis_size), dtype='f')
        else:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('ilogvs', shape=(tsize, time_evolution.shape[-1]), dtype='f')
                out.create_dataset('ilogu_coeffs', shape=(tsize, basis_size), dtype='f')

    unitaries = time_evolution
    tlist_norm = tlist / tend
    
    if use_jax:
        npmod = jnp
        sciopt = jax.scipy.optimize
    else:
        npmod = np
        sciopt = scipy.optimize
                
    def fun(params, ydata, mask):
        residual_masked = (tlist_norm * params[0] - ydata) * mask
        return npmod.sum(npmod.square(residual_masked))

    @partial(npmod.vectorize, signature='(t),(),(t)->(),()')
    def get_coeffs(ydata, x0, mask):
        # Using minimize instead of curve_fit because the latter is not available in jax
        res = sciopt.minimize(fun, npmod.array([x0]), args=(ydata, mask), method='BFGS')
        return res.x[0] / tend, res.success
    
    def update_heff(ilogus, last_valid_it, heff_coeffs):
        # Divide the trace by two to account for the normalization of the generalized Paulis
        ilogu_coeffs = npmod.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.

        ## Do a linear fit to each component
        mask = (npmod.arange(tsize) < last_valid_it).astype(float)
        coeffs, success = get_coeffs(ilogu_coeffs.T, ilogu_coeffs.T[:, last_valid_it - 1], mask[None, :])
        
        cumul = npmod.cumsum((tlist[:, None] * coeffs[None, :] - ilogu_coeffs) * mask[:, None], axis=0)
        maxabs = npmod.amax(npmod.abs(ilogu_coeffs) * mask[:, None], axis=0)
        com = npmod.amax(npmod.abs(cumul), axis=0) / maxabs

        ## Update Heff with the best-fit coefficients
        is_update_candidate = success & (com < max_com)
        # If we have successfully made ilogu small that the fit range is the entire tlist,
        # also cut on the size of the coefficient (to avoid updating Heff with ~0 indefinitely)
        coeff_nonnegligible = (npmod.abs(coeffs) > min_coeff_ratio * npmod.amax(npmod.abs(heff_coeffs)))
        is_update_candidate &= coeff_nonnegligible | (last_valid_it != tsize)
        
        num_candidates = npmod.min(npmod.array([num_update_per_iteration, npmod.count_nonzero(is_update_candidate)]))

        ## Take the first n largest-coefficient candidates
        update_indices = npmod.argsort(npmod.where(is_update_candidate, -npmod.abs(coeffs), 0.))
        
        ## Update the output array taking the coefficient of the best-fit component
        mask = (npmod.arange(coeffs.shape[0]) < num_candidates).astype(float)
        coeffs_update = coeffs[update_indices] * mask
        coeffs_update = coeffs_update[npmod.argsort(update_indices)]
        
        if save_result_to:
            return heff_coeffs + coeffs_update, num_candidates > 0, ilogu_coeffs, coeffs, success, com
        else:
            return heff_coeffs + coeffs_update, num_candidates > 0

    def update_unitaries(heff_coeffs):
        ueff_dagger = make_ueff(heff_coeffs, basis_list, num_qubits, tlist, phase_factor=1., npmod=jnp)
        return npmod.matmul(time_evolution, ueff_dagger)
    
    if use_jax:
        time_evolution = jnp.array(time_evolution)
        basis_list = jnp.array(basis_list)
        # Normalized tlist
        tlist_norm = jnp.array(tlist_norm)
        
        update_heff = jax.jit(update_heff)
        update_unitaries = jax.jit(update_unitaries)
    
    ## Output array
    heff_coeffs = npmod.zeros(basis_size)

    ## Iterative fit & subtract loop
    for iloop in range(max_iterations):
        logging.info('Fit-and-subtract iteration %d', iloop)
        
        ilogus, ilogvs, last_valid_it = get_ilogus_and_valid_it(unitaries)
    
        update_result = update_heff(ilogus, last_valid_it, heff_coeffs)
        
        ## Save the computation results
        if save_result_to:
            heff_coeffs, updated, ilogu_coeffs, coeffs, success, com = update_result

            if save_iterations:
                with h5py.File(f'{save_result_to}_iter.h5', 'a') as out:
                    out['ilogvs'][iloop] = ilogvs
                    out['last_valid_it'][iloop] = last_valid_it
                    out['ilogu_coeffs'][iloop] = ilogu_coeffs
                    out['coeffs'][iloop] = coeffs
                    out['fit_success'][iloop] = success
                    out['com'][iloop] = com
                    out['heff_coeffs'][iloop] = heff_coeffs
                    
            elif iloop == 0:
                with h5py.File(f'{save_result_to}.h5', 'a') as out:
                    out['ilogvs'][:] = ilogvs
                    out['ilogu_coeffs'][:] = ilogu_coeffs
                    
        else:
            heff_coeffs, updated = update_result
            
        ## Break if there were no updates in this iteration
        if not updated:
            break
        
        ## Unitarily subtract the current Heff from the time evolution
        if iloop != max_iterations - 1:
            unitaries = update_unitaries(heff_coeffs)
            
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

    heff_coeffs = truncate_heff(heff_coeffs, num_sim_levels, comp_dim, num_qubits)
        
    logging.getLogger().setLevel(original_log_level)

    return heff_coeffs
