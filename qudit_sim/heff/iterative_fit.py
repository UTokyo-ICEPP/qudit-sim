from typing import Optional
import os
import logging
from functools import partial
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import qutip as qtp

try:
    import jax
except ImportError:
    JAX_AVAILABLE = False
    import scipy.optimize as sciopt
else:
    JAX_AVAILABLE = True
    import jax.numpy as jnp
    import jax.scipy.optimize as jsciopt

from ..paulis import (get_num_paulis, make_generalized_paulis, make_prod_basis,
                      unravel_basis_index, get_l0_projection)

from ..utils import matrix_ufunc

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
    use_jax: bool = True
) -> np.ndarray:
    logging.getLogger().setLevel(log_level)
    
    num_qubits = len(result.states[0].dims[0])
    num_sim_levels = result.states[0].dims[0][0]
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    if use_jax:
        assert JAX_AVAILABLE, 'JAX is not installed'
    
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
                out.create_dataset('tmax', shape=(max_iterations,), dtype='i')
                out.create_dataset('heff_coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('coeffs', shape=(max_iterations, basis_size), dtype='f')
                out.create_dataset('fit_success', shape=(max_iterations, basis_size), dtype='i')
                out.create_dataset('com', shape=(max_iterations, basis_size), dtype='f')
        else:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('ilogvs', shape=(tsize, time_evolution.shape[-1]), dtype='f')
                out.create_dataset('ilogu_coeffs', shape=(tsize, basis_size), dtype='f')
                
    if use_jax:
        unitaries = time_evolution

        time_evolution = jnp.array(time_evolution)
        basis_list = jnp.array(basis_list)
        
        # Normalized tlist
        tlist_norm = jnp.array(tlist / tend)
        # Basis array multiplied with time - concatenate with coeffs to produce Heff*t
        basis_t = jnp.array(tlist[:, None, None, None]) * basis_list[None, ...] / (2 ** (num_qubits - 1))
        
        @jax.jit
        def diff(params, ydata):
            return ydata - tlist_norm * params[0]

        @jax.jit
        def fun(params, ydata, mask):
            residual_masked = diff(params, ydata) * mask
            return jnp.sum(jnp.square(residual_masked))

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0, 0))
        def get_coeffs(ydata, tmax):
            mask = (jnp.arange(tsize) < tmax).astype(float)
            res = jsciopt.minimize(fun, jnp.array([ydata[tmax]]), args=(ydata, mask), method='BFGS')
            cumul = jnp.cumsum(diff(res.x, ydata) * mask)
            com = jnp.amax(jnp.abs(cumul)) / jnp.amax(jnp.abs(ydata) * mask)
            return res.x[0], res.success, com

        @jax.jit
        def update_heff(ilogus, tmax, heff_coeffs):
            # Divide the trace by two to account for the normalization of the generalized Paulis
            ilogu_coeffs = jnp.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.

            ## Do a linear fit to each component
            coeffs, success, com = get_coeffs(ilogu_coeffs.T, tmax)

            ## Update Heff with the best-fit coefficients
            is_update_candidate = success & (com < max_com)
            # If we have successfully made ilogu small that the fit range is the entire tlist,
            # also cut on the size of the coefficient (to avoid updating Heff with ~0 indefinitely)
            is_update_candidate = jax.lax.cond(tmax == tsize,
                                               lambda i, c: i & (jnp.abs(c) > min_coeff_ratio * jnp.amax(jnp.abs(heff_coeffs))),
                                               lambda i, _: i,
                                               is_update_candidate, coeffs)

            num_candidates = jnp.min(jnp.array([num_update_per_iteration, jnp.count_nonzero(is_update_candidate)]))

            ## Take the first n largest-coefficient candidates
            update_indices = jnp.argsort(jnp.where(is_update_candidate, -jnp.abs(coeffs), 0.))

            ## Update the output array taking the coefficient of the best-fit component
            mask = (jnp.arange(coeffs.shape[0]) < num_candidates).astype(float)
            coeffs_sorted = coeffs[update_indices] * mask
            coeffs = coeffs_sorted[jnp.argsort(update_indices)]

            if save_result_to:
                return heff_coeffs + coeffs, num_candidates > 0, ilogu_coeffs, coeffs, success, com
            else:
                return heff_coeffs + coeffs, num_candidates > 0
        
        @jax.jit
        def update_unitaries(time_evolution, basis_t, heff_coeffs):
            heff_t = jnp.tensordot(basis_t, heff_coeffs, (1, 0))
            exp_iheffs = matrix_ufunc(jnp.exp, 1.j * heff_t, hermitian=True, numpy=jnp)
            #eigvals, eigcols = jnp.linalg.eigh(heff_t)
            #eigrows = jnp.conjugate(jnp.moveaxis(eigcols, -2, -1))
            #exp_iheffs = jnp.matmul(eigcols * jnp.exp(eigvals)[..., None, :], eigrows)
            return jnp.matmul(time_evolution, exp_iheffs)
        
    else:
        tlist_norm = tlist / tend
        basis_t = tlist[:, None, None, None] * basis_list[None, ...] / (2 ** (num_qubits - 1))
        
        @partial(np.vectorize, otypes=[float, bool, float], signature='(t),()->(),(),()')
        def get_coeffs(ydata, tmax):
            try:
                popt, pcov = sciopt.curve_fit(lambda t, s: s * t,
                                              tlist_norm[:tmax],
                                              ydata[:tmax],
                                              p0=(ydata[tmax] / tlist_norm[tmax],))
            except:
                coeff = 0.
                success = False
                com = -1.
            else:
                coeff = popt[0] / tend
                success = True
                curve_opt = popt[0] * tlist_norm[:tmax]
                cumul = np.cumsum(ydata[:tmax] - curve_opt)
                com = np.amax(np.abs(cumul)) / np.amax(np.abs(ydata[:tmax]))
            
            return coeff, success, com

        def update_heff(ilogus, tmax, heff_coeffs):
            ## Extract the Pauli components
            # Divide the trace by two to account for the normalization of the generalized Paulis
            ilogu_coeffs = np.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.

            ## Do a linear fit to each component
            coeffs, success, com = get_coeffs(ilogu_coeffs[:tmax].T, tmax)

            ## Update Heff with the best-fit coefficients
            is_update_candidate = success & (com < max_com)
            # If we have successfully made ilogu small that the fit range is the entire tlist,
            # also cut on the size of the coefficient (to avoid updating Heff with ~0 indefinitely)
            if tmax == tsize:
                max_coeff = np.amax(np.abs(heff_coeffs))
                is_update_candidate &= (np.abs(coeffs) > min_coeff_ratio * max_coeff)

            num_candidates = min(num_update_per_iteration, np.count_nonzero(is_update_candidate))
            
            if num_candidates:
                if save_result_to:
                    return heff_coeffs, False, ilogu_coeffs, coeffs, success, com
                else:
                    return heff_coeffs, False

            ## Take the first n largest-coefficient candidates
            update_indices = np.argsort(np.where(is_update_candidate, -np.abs(coeffs), 0.))
            update_indices = update_indices[:num_candidates]

            ## Update the output array taking the coefficient of the best-fit component
            heff_coeffs[update_indices] += coeffs[update_indices]
            
            if save_result_to:
                return heff_coeffs, True, ilogu_coeffs, coeffs, success, com
            else:
                return heff_coeffs, True
        
        def update_unitaries(time_evolution, basis_t, heff_coeffs):
            heff_t = np.tensordot(basis_t, heff_coeffs, (1, 0))
            exp_iheffs = matrix_ufunc(np.exp, 1.j * heff_t, hermitian=True)
            return np.matmul(time_evolution, exp_iheffs)
            
        unitaries = time_evolution
    
    ## Output array
    heff_coeffs = np.zeros(basis_size)

    ## Iterative fit & subtract loop
    for iloop in range(max_iterations):
        logging.info('Fit-and-subtract iteration %d', iloop)
        
        ## Compute ilog(U(t))
        ilogus, ilogvs = matrix_ufunc(lambda u: -np.angle(u), unitaries, with_diagonals=True)
        
        ## Find the first t where an eigenvalue does a 2pi jump
        tmax = tsize
        for ilogv_ext in [np.amin(ilogvs, axis=1), -np.amax(ilogvs, axis=1)]:
            margin = 0.1
            hits_minus_pi = np.asarray(ilogv_ext < -np.pi + margin).nonzero()[0]
            if len(hits_minus_pi) != 0:
                tmax = min(tmax, hits_minus_pi[0])
    
        update_result = update_heff(ilogus, tmax, heff_coeffs)
        
        ## Save the computation results
        if save_result_to:
            heff_coeffs, updated, ilogu_coeffs, coeffs, success, com = update_result

            if save_iterations:
                with h5py.File(f'{save_result_to}_iter.h5', 'a') as out:
                    out['ilogvs'][iloop] = ilogvs
                    out['tmax'][iloop] = tmax
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
            unitaries = update_unitaries(time_evolution, basis_t, heff_coeffs)
            
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