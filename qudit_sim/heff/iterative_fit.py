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
    use_jax: bool = HAS_JAX
) -> np.ndarray:
    logging.getLogger().setLevel(log_level)
    
    num_qubits = len(result.states[0].dims[0])
    num_sim_levels = result.states[0].dims[0][0]
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    if use_jax:
        assert HAS_JAX, 'JAX is not installed'
    
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

    unitaries = time_evolution
    
    if use_jax:
        time_evolution = jnp.array(time_evolution)
        basis_list = jnp.array(basis_list)
        # Normalized tlist
        tlist_norm = jnp.array(tlist / tend)
        
        update_heff = jax.jit(partial(update_heff, numpy=jnp))
        update_unitaries = jax.jit(partial(update_unitaries, numpy=jnp))
        
        numpy = jnp
        sciopt = jax.scipy.optimize
        
    else:
        tlist_norm = tlist / tend
        
        numpy = np
        sciopt = scipy.optimize
                
    def diff(params, ydata):
        return ydata - tlist_norm * params[0]

    def fun(params, ydata, mask):
        residual_masked = diff(params, ydata) * mask
        return numpy.sum(numpy.square(residual_masked))

    @partial(numpy.vectorize, excluded=(1,), signature='(t),()->(),(),()')
    def get_coeffs(ydata, tmax):
        mask = (numpy.arange(tsize) < tmax).astype(float)
        res = jsciopt.minimize(fun, numpy.array([ydata[tmax]]), args=(ydata, mask), method='BFGS')
        cumul = numpy.cumsum(diff(res.x, ydata) * mask)
        com = numpy.amax(numpy.abs(cumul)) / numpy.amax(numpy.abs(ydata) * mask)
        return res.x[0], res.success, com
    
    def update_heff(ilogus, tmax, heff_coeffs):
        # Divide the trace by two to account for the normalization of the generalized Paulis
        ilogu_coeffs = numpy.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.

        ## Do a linear fit to each component
        coeffs, success, com = get_coeffs(ilogu_coeffs.T, tmax, numpy=numpy)

        ## Update Heff with the best-fit coefficients
        is_update_candidate = success & (com < max_com)
        # If we have successfully made ilogu small that the fit range is the entire tlist,
        # also cut on the size of the coefficient (to avoid updating Heff with ~0 indefinitely)
        coeff_nonnegligible = (numpy.abs(coeffs) > min_coeff_ratio * numpy.amax(numpy.abs(heff_coeffs)))
        is_update_candidate &= coeff_nonnegligible | (tmax == tsize)

        num_candidates = numpy.min(numpy.array([num_update_per_iteration, numpy.count_nonzero(is_update_candidate)]))

        ## Take the first n largest-coefficient candidates
        update_indices = numpy.argsort(numpy.where(is_update_candidate, -numpy.abs(coeffs), 0.))

        ## Update the output array taking the coefficient of the best-fit component
        mask = (numpy.arange(coeffs.shape[0]) < num_candidates).astype(float)
        coeffs_sorted = coeffs[update_indices] * mask
        coeffs = coeffs_sorted[numpy.argsort(update_indices)]

        if save_result_to:
            return heff_coeffs + coeffs, num_candidates > 0, ilogu_coeffs, coeffs, success, com
        else:
            return heff_coeffs + coeffs, num_candidates > 0

    def update_unitaries(heff_coeffs):
        heff = numpy.tensordot(basis_list, heff_coeffs, (0, 0))
        heff_t = tlist[:, None, None] * heff[None, ...]
        exp_iheffs = matrix_ufunc(numpy.exp, 1.j * heff_t, hermitian=True, numpy=numpy)
        return numpy.matmul(time_evolution, exp_iheffs)
    
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