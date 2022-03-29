from typing import Optional, Union
import logging
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import h5py

from ..paulis import make_generalized_paulis, make_prod_basis, extract_coefficients
from .leastsq_minimization import leastsq_minimization
from .common import get_ilogus_and_valid_it, heff_fidelity

logger = logging.getLogger(__name__)

def fidelity_maximization(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    num_qubits: int = 1,
    num_sim_levels: int = 2,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    jax_device_id: Optional[int] = None,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.05), 
    init: Union[str, np.ndarray] = 'slope_estimate',
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    **kwargs
) -> np.ndarray:
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    matrix_dim = num_sim_levels ** num_qubits
    assert time_evolution.shape == (tlist.shape[0], matrix_dim, matrix_dim), 'Inconsistent input shape'
    
    if jax_device_id is None:
        jax_device = None
    else:
        jax_device = jax.devices()[jax_device_id]
    
    ## Set up the Pauli product basis of the space of Hermitian operators
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    del paulis
    # Flattened list of basis operators excluding the identity operator
    basis_list = jax.device_put(basis.reshape(-1, *basis.shape[-2:])[1:], device=jax_device)
    
    ## Set the initial parameter values
    if isinstance(init, str):
        if init == 'slope_estimate':
            ilogus, _, last_valid_it = get_ilogus_and_valid_it(time_evolution)
            if last_valid_it <= 1:
                raise RuntimeError('Failed to obtain an initial estimate of the slopes')
                
            ilogu_coeffs = extract_coefficients(ilogus, num_qubits)
            init = ilogu_coeffs[last_valid_it - 1].reshape(-1)[1:] / tlist[last_valid_it - 1]

        elif init == 'leastsq':
            logger.info('Performing iterative fit to estimate the initial parameter values')
            
            init = leastsq_minimization(
                time_evolution,
                tlist,
                num_qubits=num_qubits,
                num_sim_levels=num_sim_levels,
                log_level=log_level,
                jax_device_id=jax_device_id,
                **kwargs).reshape(-1)[1:]
            
        elif init == 'random':
            init = (np.random.random(basis_list.shape) * 2. - 1.) / tlist[-1]
            
    initial = jax.device_put(init * tlist[-1], device=jax_device)
    
    ## Working arrays (index 0 is trivial and in fact causes the grad to diverge)
    time_evolution = jax.device_put(time_evolution[1:], device=jax_device)
    tlist_norm = jax.device_put(tlist[1:] / tlist[-1], device=jax_device)
    
    ## Loss minimization (fidelity maximization)
    @partial(jax.jit, device=jax_device)
    def _loss_fn(time_evolution, heff_coeffs_norm, basis_list, num_qubits, tlist_norm):
        fidelity = heff_fidelity(time_evolution, heff_coeffs_norm, basis_list, tlist_norm, num_qubits, npmod=jnp)
        return 1. - 1. / (tlist_norm.shape[0] + 1) - jnp.mean(fidelity)
    
    if optimizer == 'minuit':
        from iminuit import Minuit
        
        loss_fn = jax.jit(lambda c: _loss_fn(time_evolution, c, basis_list, num_qubits, tlist_norm),
                          device=jax_device)
        grad = jax.jit(jax.grad(loss_fn), device=jax_device)
        
        minimizer = Minuit(loss_fn, initial, grad=grad)
        minimizer.strategy = 0
        
        logger.info('Running MIGRAD..')
        
        minimizer.migrad()
        
        logger.info('Done.')
        
        num_updates = minimizer.nfcn

        heff_coeffs = np.concatenate(([0.], minimizer.values / tlist[-1])).reshape(basis.shape[:-2])
        
    else:
        if save_result_to:
            coeff_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            loss_values = np.zeros(max_updates, dtype='f8')
            grad_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            
        loss_fn = jax.jit(lambda params: _loss_fn(time_evolution, params['c'], basis_list, num_qubits, tlist_norm),
                          device=jax_device)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn), device=jax_device)

        @jax.jit
        def step(params, opt_state):
            loss, gradient = loss_and_grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, gradient
        
        logger.info('Starting maximization loop..')
        
        params = {'c': initial}
        opt_state = optimizer.init(params)

        for iup in range(max_updates):
            new_params, opt_state, loss, gradient = step(params, opt_state)

            if save_result_to:
                coeff_values[iup] = params['c'] / tlist[-1]
                loss_values[iup] = loss
                grad_values[iup] = gradient['c']
                
            max_grad = np.amax(np.abs(gradient['c']))
                
            logger.debug('Iteration %d: loss %f max_grad %f', iup, loss, max_grad)

            if max_grad < convergence:
                break
                
            params = new_params
                
        num_updates = iup + 1

        logger.info('Done after %d steps.', iup)
        
        heff_coeffs = np.concatenate(([0.], params['c'] / tlist[-1])).reshape(basis.shape[:-2])

    if save_result_to:
        final_fidelity = np.concatenate(([1.], heff_fidelity(time_evolution, heff_coeffs, basis, tlist[1:], num_qubits)))
        with h5py.File(f'{save_result_to}.h5', 'a') as out:
            out.create_dataset('final_fidelity', data=final_fidelity)
            if optimizer != 'minuit':
                out.create_dataset('coeffs', data=coeff_values[:num_updates])
                out.create_dataset('loss', data=loss_values[:num_updates])
                out.create_dataset('grad', data=grad_values[:num_updates])
        
    logger.setLevel(original_log_level)
    
    return heff_coeffs
