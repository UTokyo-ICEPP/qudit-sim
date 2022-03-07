from typing import Optional, Union
import logging

import numpy as np
import jax
import jax.numpy as jnp
from iminuit import Minuit
import optax
import h5py
import qutip as qtp

from ..paulis import make_generalized_paulis, make_prod_basis
from ..utils import matrix_ufunc, heff_fidelity
from .iterative_fit import iterative_fit
from .common import truncate_heff

def maximize_fidelity(
    result: qtp.solver.Result,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.01),
    init: Union[str, np.ndarray] = 'iterative_fit',
    max_updates: int = 1000,
    convergence: float = 1.e-6,
    **kwargs
) -> np.ndarray:
    original_log_level = logging.getLogger().level
    logging.getLogger().setLevel(log_level)
    
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
    basis_list = jnp.array(basis.reshape(-1, *basis.shape[-2:])[1:])
    basis_size = basis_list.shape[0]
    
    ## Time evolution unitaries
    time_evolution = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    time_evolution = jnp.array(time_evolution[1:])
    
    tlist_norm = jnp.array(tlist[1:] / tlist[-1])
    
    if isinstance(init, str):
        if init == 'iterative_fit':
            init = iterative_fit(
                result,
                comp_dim=num_sim_levels,
                log_level=log_level,
                **kwargs)[1:]
        elif init == 'random':
            init = np.random.random(basis_size) * 2. - 1.

    initial = jnp.array(init * tlist[-1])
    
    def loss_fn(heff_coeffs_norm):
        fidelity = heff_fidelity(time_evolution, heff_coeffs_norm, basis_list, tlist_norm, num_qubits, npmod=jnp)
        return 1. - 1. / tsize - jnp.mean(fidelity)
    
    if optimizer == 'minuit':
        minimizer = Minuit(loss_fn, initial, grad=jax.grad(loss_fn))
        minimizer.strategy = 0
        
        logging.info('Running MIGRAD..')
        
        minimizer.migrad()
        
        logging.info('Done.')

        heff_coeffs = np.concatenate(([0.], minimizer.values / tlist[-1])).reshape(basis.shape[:-2])
        
    else:
        loss_and_grad = jax.jit(jax.value_and_grad(lambda params: loss_fn(params['c'])))

        params = {'c': initial}
        opt_state = optimizer.init(params)

        loss_values = np.empty(max_updates, dtype='f8')

        @jax.jit
        def step(params, opt_state):
            loss, gradient = loss_and_grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, gradient
        
        logging.info('Starting maximization loop..')

        for iup in range(max_updates):
            params, opt_state, loss, gradient = step(params, opt_state)
            
            logging.info('Iteration %d: loss %f', iup, loss)

            loss_values[iup] = loss

            if jnp.amax(jnp.abs(gradient['c'])) < convergence:
                break
                
        logging.info('Done.')

        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'w') as out:
                out.create_dataset('loss', data=loss_values)
                out.create_dataset('num_updates', data=np.array(iup))

        heff_coeffs = np.concatenate(([0.], params['c'] / tlist[-1])).reshape(basis.shape[:-2])
        
    heff_coeffs = truncate_heff(heff_coeffs, num_sim_levels, comp_dim, num_qubits)
        
    logging.getLogger().setLevel(original_log_level)
    
    return heff_coeffs
