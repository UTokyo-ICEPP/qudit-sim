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
from .common import get_ilogus_and_valid_it, truncate_heff

## TODO Add jax_device keyword and allow parallel execution over multiple devices

# Having this function defined here allows a reuse of jitted code, presumably
@jax.jit
def _loss_fn(time_evolution, heff_coeffs_norm, basis_list, num_qubits, tlist_norm):
    fidelity = heff_fidelity(time_evolution, heff_coeffs_norm, basis_list, num_qubits, tlist_norm, npmod=jnp)
    return 1. - 1. / (tlist_norm.shape[0] + 1) - jnp.mean(fidelity)

def maximize_fidelity(
    result: qtp.solver.Result,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.05),
    init: Union[str, np.ndarray] = 'slope_estimate',
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    **kwargs
) -> np.ndarray:
    original_log_level = logging.getLogger().level
    logging.getLogger().setLevel(log_level)
    
    num_qubits = len(result.states[0].dims[0])
    num_sim_levels = result.states[0].dims[0][0]
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    tlist = result.times

    ## Set up the Pauli product basis of the space of Hermitian operators
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    del paulis
    # Flattened list of basis operators excluding the identity operator
    basis_list = jnp.array(basis.reshape(-1, *basis.shape[-2:])[1:])
    basis_size = basis_list.shape[0]
    
    ## Time evolution unitaries
    time_evolution = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    
    ## Save the setup
    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('num_qubits', data=num_qubits)
            out.create_dataset('num_sim_levels', data=num_sim_levels)
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('time_evolution', data=time_evolution)
            out.create_dataset('tlist', data=result.times)
    
    ## Set the initial parameter values
    if isinstance(init, str):
        if init == 'slope_estimate':
            ilogus, _, last_valid_it = get_ilogus_and_valid_it(time_evolution)
            ilogu_coeffs = np.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.
            init = ilogu_coeffs[last_valid_it - 1] / tlist[last_valid_it - 1]

        elif init == 'iterative_fit':
            logging.info('Performing iterative fit to estimate the initial parameter values')
            
            init = iterative_fit(
                result,
                comp_dim=num_sim_levels,
                log_level=log_level,
                **kwargs).reshape(-1)[1:]
            
        elif init == 'random':
            init = (np.random.random(basis_size) * 2. - 1.) / tlist[-1]
            
    initial = jnp.array(init * tlist[-1])
    
    ## Working arrays (index 0 is trivial and in fact causes the grad to diverge)
    time_evolution = jnp.array(time_evolution[1:])
    tlist_norm = jnp.array(tlist[1:] / tlist[-1])
    
    ## Loss minimization (fidelity maximization)
    if optimizer == 'minuit':
        loss_fn = jax.jit(lambda c: _loss_fn(time_evolution, c, basis_list, num_qubits, tlist_norm))
        grad = jax.jit(jax.grad(loss_fn))
        
        minimizer = Minuit(loss_fn, initial, grad=grad)
        minimizer.strategy = 0
        
        logging.info('Running MIGRAD..')
        
        minimizer.migrad()
        
        logging.info('Done.')
        
        num_updates = minimizer.nfcn

        heff_coeffs = np.concatenate(([0.], minimizer.values / tlist[-1])).reshape(basis.shape[:-2])
        
    else:
        if save_result_to:
            coeff_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            loss_values = np.zeros(max_updates, dtype='f8')
            grad_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            
        loss_fn = jax.jit(lambda params: _loss_fn(time_evolution, params['c'], basis_list, num_qubits, tlist_norm))
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        @jax.jit
        def step(params, opt_state):
            loss, gradient = loss_and_grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, gradient
        
        logging.info('Starting maximization loop..')
        
        params = {'c': initial}
        opt_state = optimizer.init(params)

        for iup in range(max_updates):
            new_params, opt_state, loss, gradient = step(params, opt_state)

            if save_result_to:
                coeff_values[iup] = params['c'] / tlist[-1]
                loss_values[iup] = loss
                grad_values[iup] = gradient['c']
                
            max_grad = np.amax(np.abs(gradient['c']))
                
            logging.debug('Iteration %d: loss %f max_grad %f', iup, loss, max_grad)

            if max_grad < convergence:
                break
                
            params = new_params
                
        num_updates = iup + 1

        logging.info('Done after %d steps.', iup)
        
        heff_coeffs = np.concatenate(([0.], params['c'] / tlist[-1])).reshape(basis.shape[:-2])

    if save_result_to:
        final_fidelity = np.concatenate(([1.], heff_fidelity(time_evolution, heff_coeffs, basis, num_qubits, tlist[1:])))
        with h5py.File(f'{save_result_to}.h5', 'a') as out:
            out.create_dataset('heff_coeffs', data=heff_coeffs)
            out.create_dataset('final_fidelity', data=final_fidelity)
            if optimizer != 'minuit':
                out.create_dataset('coeffs', data=coeff_values[:num_updates])
                out.create_dataset('loss', data=loss_values[:num_updates])
                out.create_dataset('grad', data=grad_values[:num_updates])
        
    heff_coeffs = truncate_heff(heff_coeffs, num_sim_levels, comp_dim, num_qubits)
        
    logging.getLogger().setLevel(original_log_level)
    
    return heff_coeffs
