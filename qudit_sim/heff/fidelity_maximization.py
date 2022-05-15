from typing import Optional, Union, Sequence
import logging
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jsciopt
import optax
import h5py

import rqutils.paulis as paulis
from rqutils.math import matrix_angle

from .leastsq_minimization import leastsq_minimization
from .common import get_ilogus_and_valid_it, heff_fidelity, compose_ueff

logger = logging.getLogger(__name__)

def fidelity_maximization(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    pauli_dim: Union[int, Sequence[int]],
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    jax_device_id: Optional[int] = None,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.05), 
    init: Union[str, np.ndarray] = 'slope_estimate',
    residual_adjust: bool = True,
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    **kwargs
) -> np.ndarray:
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    matrix_dim = np.prod(pauli_dim)
    assert time_evolution.shape == (tlist.shape[0], matrix_dim, matrix_dim), 'Inconsistent input shape'
    
    if jax_device_id is None:
        jax_device = None
    else:
        jax_device = jax.devices()[jax_device_id]
    
    ## Set up the Pauli product basis of the space of Hermitian operators
    basis = paulis.paulis(pauli_dim)
    # Flattened list of basis operators excluding the identity operator
    basis_list = jax.device_put(basis.reshape(-1, *basis.shape[-2:])[1:], device=jax_device)
    
    ## Set the initial parameter values
    if isinstance(init, str):
        if init == 'slope_estimate':
            ilogus, _, last_valid_it = get_ilogus_and_valid_it(time_evolution)
            if last_valid_it <= 1:
                raise RuntimeError('Failed to obtain an initial estimate of the slopes')
                
            ilogu_compos = paulis.components(ilogus, dim=pauli_dim).real
            init = ilogu_compos[last_valid_it - 1].reshape(-1)[1:] / tlist[last_valid_it - 1]

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
    def _loss_fn(time_evolution, heff_compos_norm, basis_list, tlist_norm):
        fidelity = heff_fidelity(time_evolution, heff_compos_norm, basis_list, tlist_norm, npmod=jnp)
        return 1. - 1. / (tlist_norm.shape[0] + 1) - jnp.mean(fidelity)
    
    if optimizer == 'minuit':
        from iminuit import Minuit
        
        loss_fn = jax.jit(lambda c: _loss_fn(time_evolution, c, basis_list, tlist_norm),
                          device=jax_device)
        grad = jax.jit(jax.grad(loss_fn), device=jax_device)
        
        minimizer = Minuit(loss_fn, initial, grad=grad)
        minimizer.strategy = 0
        
        logger.info('Running MIGRAD..')
        
        minimizer.migrad()
        
        logger.info('Done.')
        
        num_updates = minimizer.nfcn
        
        copt = minimizer.values
        
    else:
        if save_result_to:
            compo_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            loss_values = np.zeros(max_updates, dtype='f8')
            grad_values = np.zeros((max_updates, initial.shape[0]), dtype='f8')
            
        loss_fn = jax.jit(lambda params: _loss_fn(time_evolution, params['c'], basis_list, tlist_norm),
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
                compo_values[iup] = params['c'] / tlist[-1]
                loss_values[iup] = loss
                grad_values[iup] = gradient['c']
                
            max_grad = np.amax(np.abs(gradient['c']))
                
            logger.debug('Iteration %d: loss %f max_grad %f', iup, loss, max_grad)

            if max_grad < convergence:
                break
                
            params = new_params
                
        num_updates = iup + 1

        logger.info('Done after %d steps.', iup)
        
        copt = params['c']
        
    if residual_adjust:
        logger.info('Adjusting the Pauli components obtained from a linear fit to the residuals..')
        
        def fun(params, ydata):
            return jnp.sum(jnp.square(tlist_norm * params[0] - ydata + params[1]))

        @partial(jax.vmap, in_axes=(1, 0))
        def fit_compos(ydata, c0):
            res = jsciopt.minimize(fun, c0, args=(ydata,), method='BFGS')
            return res.x[0], res.success

        ueff_dagger = compose_ueff(copt, basis_list, tlist_norm, phase_factor=1.)
        target = jnp.matmul(time_evolution, ueff_dagger)

        ilogtargets = -matrix_angle(target)
        ilogtarget_compos = paulis.components(ilogtargets, dim=pauli_dim).real
        ilogtarget_compos = ilogtarget_compos.reshape(tlist_norm.shape[0], -1)[:, 1:]

        ## Do a linear fit to each component
        dopt, success = fit_compos(ilogtarget_compos, jax.device_put(np.zeros(copt.shape + (2,)), device=jax_device))

        failed = np.logical_not(success)
        if np.any(failed):
            failed_indices = np.nonzero(failed)[0]
            # compos list has the first element (identity) removed -> add 1 to the indices before unraveling
            failed_basis_indices = np.unravel_index(failed_indices + 1, basis.shape[:-2])
            # convert to a list of tuples
            failed_basis_indices = list(zip(*failed_basis_indices))
            residual_values_str = ', '.join(f'{idx_tuple}: {np.max(np.abs(ilogtarget_compos[:, idx]))}'
                                           for idx_tuple, idx in zip(failed_basis_indices, failed_indices))
            logger.warning('Residual adjustment failed for components %s', residual_values_str)
            
        copt += dopt
            
    heff_compos = np.concatenate(([0.], copt / tlist[-1])).reshape(basis.shape[:-2])

    if save_result_to:
        final_fidelity = np.concatenate(([1.], heff_fidelity(time_evolution, heff_compos, basis, tlist[1:])))
        with h5py.File(f'{save_result_to}_ext.h5', 'w') as out:
            out.create_dataset('final_fidelity', data=final_fidelity)
            if optimizer != 'minuit':
                out.create_dataset('compos', data=compo_values[:num_updates])
                out.create_dataset('loss', data=loss_values[:num_updates])
                out.create_dataset('grad', data=grad_values[:num_updates])
        
    logger.setLevel(original_log_level)
    
    return heff_compos
