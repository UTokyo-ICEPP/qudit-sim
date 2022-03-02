from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import string
import sys
from functools import partial
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscila
from iminuit import Minuit
import optax
import h5py

from .paulis import make_prod_basis

def process_fidelity(time_evolution, basis, coeffs, tlist, numpy=np):
    num_qubits = len(coeffs.shape)
    axes = list(range(num_qubits))
    heff = numpy.tensordot(basis, coeffs, (axes, axes)) / (2 ** (num_qubits - 1))
    
    return _process_fidelity(time_evolution, heff, tlist, numpy)


def _process_fidelity(time_evolution, heff, tlist, numpy):
    if numpy is np:
        eigh = np.linalg.eig
    elif numpy is jnp:
        eigh = jax.scipy.linalg.eigh
        
    heff_t = tlist[:, None, None] * numpy.tile(heff[None, ...], (tlist.shape[0], 1, 1))
    
    def unitarize(mat):
        eigvals, eigvecs = eigh(mat)
        dd = numpy.tile(eigvals[:, None, :], (1, eigvecs.shape[1], 1))
        return numpy.matmul(eigvecs * numpy.exp(1.j * dd), eigvecs.conj().transpose(0, 2, 1))
    
    udag_t = unitarize(heff_t)
    
    tr_u_udag = numpy.trace(numpy.matmul(time_evolution, udag_t), axis1=1, axis2=2)
    fidelity = (numpy.square(tr_u_udag.real) + numpy.square(tr_u_udag.imag)) / (heff.shape[-1] ** 2)
    
    return fidelity


def maximize_process_fidelity(
    time_evolution: np.ndarray,
    paulis: np.ndarray,
    num_qubits: int,
    tlist: np.ndarray,
    save_result_to: Optional[str] = None,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.01),
    init: Union[str, np.ndarray] = 'log_unitary_fit',
    max_updates: int = 1000,
    convergence: float = 1.e-6,
    **kwargs
) -> np.ndarray:
    
    basis = jnp.array(make_prod_basis(paulis, num_qubits))
    
    # Compute the loss from the flattened coeffs array
    basis_no_id = basis.reshape(-1, basis.shape[-2], basis.shape[-1])[1:]
    
    if isinstance(init, str) and init == 'log_unitary_fit':
        initial = fit_to_log_unitary(
            time_evolution,
            paulis,
            num_qubits,
            tlist,
            save_result_to=save_result_to,
            **kwargs)[1:] * tlist[-1]
    else:
        initial = jnp.array(init * tlist[-1])

    time_evolution = jnp.asarray(time_evolution)
    tlist_norm = jnp.array(tlist / tlist[-1])
    
    def loss_fn(coeffs):
        heff = jnp.tensordot(basis_no_id, coeffs, (0, 0)) / (2 ** (num_qubits - 1))
        fidelity = _process_fidelity(time_evolution[1:], heff, tlist_norm[1:], jnp)
        fidelity = jnp.concatenate((jnp.array([1.]), fidelity))
        fidelity_spectrum = jnp.fft.fft(fidelity)
        lowpass = jnp.exp(-2. * jnp.linspace(0., 1., tlist_norm.shape[0] - 1))
        return jnp.sum(-jnp.abs(fidelity_spectrum[1:]) * lowpass)
        #return 1. - 1. / tlist_norm.shape[0] - jnp.mean(fidelity)
    
    if optimizer == 'minuit':
        minimizer = Minuit(loss_fn, initial, grad=jax.grad(loss_fn))
        minimizer.strategy = 0
        minimizer.migrad()

        coeffs = np.concatenate(([0.], minimizer.values / tlist[-1])).reshape(basis.shape[:-2])
        
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

        for iup in range(max_updates):
            params, opt_state, loss, gradient = step(params, opt_state)

            loss_values[iup] = loss

            if jnp.amax(jnp.abs(gradient['c'])) < convergence:
                break

        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('loss', data=loss_values)
                out.create_dataset('num_updates', data=np.array(iup))

        coeffs = np.concatenate(([0.], params['c'] / tlist[-1])).reshape(basis.shape[:-2])
        
    return coeffs
