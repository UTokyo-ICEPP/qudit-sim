"""Effective Hamiltonian extraction frontend."""

from typing import Any, List, Tuple, Optional, Union
import os
from functools import partial
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import jax
import jax.numpy as jnp
import optax

import rqutils.paulis as paulis
from rqutils.math import matrix_exp, matrix_angle

from .util import PulseSimResult
from .config import config
from .parallel import parallel_map
from .heff_tools import unitary_subtraction, trace_norm_squared

logger = logging.getLogger(__name__)

def find_heff(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: int = 2,
    optimizer: str = 'adam',
    optimizer_args: Any = 0.05,
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
) -> Union[np.ndarray, List[np.ndarray]]:
    r"""Determine the effective Hamiltonian from the result of constant-drive simulations.

    The input to this function must be the result of pulse_sim with constant drives.

    Args:
        sim_result: Result from pulse_sim.
        comp_dim: Dimensionality of the computational space.
        method: Pauli component extraction method. Currently possible values are 'fidelity' and 'leastsq'.
        method_params: Optional keyword arguments to pass to the extraction function.
        save_result_to: File name (without the extension) to save the extraction results to.
        log_level: Log level.

    Returns:
        An array of Pauli components or a list thereof (if a list is passed to ``sim_result``).
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    if isinstance(sim_result, list):
        num_tasks = len(sim_result)

        common_kwargs = {'comp_dim': comp_dim, 'optimizer': optimizer, 'optimizer_args': optimizer_args,
                         'max_updates': max_updates, 'convergence': convergence, 'log_level': log_level}

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

        components = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys,
                                  kwarg_values=kwarg_values, common_kwargs=common_kwargs,
                                  log_level=log_level, thread_based=True)

    else:
        components = _run_single(sim_result.states, sim_result.times, sim_result.dim,
                                 comp_dim=comp_dim, optimizer=optimizer, optimizer_args=optimizer_args,
                                 max_updates=max_updates, convergence=convergence,
                                 save_result_to=save_result_to, log_level=log_level)

    logger.setLevel(original_log_level)

    return components


def _run_single(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: tuple,
    comp_dim: int,
    optimizer: str,
    optimizer_args: Any,
    max_updates: int,
    convergence: float,
    save_result_to: Union[str, None],
    log_level: int,
    logger_name: str = __name__
):
    logger = logging.getLogger(logger_name)

    matrix_dim = np.prod(dim)
    assert time_evolution.shape == (tlist.shape[0], matrix_dim, matrix_dim), 'Inconsistent input shape'

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('num_qudits', data=len(dim))
            out.create_dataset('num_sim_levels', data=dim[0])
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('time_evolution', data=time_evolution)
            out.create_dataset('tlist', data=tlist)

    heff_compos, offset_compos = _maximize_fidelity(time_evolution,
                                                    tlist,
                                                    dim,
                                                    optimizer,
                                                    optimizer_args,
                                                    max_updates,
                                                    convergence,
                                                    save_result_to)

    offset_range = _estimate_offset_range(time_evolution,
                                          tlist,
                                          dim,
                                          heff_compos,
                                          offset_compos,
                                          save_result_to)

    components = np.stack((heff_compos, offset_compos, offset_range), axis=0)

    if dim[0] != comp_dim:
        reduced_dim = (comp_dim,) * len(dim)
        components_original = components
        components = paulis.truncate(components, reduced_dim)
    else:
        components_original = None

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'a') as out:
            out.create_dataset('components', data=components)
            if components_original is not None:
                out.create_dataset('components_original', data=components_original)

    return components


_vexpm = jax.vmap(jax.scipy.linalg.expm, in_axes=0, out_axes=0)
_matexp = partial(matrix_exp, hermitian=-1, npmod=jnp)

def _maximize_fidelity(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    optimizer: str,
    optimizer_args: Any,
    max_updates: int,
    convergence: float,
    save_result_to: Union[str, None]
) -> Tuple[np.ndarray, np.ndarray]:
    if not config.jax_devices:
        jax_device = None
    else:
        # parallel.parallel_map sets jax_devices[0] to the ID of the GPU to be used in this thread
        jax_device = jax.devices()[config.jax_devices[0]]

    ## Time normalization: make tlist equivalent to arange(len(tlist)) / len(tlist)
    time_norm = tlist[-1] + tlist[1]

    ## Set up the Pauli product basis of the space of Hermitian operators
    basis = paulis.paulis(dim)
    # Flattened list of basis operators excluding the identity operator
    basis_list = jax.device_put(basis.reshape(-1, *basis.shape[-2:])[1:], device=jax_device)
    # Matrix size of the Pauli products
    matrix_dim = np.prod(dim)

    ## Set the initial Heff values from a rough slope estimate
    # Compute ilog(U(t))
    ilogus, ilogvs = matrix_angle(time_evolution, with_diagonals=True)
    ilogus *= -1.

    # Find the first t where an eigenvalue does a 2pi jump
    last_valid_tidx = ilogus.shape[0]
    for ilogv_ext in [np.amin(ilogvs, axis=1), -np.amax(ilogvs, axis=1)]:
        margin = 0.1
        hits_minus_pi = np.asarray(ilogv_ext < -np.pi + margin).nonzero()[0]
        if len(hits_minus_pi) != 0:
            last_valid_tidx = min(last_valid_tidx, hits_minus_pi[0])

    if last_valid_tidx <= 1:
        raise RuntimeError('Failed to obtain an initial estimate of the slopes')

    ilogu_compos = paulis.components(ilogus, dim=dim).real

    init = ilogu_compos[last_valid_tidx - 1] / tlist[last_valid_tidx - 1]

    # Reshape and truncate the components array to match the basis_list
    init = init.reshape(-1)[1:]

    ## Stack the initial parameter values
    # initial[0]: init multiplied by time_norm
    # initial[1]: offset components (initialized to zero)
    initial = np.stack((init * time_norm, np.zeros_like(init)), axis=0)
    initial = jax.device_put(initial, device=jax_device)

    ## Working arrays
    time_evolution = jax.device_put(time_evolution, device=jax_device)
    tlist_norm = jax.device_put(tlist / time_norm, device=jax_device)

    ## Loss function
    offdiagonals = np.nonzero(paulis.symmetry(dim).reshape(-1)[1:])

    @partial(jax.jit, device=jax_device)
    def _loss_fn(heff_compos, offset_compos):
        # The following basically computes the same thing as heff_tools.heff_fidelity.
        # We need to reimplement the fidelity calculation here because the gradient
        # diverges when matrix_exp is used with a diagonal hermitian where some parameters
        # only control off-diagonal elements.

        components = heff_compos[None, :] * tlist_norm[:, None] + offset_compos
        is_diagonal = ~jnp.any(components.T[offdiagonals], axis=0)
        has_diagonal = jnp.any(is_diagonal)

        mat = 1.j * jnp.tensordot(components, basis_list, (1, 0))

        unitary = jax.lax.cond(has_diagonal, _vexpm, _matexp, mat)

        target = jnp.matmul(time_evolution, unitary)

        fidelity = trace_norm_squared(target, npmod=jnp)

        return -jnp.mean(fidelity)

    if optimizer == 'minuit':
        from iminuit import Minuit

        @partial(jax.jit, device=jax_device)
        def loss_fn(params):
            return _loss_fn(params[0], params[1])

        grad = jax.jit(jax.grad(loss_fn), device=jax_device)

        minimizer = Minuit(loss_fn, initial, grad=grad)
        minimizer.strategy = 0

        logger.info('Running MIGRAD..')

        minimizer.migrad()

        logger.info('Done.')

        num_updates = minimizer.nfcn

        compos_norm = minimizer.values[0]
        offset_compos = minimizer.values[1]

    else:
        ## Set up the optimizer, loss & grad functions, and the parameter update function
        if not isinstance(optimizer_args, tuple):
            optimizer_args = (optimizer_args,)

        grad_trans = getattr(optax, optimizer)(*optimizer_args)

        @partial(jax.jit, device=jax_device)
        def loss_fn(opt_params):
            return _loss_fn(opt_params['c'][0], opt_params['c'][1])

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn), device=jax_device)

        @jax.jit
        def step(opt_params, opt_state):
            loss, gradient = loss_and_grad(opt_params)
            updates, opt_state = grad_trans.update(gradient, opt_state)
            new_params = optax.apply_updates(opt_params, updates)
            return new_params, opt_state, loss, gradient

        ## With optax we have access to intermediate results, so we save them
        if save_result_to:
            compo_values = np.zeros((max_updates,) + initial.shape, dtype='f8')
            loss_values = np.zeros(max_updates, dtype='f8')
            grad_values = np.zeros((max_updates,) + initial.shape, dtype='f8')

        ## Start the fidelity maximization loop
        logger.info('Starting maximization loop..')

        opt_params = {'c': initial}
        opt_state = grad_trans.init(opt_params)

        for iup in range(max_updates):
            new_params, opt_state, loss, gradient = step(opt_params, opt_state)

            if save_result_to:
                compo_values[iup] = opt_params['c']
                compo_values[iup][0] /= time_norm
                loss_values[iup] = loss
                grad_values[iup] = gradient['c']

            max_grad = np.amax(np.abs(gradient['c']))

            logger.debug('Iteration %d: loss %f max_grad %f', iup, loss, max_grad)

            if max_grad < convergence:
                break

            opt_params = new_params

        num_updates = iup + 1

        logger.info('Done after %d steps.', num_updates)

        compos_norm = np.array(opt_params['c'][0])
        offset_compos = np.array(opt_params['c'][1])

        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('compos', data=compo_values[:num_updates])
                out.create_dataset('loss', data=loss_values[:num_updates])
                out.create_dataset('grad', data=grad_values[:num_updates])

    if not np.all(np.isfinite(compos_norm)) or not np.all(np.isfinite(offset_compos)):
        raise ValueError('Optimized components not finite')

    ## Recover the shape and normalization of the components arrays before returning
    heff_compos = np.concatenate(([0.], compos_norm / time_norm)).reshape(basis.shape[:-2])

    offset_compos = np.concatenate(([0.], offset_compos)).reshape(basis.shape[:-2])

    return heff_compos, offset_compos


def _estimate_offset_range(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    heff_compos: np.ndarray,
    offset_compos: np.ndarray,
    save_result_to: Union[str, None]
) -> np.ndarray:
    target = unitary_subtraction(time_evolution, heff_compos, offset_compos, tlist)
    ilogtargets = -matrix_angle(target)
    ilogtarget_compos = paulis.components(ilogtargets, dim=dim).real

    offset_range = np.amax(ilogtarget_compos, axis=0) - np.amin(ilogtarget_compos, axis=0)
    offset_range /= 2.

    return offset_range
