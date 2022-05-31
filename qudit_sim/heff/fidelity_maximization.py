from typing import Optional, Union, Tuple
import logging
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
#import jax.scipy.optimize as jsciopt
import scipy.optimize as sciopt
import scipy.fft as scifft
import optax
import h5py

import rqutils.paulis as paulis
from rqutils.math import matrix_angle

from .leastsq_minimization import leastsq_minimization
from .common import get_ilogus_and_valid_it, heff_fidelity, compose_ueff
from ..config import config

logger = logging.getLogger(__name__)

def fidelity_maximization(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.05),
    init: Union[str, np.ndarray] = 'slope_estimate',
    l1_reg: float = 0.,
    residual_adjust: int = 2,
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    **kwargs
) -> np.ndarray:
    r"""Determine the effective Hamiltonian from the simulation result by fidelity maximization.

    The function tries to maximize the sum over all time points of the fidelity of :math:`\exp (-i H_{\mathrm{eff}} t)` with respect to
    the given time evolution operator.

    The behavior of the extraction can be controlled with the `optimizer`, `init`, and `residual_adjust` parameters. Allowed values for
    `init` are:

    - `'slope_estimate'`: Estimate the slope by :math:`\Delta C / \Delta t` where :math:`\Delta t` is the maximum time before one of
      the Pauli components of :math:`i\mathrm{log}U_H(t)` does a :math:`2\pi` jump.
    - `'leastsq'`: Perform `leastsq_minization` to obtain the initial estimate.
    - `'random'`: Start with random values.

    Args:
        time_evolution: Time evolution operator (shape (T, d1*d2*..., d1*d2*...))
        tlist: Time points (shape (T,)).
        dim: Subsystem dimensions.
        save_result_to: File name (without an extension) to save the intermediate results to.
        log_level: Log level.
        optimizer: The optimizer object. An `optax.GradientTransformation` object or `'minuit'`.
        init: Parameter initialization method. `'slope_estimate'`, `'leastsq'`, or `'random'`.
        l1_reg: L1 regularization coefficient. If not zero, adds a term `l1_reg * np.sum(np.abs(components))` to the loss function.
        residual_adjust: Number of iterations for linear fits with floating intercepts on the Pauli components of
            :math:`i \mathrm{log} \left[U_{H}(t) U_{\mathrm{eff}}(t)^{\dagger}\right]` after the best-fit :math:`U_{\mathrm{eff}}(t)`
            is found.
        max_updates: Number of maximum gradient-descent iterations.
        convergence: Convergence condition.

    Returns:
        Pauli components of the effective Hamiltonian.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    matrix_dim = np.prod(dim)
    assert time_evolution.shape == (tlist.shape[0], matrix_dim, matrix_dim), 'Inconsistent input shape'

    if not config.jax_devices:
        jax_device = None
    else:
        # parallel.parallel_map sets jax_devices[0] to the ID of the GPU to be used in this thread
        jax_device = jax.devices()[config.jax_devices[0]]

    ## Set up the Pauli product basis of the space of Hermitian operators
    basis = paulis.paulis(dim)
    # Flattened list of basis operators excluding the identity operator
    basis_list = jax.device_put(basis.reshape(-1, *basis.shape[-2:])[1:], device=jax_device)

    ## Set the initial parameter values
    if isinstance(init, str):
        if init == 'slope_estimate':
            ilogus, _, last_valid_it = get_ilogus_and_valid_it(time_evolution)
            if last_valid_it <= 1:
                raise RuntimeError('Failed to obtain an initial estimate of the slopes')

            ilogu_compos = paulis.components(ilogus, dim=dim).real
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

    initial = np.stack((init * tlist[-1], np.zeros_like(init)), axis=0)
    initial = jax.device_put(initial, device=jax_device)

    ## Working arrays (index 0 is trivial and in fact causes the grad to diverge)
    time_evolution = jax.device_put(time_evolution[1:], device=jax_device)
    tlist_norm = jax.device_put(tlist[1:] / tlist[-1], device=jax_device)

    ## Loss minimization (fidelity maximization)
    # Base loss function
    @partial(jax.jit, device=jax_device)
    def _loss_fn(time_evolution, params, basis_list, tlist_norm):
        fidelity = heff_fidelity(time_evolution, params[0], basis_list, tlist_norm, params[1], npmod=jnp)
        loss = 1. - 1. / (tlist_norm.shape[0] + 1) - jnp.mean(fidelity)
        if l1_reg != 0.:
            loss += l1_reg * jnp.sum(jnp.abs(heff_compos_norm))

        return loss

    if optimizer == 'minuit':
        from iminuit import Minuit

        @partial(jax.jit, device=jax_device)
        def loss_fn(params):
            return _loss_fn(time_evolution, params, basis_list, tlist_norm)

        grad = jax.jit(jax.grad(loss_fn), device=jax_device)

        minimizer = Minuit(loss_fn, initial, grad=grad)
        minimizer.strategy = 0

        logger.info('Running MIGRAD..')

        minimizer.migrad()

        logger.info('Done.')

        num_updates = minimizer.nfcn

        copt = minimizer.values[0]

    else:
        # With optax we have access to intermediate results, so we save them
        if save_result_to:
            compo_values = np.zeros((max_updates,) + initial.shape, dtype='f8')
            loss_values = np.zeros(max_updates, dtype='f8')
            grad_values = np.zeros((max_updates,) + initial.shape, dtype='f8')

        @partial(jax.jit, device=jax_device)
        def loss_fn(opt_params):
            return _loss_fn(time_evolution, opt_params['c'], basis_list, tlist_norm)

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn), device=jax_device)

        @jax.jit
        def step(opt_params, opt_state):
            loss, gradient = loss_and_grad(opt_params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            new_params = optax.apply_updates(opt_params, updates)
            return new_params, opt_state, loss, gradient

        logger.info('Starting maximization loop..')
        logger.info(initial.shape)

        opt_params = {'c': initial}
        opt_state = optimizer.init(opt_params)

        for iup in range(max_updates):
            new_params, opt_state, loss, gradient = step(opt_params, opt_state)

            if save_result_to:
                compo_values[iup] = opt_params['c']
                compo_values[iup][0] /= tlist[-1]
                loss_values[iup] = loss
                grad_values[iup] = gradient['c']

            max_grad = np.amax(np.abs(gradient['c']))

            logger.debug('Iteration %d: loss %f max_grad %f', iup, loss, max_grad)

            if max_grad < convergence:
                break

            opt_params = new_params

        num_updates = iup + 1

        logger.info('Done after %d steps.', iup)

        copt = np.array(opt_params['c'][0])

    if not np.all(np.isfinite(copt)):
        raise ValueError('Optimized components not finite')

    ## Residual adjustment (fidelity-maximizing copt is not necessarily the best set of parameters)
    ## We would like to extract the linear trend in the components of ilog[U_eff^dag U_H].
    ## Because a line fit can get confused by oscillations in the components, we work in the frequency domain.
    ## Objective:
    ##   For each component C, find the slope that minimizes sum_k |FFT(C)_k - FFT(slope * t + intercept)_k|^2
    ## From the normalization of tlist_norm:
    ## t = n / (N-1)
    ## FFT(slope * t + intercept)_k = N * delta_k0 * (intercept + 1/2 * slope)
    ##                              + 1/2 N/(N-1) * (1 - delta_k0) * slope * [-1 + i * cos(k pi / N) / sin(k pi / N)]
    ## The AC components of FFT(C) is concentrated in the first ~tenth of the spectrum (i.e. up to the drive frequency)
    ## so we adjust the objective to minimizing sum_{k=1/5N}^{k=4/5N}
    ## Furthermore, since C and line are both real, their transforms are symmetric: FFT_k = FFT*_{N-k}
    ## -> minimize sum_{k=1/5N}^{1/2N}
    ## Because of the lower k cutoff, the intercept is gone from the loss function:
    ##   L = sum_k |FFT(C)_k - 1/2 N/(N-1) * slope * [-1 + i cot(k pi / N)]|^2
    ##     = sum_k [(ReF_k + 1/2 N/(N-1) * slope)^2 + (ImF_k - 1/2 N/(N-1) * slope * cot(k pi / N))^2]
    ## At this point, there is no more a need for a numerical optimization:
    ##   dL/dslope = N/(N-1) sum_k [(ReF_k + 1/2 N/(N-1) * slope) - cot(k pi / N) (ImF_k - 1/2 N/(N-1) * slope * cot(k pi / N))]
    ##   dL/dslope = 0 <=> slope = -[sum_k (ReF_k - cot ImF_k)] / [1/2 N/(N-1) sum_k (1 + cot^2)]
    ## If we power-suppress both Fourier transforms at each k to further suppress local peaks
    ##   (slope)^(1/n) = -[sum_k (ReF_k^(1/n) - cot^(1/n) ImF_K^(1/n))] / [(1/2 N/(N-1))^(1/n) sum_k (1 + cot^(2/n))]
    if save_result_to:
        adjustments = np.zeros((residual_adjust,) + copt.shape)

    if residual_adjust > 0:
        # Get the discrete Fourier transform of a line with:
        # - The first and last 20% of samples removed to cut out the AC components
        # - Absolute values taken to the power of 1/4 to suppress the AC peaks
        nsamp = tlist_norm.shape[0]
        line_dft = _line_dft(nsamp, nsamp // 5, nsamp - nsamp // 5, 0.25)

    # Adjustment is done iteratively
    for iadj in range(residual_adjust):
        logger.info(f'Adjusting the Pauli components obtained from a linear fit to the residuals (iteration {iadj})')

        ueff_dagger = compose_ueff(copt, basis_list, tlist_norm, phase_factor=1.)
        target = np.matmul(time_evolution, ueff_dagger)

        ilogtargets = -matrix_angle(target)
        ilogtarget_compos = paulis.components(ilogtargets, dim=dim).real
        ilogtarget_compos = ilogtarget_compos.reshape(tlist_norm.shape[0], -1)[:, 1:]

        # Prepend the T=0 components
        compos = np.concatenate((np.zeros(ilogtarget_compos.shape[1]), ilogtarget_compos), axis=0)

        # scipy FFT handles multiple arrays "in parallel" (likely serial internally)
        ydata = scifft.fft(compos.T)

        # Find the

        for idx in np.arange(copt.shape[0]):
            loss_fn = lambda params: np.sum(np.square(params[0] * _line_dft - fydata))
            jac = lambda params: np.sum(2. * _line_dft * (params[0] * _line_dft - fydata)).reshape(1)
            res = sciopt.minimize(loss_fn, [0.], jac=jac)

            if res.success:
                copt[idx] += res.x[0]
                if save_result_to:
                    adjustments[iadj, idx] = res.x[0]

#             ## Try the AC + line curve first
#             # Find a rough estimate for the AC component
#             fourier = np.abs(scifft.fft(ydata))
#             dfourier = np.diff(fourier)

#             # First peak in the frequency spectrum
#             freq_idx = np.nonzero((dfourier[:-1] > 0.) & (dfourier[1:] < 0.))[0] + 1
#             if freq_idx.shape[0] == 0:
#                 freq_idx = [0]

#             freq_est = freq_idx[0] / xdata[-1] * np.pi * 2.

#             amp_est = (np.amax(ydata) - np.amin(ydata)) * 0.5

#             p0 = [0., 0., amp_est, freq_est, 0.]

#             popt, _ = sciopt.curve_fit(line_plus_ac, xdata, ydata, p0=p0)

#             mean_abs_diff = np.mean(np.abs(ydata - line_plus_ac(xdata, *popt)))
#             if mean_abs_diff < 1.e-2 * amp_est:
#                 # Mean absolute difference is less than a percent of the amplitude estimate
#                 # -> consider fit as OK
#                 copt[idx] += popt[0]
#                 if save_result_to:
#                     adjustments[iadj, idx] = popt[0]

#                 continue

#             ## Fallback to a simple line fit
#             p0 = [0., 0.]

#             popt, cov = sciopt.curve_fit(line, xdata, ydata, p0=p0)

#            if np.all(np.isfinite(cov)):
#                copt[idx] += popt[0]
#                if save_result_to:
#                    adjustments[iadj, idx] = popt[0]

    heff_compos = np.concatenate(([0.], copt / tlist[-1])).reshape(basis.shape[:-2])

    if save_result_to:
        final_fidelity = np.concatenate(([1.], heff_fidelity(time_evolution, heff_compos, basis, tlist[1:])))
        residual_adjustments = np.zeros((residual_adjust,) + basis.shape[:-2])
        if residual_adjust > 0:
            residual_adjustments.reshape(residual_adjust, -1)[:, 1:] = adjustments / tlist[-1]

        with h5py.File(f'{save_result_to}_ext.h5', 'w') as out:
            out.create_dataset('final_fidelity', data=final_fidelity)
            out.create_dataset('residual_adjustments', data=residual_adjustments)
            if optimizer != 'minuit':
                out.create_dataset('compos', data=compo_values[:num_updates])
                out.create_dataset('loss', data=loss_values[:num_updates])
                out.create_dataset('grad', data=grad_values[:num_updates])

    logger.setLevel(original_log_level)

    return heff_compos


def _line_dft(nsamp, begin, end, power):
    line_dft = np.empty(end - begin, dtype=np.complex)
    line_dft.real = np.full(end - begin, -np.power(0.5 * nsamp, power))

    theta = np.pi * np.arange(begin, end) / nsamp
    line_dft.imag = np.power(0.5 * nsamp * np.abs(np.cos(theta)) / np.sin(theta), power) * np.sign(np.cos(theta))

    return line_dft
