"""Effective Hamiltonian extraction frontend."""

import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import Device
import numpy as np
import optax
import qutip as qtp

from rqutils.math import matrix_exp, matrix_angle
import rqutils.paulis as paulis

from .gates.components import gate_components
from .heff_tools import unitary_subtraction, trace_norm_squared, heff_fidelity
from ..config import config
from ..expression import Parameter
from ..frame import FrameSpec, SystemFrame
from ..hamiltonian import HamiltonianBuilder
from ..parallel import parallel_map
from ..pulse import GaussianSquare
from ..pulse_sim import pulse_sim, PulseSimParameters, exponentiate_hstat
from ..sim_result import PulseSimResult, save_sim_result, load_sim_result
from ..unitary import truncate_matrix, closest_unitary

QuditSpec = Union[str, Tuple[str, ...]]
FrequencySpec = Union[float, Tuple[float, ...]]
AmplitudeSpec = Union[float, complex, Tuple[Union[float, complex], ...]]
InitSpec = Union[np.ndarray, Dict[Tuple[int, ...], Union[float, Tuple[float, bool]]]]

logger = logging.getLogger(__name__)

default_optimizer_args = optax.exponential_decay(0.001, 10, 0.99)

def find_heff(
    hgen: HamiltonianBuilder,
    qudit: QuditSpec,
    frequency: Union[FrequencySpec, List[FrequencySpec], np.ndarray],
    amplitude: Union[AmplitudeSpec, List[AmplitudeSpec], np.ndarray],
    comp_dim: Optional[Union[int, Tuple[int, ...]]] = None,
    frame: FrameSpec = 'dressed',
    cycles: float = 1000.,
    ramp_cycles: float = 100.,
    pulse_sim_solver: str = 'qutip',
    init: Optional[InitSpec] = None,
    optimizer: str = 'adam',
    optimizer_args: Any = default_optimizer_args,
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    convergence_window: int = 5,
    min_fidelity: float = 0.9,
    zero_suppression: bool = True,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union[np.ndarray, List[np.ndarray]]:
    r"""Determine the effective Hamiltonian from the result of constant-drive simulations.

    The function first sets up one-sided GuassianSquare pulses according to the ``cycles`` and
    ``ramp_cycles`` parameters and runs the pulse simulation. The resulting time evolution operator
    in the plateau region of the pulse is then used to extract the effective Hamiltonian through the
    maximization of the unitary fidelity.

    Multiple qudits can be driven simultaneously by passing tuples to ``qudit``, ``frequency``, and
    ``amplitude`` parameters.

    To evaluate multiple drive specifications in parallel (e.g. when performing an amplitude scan),
    pass a list to either of ``frequency`` or ``amplitude``. When both are lists, their lengths must
    match.

    Args:
        hgen: Qudits and couplings specification.
        qudit: The qudit(s) to apply the drive to.
        frequency: Drive frequency(ies).
        amplitude: Drive amplitude(s).
        comp_dim: Dimensionality of the computational space. If not set, number of levels in the
            Hamiltonian is used.
        frame: System frame.
        cycles: Number of drive signal cycles in the plateau of the GaussianSquare pulse.
        ramp_cycles: Number of drive signal cycles to use for ramp-up.
        init: Initial values of the effective Hamiltonian components. An array specifies all initial
            values. If a dict is passed, the format is ``{index: value or (value, fixed)}`` where
            ``index`` is the component index (tuple of ints), ``value`` is the value to set the
            component to, and ``fixed`` specifies whether the component remains fixed in the fit
            (default False). Initial values for the unmentioned components are set from a crude
            slope estimate.
        optimizer: The name of the optax function to use as the optimizer.
        optimizer_args: Arguments to the optimizer.
        max_updates: Maximum number of optimization iterations.
        convergence: The cutoff value for the change of fidelity within the last
            ``convergence_window`` iterations.
        convergence_window: The number of updates to use to compute the mean of change of fidelity.
        min_fidelity: Final fidelity threshold. If the unitary fidelity of the fit result goes below
            this value, the fit is repeated over a shortened interval.
        zero_suppression: Zero-fix the effective Hamiltonian components that appear to vanish at the
            end of a Gaussian(Square) pulse.
        save_result_to: File name (without the extension) to save the extraction results to.
        log_level: Log level.

    Returns:
        An array of Pauli components or a list thereof (if a list is passed to ``frequency`` and/or
        ``amplitude``).
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    hgen_drv, tlist, drive_args, time_range = add_drive_for_heff(hgen, qudit, frequency, amplitude,
                                                                 cycles, ramp_cycles)

    sim_result = pulse_sim(hgen_drv, tlist, drive_args=drive_args, frame=frame,
                           solver=pulse_sim_solver, save_result_to=save_result_to,
                           log_level=log_level)

    if isinstance(init, np.ndarray):
        hstat = None
    else:
        hstat = hgen_drv.build_hstatic(frame='lab')

    if isinstance(drive_args, dict):
        components = heff_fit(sim_result, comp_dim=comp_dim, time_range=time_range,
                              init=init, hstat=hstat, optimizer=optimizer,
                              optimizer_args=optimizer_args, max_updates=max_updates,
                              convergence=convergence, convergence_window=convergence_window,
                              min_fidelity=min_fidelity, zero_suppression=zero_suppression,
                              save_result_to=save_result_to, log_level=log_level)

    else:
        num_tasks = len(drive_args)

        logger_names = list(f'{__name__}.{i}' for i in range(len(drive_args)))

        if save_result_to:
            num_digits = int(np.log10(num_tasks)) + 1
            fmt = f'%0{num_digits}d'
            save_result_paths = list(os.path.join(save_result_to, fmt % i)
                                     for i in range(num_tasks))
        else:
            save_result_paths = [None] * num_tasks

        args = sim_result

        kwarg_keys = ('logger_name', 'save_result_to',)
        kwarg_values = list(zip(logger_names, save_result_paths))

        common_kwargs = {'comp_dim': comp_dim, 'time_range': time_range, 'init': init,
                         'hstat': hstat, 'optimizer': optimizer, 'optimizer_args': optimizer_args,
                         'max_updates': max_updates,
                         'convergence': convergence, 'convergence_window': convergence_window,
                         'min_fidelity': min_fidelity, 'zero_suppression': zero_suppression,
                         'log_level': log_level}

        components = parallel_map(heff_fit, args=args, kwarg_keys=kwarg_keys,
                                  kwarg_values=kwarg_values, common_kwargs=common_kwargs,
                                  log_level=log_level, thread_based=True)

    logger.setLevel(original_log_level)

    return components


def add_drive_for_heff(
    hgen: HamiltonianBuilder,
    qudit: QuditSpec,
    frequency: Union[FrequencySpec, List[FrequencySpec], np.ndarray],
    amplitude: Union[AmplitudeSpec, List[AmplitudeSpec], np.ndarray],
    cycles: float = 1000.,
    ramp_cycles: float = 100.
):
    if isinstance(frequency, np.ndarray):
        frequency = list(frequency)
    if isinstance(amplitude, np.ndarray):
        amplitude = list(amplitude)
    if isinstance(cycles, np.ndarray):
        cycles = list(cycles)
    if isinstance(ramp_cycles, np.ndarray):
        ramp_cycles = list(ramp_cycles)

    num_tasks = None
    amp_scan = False
    freq_scan = False

    if isinstance(frequency, list) and isinstance(amplitude, list):
        if len(frequency) != len(amplitude):
            raise ValueError('Inconsistent length of frequency and amplitude lists')

        num_tasks = len(frequency)
        amp_scan = True
        freq_scan = True

    elif isinstance(frequency, list):
        num_tasks = len(frequency)
        amplitude = [amplitude] * num_tasks
        freq_scan = True

    elif isinstance(amplitude, list):
        num_tasks = len(amplitude)
        frequency = [frequency] * num_tasks
        amp_scan = True

    else:
        amplitude = [amplitude]
        frequency = [frequency]

    hgen_drv = hgen.copy(clear_drive=True)

    if not isinstance(qudit, tuple):
        qudit = (qudit,)
        amplitude = list((a,) for a in amplitude)
        frequency = list((f,) for f in frequency)

    if len(qudit) > 0:
        max_cycle = 2. * np.pi / np.amin(frequency)
        ramp = max_cycle * ramp_cycles
        width = max_cycle * cycles
        duration = 2. * ramp + width

        tlist_args = {'points_per_cycle': 8, 'duration': duration, 'frame': 'lab'}

        time_range = (ramp, duration - ramp)

        sigma = ramp / 4. # let the ramp correspond to four sigmas

        for iq, qid in enumerate(qudit):
            if amp_scan:
                amp = Parameter(f'amp_{qid}')
            else:
                amp = amplitude[0][iq]

            if freq_scan:
                freq = Parameter(f'freq_{qid}')
            else:
                freq = frequency[0][iq]

            gs_pulse = GaussianSquare(duration, amp, sigma, width)
            hgen_drv.add_drive(qid, frequency=freq, amplitude=gs_pulse)

        if freq_scan:
            tlist = list()
            for freqs in frequency:
                tlist_args['freq_args'] = {f'freq_{qid}': f for qid, f in zip(qudit, freqs)}
                tlist.append(hgen_drv.make_tlist(**tlist_args))
        else:
            tlist = hgen_drv.make_tlist(**tlist_args)

    else:
        tlist_args = {'points_per_cycle': 8, 'num_cycles': int(cycles), 'frame': 'lab'}
        tlist = hgen_drv.make_tlist(**tlist_args)
        time_range = (0., tlist[-1])

    if num_tasks is None:
        drive_args = dict()
    else:
        drive_args = list()
        for freqs, amps in zip(frequency, amplitude):
            if amp_scan:
                task_drive_args = {f'amp_{qid}': a for qid, a in zip(qudit, amps)}
            else:
                task_drive_args = dict()

            if freq_scan:
                task_drive_args.update({f'freq_{qid}': f for qid, f in zip(qudit, freqs)})

            drive_args.append(task_drive_args)

    return hgen_drv, tlist, drive_args, time_range


def heff_fit(
    sim_result: Union[PulseSimResult, str],
    comp_dim: Optional[Union[int, Tuple[int, ...]]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    init: Optional[InitSpec] = None,
    hstat: Optional[qtp.Qobj] = None,
    optimizer: str = 'adam',
    optimizer_args: Any = 0.05,
    max_updates: int = 10000,
    convergence: float = 1.e-4,
    convergence_window: int = 5,
    min_fidelity: float = 0.9,
    zero_suppression: bool = True,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    logger_name: str = __name__
) -> np.ndarray:
    r"""Perform a fidelity-maximizing fit to the result of constant-drive simulation.

    The function takes the result of a constant-drive (with ring-up) simulation and identifies the
    effective Hamiltonian that best describes the time evolution.

    Args:
        sim_result: Simulation result object or the name of the file that contains one.
        comp_dim: Dimensionality of the computational space.
        time_range: Time range when the drive pulses are on the plateau.
        optimizer: The name of the optax function to use as the optimizer.
        optimizer_args: Arguments to the optimizer.
        max_updates: Maximum number of optimization iterations.
        convergence: The cutoff value for the change of fidelity within the last
            ``convergence_window`` iterations.
        convergence_window: The number of updates to use to compute the mean of change of fidelity.
        min_fidelity: Final fidelity threshold. If the unitary fidelity of the fit result goes below
            this value, the fit is repeated over a shortened interval.
        zero_suppression: Zero-fix the effective Hamiltonian components that appear to vanish at the
            end of a Gaussian(Square) pulse.
        save_result_to: File name (without the extension) to save the extraction results to.
        log_level: Log level.

    Returns:
        An array of Pauli components.
    """
    logger = logging.getLogger(logger_name)
    original_log_level = logger.level
    logger.setLevel(log_level)

    if isinstance(sim_result, str):
        filename = sim_result
        sim_result = load_sim_result(filename)

        with h5py.File(filename, 'r') as source:
            if comp_dim is None:
                comp_dim = tuple(source['comp_dim'][()])
            if time_range is None:
                time_range = tuple(sim_result.times[list(source['fit_range'][()])])

        if save_result_to:
            save_sim_result(f'{save_result_to}.h5', sim_result)

    frame = sim_result.frame

    num_qudits = frame.num_qudits

    if comp_dim is None:
        comp_dim = frame.dim
    elif isinstance(comp_dim, int):
        comp_dim = (comp_dim,) * num_qudits

    if time_range is None:
        time_range = (0., sim_result.times[-1])

    ## Time bins for the GaussianSquare plateau
    flat_start = np.searchsorted(sim_result.times, time_range[0], 'right') - 1
    flat_end = np.searchsorted(sim_result.times, time_range[1], 'right') - 1

    ## Truncate the unitaries if comp_dim is less than the system dimension
    time_evolution = truncate_matrix(sim_result.states, frame.dim, comp_dim)

    ## Find the initial values for the fit
    logger.info('Determining the initial values for Heff and offset..')

    heff_init, fixed, offset_init = _find_init(time_evolution, sim_result.times, comp_dim,
                                               init, hstat, sim_result.frame, zero_suppression,
                                               flat_start, flat_end, logger)
    # Fix the identity component because its gradient is identically zero
    fixed[(0,) * num_qudits] = True

    logger.debug('Heff initial values: %s', heff_init)
    logger.debug('Fixed components: %s', fixed)
    logger.debug('Offset initial values: %s', offset_init)

    ## Time normalization: make tlist equivalent to arange(len(tlist)) / len(tlist)
    # Normalize the tlist and Heff components for a better numerical stability in fit
    time_norm = sim_result.times[-1] + sim_result.times[1]
    tlist_full = sim_result.times.copy()
    tlist_full /= time_norm
    heff_init *= time_norm

    fit_start = flat_start
    fit_end = flat_end

    attempt = 1

    while True:
        time_evolution_in_range = time_evolution[fit_start:fit_end + 1]
        tlist = tlist_full[fit_start:fit_end + 1] - tlist_full[fit_start]

        logger.info('Maximizing mean fidelity..')
        result = _maximize_fidelity(time_evolution_in_range, tlist, comp_dim,
                                    heff_init, fixed, offset_init,
                                    optimizer, optimizer_args, max_updates,
                                    convergence, convergence_window,
                                    bool(save_result_to), logger)

        heff_compos, offset_compos, intermediate_results = result

        logger.debug('Heff component values: %s', heff_compos / time_norm)
        logger.debug('Offset component values: %s', offset_compos)

        ## Test the optimized values - is the minimum fidelity above the threshold?
        fidelity = heff_fidelity(time_evolution[flat_start:flat_end + 1], heff_compos, offset_compos,
                                 tlist_full[flat_start:flat_end + 1] - tlist_full[flat_start])
        minf = np.amin(fidelity)

        if minf > min_fidelity:
            logger.info('Found Heff and offsets with minimum fidelity %f', minf)
            break
        else:
            logger.info('Minimum fidelity value: %f', minf)
            logger.info('Fit attempt %d (fit_end=%d) did not produce a valid result.',
                        attempt, fit_end)

            fit_end = int(fit_end / 1.5)

            if fit_end <= 64:
                logger.warning('Reached the minimum possible fit_end value %d.', fit_end)
                break

            logger.info('Reducing fit_end to %d.', fit_end)

            attempt += 1

    # Recover the normalization
    heff_compos /= time_norm

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'a') as out:
            out.create_dataset('comp_dim', data=comp_dim)
            out.create_dataset('fit_range', data=[fit_start, fit_end])

            out.create_dataset('fixed', data=fixed)

            out.create_dataset('components', data=heff_compos)
            out.create_dataset('offset_components', data=offset_compos)

            if intermediate_results is not None:
                intermediate_results['heff_compos'] /= time_norm
                for key, value in intermediate_results.items():
                    out.create_dataset(key, data=value)

    logger.setLevel(original_log_level)

    return heff_compos


def _find_init(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    init: Union[InitSpec, None],
    hstat: Union[qtp.Qobj, None],
    sim_frame: SystemFrame,
    zero_suppression: bool,
    flat_start: int,
    flat_end: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the initial values of the effective Hamiltonian components."""
    unitaries = closest_unitary(time_evolution)
    generator, eigenvalues = matrix_angle(unitaries, with_diagonals=True)
    generator_compos = paulis.components(-generator, dim=dim).real
    offset_init = generator_compos[flat_start]

    fixed = np.zeros_like(offset_init, dtype=bool)

    if isinstance(init, np.ndarray):
        heff_init = init
    else:
        ## 1. Set the initial Heff values from a rough slope estimate
        # Find the first t where an eigenvalue does a 2pi jump

        last_valid_tidx = flat_end
        for eigenvalue_envelope in [np.amin(eigenvalues, axis=1), -np.amax(eigenvalues, axis=1)]:
            margin = 0.1
            hits_minus_pi = np.asarray(eigenvalue_envelope < -np.pi + margin).nonzero()[0]
            if hits_minus_pi.shape[0] != 0:
                last_valid_tidx = min(last_valid_tidx, hits_minus_pi[0] - 1)

        if last_valid_tidx > 0:
            # If C_i = a t_i + b = a (t_L / L * i) + b,
            # S = sum_{i=0}^{L} C_i = a t_L / L * 1/2 * L * (L + 1) + b (L + 1)
            # => a = (S / (L + 1) - b) * 2/t_L

            heff_init = np.sum(generator_compos[flat_start:last_valid_tidx + 1], axis=0)
            heff_init /= last_valid_tidx - flat_start + 1
            heff_init -= generator_compos[flat_start]
            heff_init *= 2. / (tlist[last_valid_tidx] - tlist[flat_start])

        else:
            logger.warning('Failed to obtain an initial estimate of the slopes')
            heff_init = np.zeros(generator_compos.shape[1:])

        ## 2. Find off-diagonals with finite values under no drive -> static terms; zero slope

        psi0 = qtp.qeye(hstat.dims[0])

        parameters = PulseSimParameters(sim_frame, tlist, psi0)

        states, _ = exponentiate_hstat(hstat, parameters, logger.name)

        time_evolution_nodrv = truncate_matrix(states, sim_frame.dim, dim)
        unitaries_nodrv = closest_unitary(time_evolution_nodrv)
        generator_nodrv = matrix_angle(unitaries_nodrv)
        nodrv_compos = paulis.components(-generator_nodrv, dim=dim).real
        is_finite = np.logical_not(np.all(np.isclose(nodrv_compos, 0., atol=1.e-4), axis=0))

        symmetry = paulis.symmetry(dim)
        finite_offdiag = is_finite & np.asarray(symmetry, dtype=bool)

        logger.debug('Off-diagonals with finite values under no drive: %s',
                     list(zip(*np.nonzero(finite_offdiag))))

        heff_init[finite_offdiag] = 0.

        ## 3. Find finite values converging to zero at the end of the pulse -> zero slope & offset

        is_significant = np.any(np.abs(generator_compos) > 0.1, axis=0)
        quiet_at_end = (np.abs(generator_compos[-1]) <= 0.01 * np.amax(generator_compos, axis=0))
        converges = is_significant & quiet_at_end

        logger.debug('Converging components: %s', list(zip(*np.nonzero(converges))))

        heff_init[converges] = 0.
        offset_init[converges] = 0.
        if zero_suppression:
            fixed[converges] = True

        ## 4. Values set by hand

        if isinstance(init, dict):
            for key, value in init.items():
                if not isinstance(key, tuple) or len(key) != sim_frame.num_qudits:
                    raise ValueError(f'Invalid init key {key}')

                if isinstance(value, tuple):
                    if value[0] is not None:
                        heff_init[key] = value[0]
                    fixed[key] = value[1]
                else:
                    heff_init[key] = value

    return heff_init, fixed, offset_init


_vexpm = jax.vmap(jax.scipy.linalg.expm, in_axes=0, out_axes=0)
_matexp = partial(matrix_exp, hermitian=-1, npmod=jnp)

def _maximize_fidelity(
    time_evolution: np.ndarray,
    tlist: np.ndarray,
    dim: Tuple[int, ...],
    heff_init: np.ndarray,
    fixed: np.ndarray,
    offset_init: np.ndarray,
    optimizer: str,
    optimizer_args: Any,
    max_updates: int,
    convergence: float,
    convergence_window: int,
    return_intermediate: bool,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    ## Set up the Pauli product basis of the space of Hermitian operators
    basis = paulis.paulis(dim)

    ## Static and dynamic Heff components and basis list
    floating = np.logical_not(fixed)

    heff_static = np.tensordot(heff_init[fixed], basis[fixed], (0, 0))

    ## Working arrays
    # parallel.parallel_map sets jax_devices[0] to the ID of the GPU to be used in this thread
    jax_device = jax.devices()[config.jax_devices[0]]

    time_evolution = jax.device_put(time_evolution, device=jax_device)
    tlist = jax.device_put(tlist, device=jax_device)
    heff_static = jax.device_put(heff_static, device=jax_device)
    heff_init_dev = jax.device_put(heff_init[floating], device=jax_device)
    heff_basis = jax.device_put(basis[floating], device=jax_device)
    offset_init_dev = jax.device_put(offset_init.reshape(-1)[1:], device=jax_device)
    offset_basis = jax.device_put(basis.reshape(-1, *basis.shape[-2:])[1:], device=jax_device)

    ## Loss function
    def loss_fn(heff_compos, offset_compos):
        # The following basically computes the same thing as heff_tools.heff_fidelity.
        # We need to reimplement the fidelity calculation here because the gradient
        # diverges when matrix_exp is used with a diagonal hermitian where some parameters
        # only control off-diagonal elements.

        heff = jnp.tensordot(heff_compos, heff_basis, (0, 0))
        heff += heff_static
        offset = jnp.tensordot(offset_compos, offset_basis, (0, 0))
        generator = heff[None, ...] * tlist[:, None, None] + offset
        unitary = _vexpm(1.j * generator)

        target = jnp.matmul(time_evolution, unitary)
        fidelity = trace_norm_squared(target, npmod=jnp)

        return -jnp.mean(fidelity)

    ## Minimize
    if optimizer == 'minuit':
        heff_compos, offset_compos = _minimize_minuit(loss_fn, heff_init_dev, offset_init_dev,
                                                      logger)
        intermediate_results = None
    else:
        ## Set up the optimizer, loss & grad functions, and the parameter update function
        if not isinstance(optimizer_args, tuple):
            optimizer_args = (optimizer_args,)

        grad_trans = getattr(optax, optimizer)(*optimizer_args)

        heff_compos, offset_compos, intermediate_results = _minimize(grad_trans, loss_fn,
                                                                     heff_init_dev, offset_init_dev,
                                                                     max_updates, convergence,
                                                                     convergence_window,
                                                                     return_intermediate, logger)

    if not np.all(np.isfinite(heff_compos)) or not np.all(np.isfinite(offset_compos)):
        raise ValueError('Optimized components not finite')

    heff_compos_full = heff_init.copy()
    heff_compos_full[floating] = heff_compos
    offset_compos_full = np.concatenate(([0.], offset_compos)).reshape(offset_init.shape)

    return heff_compos_full, offset_compos_full, intermediate_results


def _minimize_minuit(
    loss_fn: Callable,
    heff_init: jnp.DeviceArray,
    offset_init: jnp.DeviceArray,
    logger: logging.Logger
):
    from iminuit import Minuit

    jax_device = heff_init.device()
    logger.debug('Using JAX device %s', jax_device)

    initial = jnp.concatenate((heff_init, offset_init))
    num_heff = heff_init.shape[0]

    fcn = jax.jit(lambda p: loss_fn(p[:num_heff], p[num_heff:]))
    grad = jax.jit(jax.grad(fcn))

    minimizer = Minuit(fcn, initial, grad=grad)
    minimizer.strategy = 0

    logger.info('Running MIGRAD..')

    # https://github.com/google/jax/issues/11478
    # default_device is thread-local
    with jax.default_device(jax_device):
        minimizer.migrad()

    logger.info('Done.')

    return minimizer.values[0], minimizer.values[1]


def _minimize(
    grad_trans,
    loss_fn: Callable,
    heff_init: jnp.DeviceArray,
    offset_init: jnp.DeviceArray,
    max_updates: int,
    convergence: float,
    convergence_window: int,
    return_intermediate: bool,
    logger: logging.Logger
):
    jax_device = heff_init.device()
    logger.debug('Using JAX device %s', jax_device)

    value_and_grad = jax.value_and_grad(lambda p: loss_fn(p['heff'], p['offset']))

    @jax.jit
    def step(opt_params, opt_state):
        # https://github.com/google/jax/issues/11478
        # default_device is thread-local
        with jax.default_device(jax_device):
            loss, gradient = value_and_grad(opt_params)
            updates, opt_state = grad_trans.update(gradient, opt_state)
            new_params = optax.apply_updates(opt_params, updates)

        return new_params, opt_state, loss, gradient

    ## Start the fidelity maximization loop
    logger.info('Starting maximization loop..')

    opt_params = {'heff': heff_init, 'offset': offset_init}
    with jax.default_device(jax_device):
        opt_state = grad_trans.init(opt_params)

    ## Compile the step function
    step = step.lower(opt_params, opt_state).compile()

    losses = np.ones(convergence_window)

    if return_intermediate:
        intermediate_results = {
            'heff_compos': np.zeros((max_updates, heff_init.shape[0]), dtype='f8'),
            'heff_grads': np.zeros((max_updates, heff_init.shape[0]), dtype='f8'),
            'offset_compos': np.zeros((max_updates, offset_init.shape[0]), dtype='f8'),
            'offset_grads': np.zeros((max_updates, offset_init.shape[0]), dtype='f8'),
            'loss': np.zeros(max_updates, dtype='f8')
        }
    else:
        intermediate_results = None

    for iup in range(max_updates):
        new_params, opt_state, loss, gradient = step(opt_params, opt_state)

        if return_intermediate:
            intermediate_results['loss'][iup] = loss
            for key in ['heff', 'offset']:
                intermediate_results[f'{key}_compos'][iup] = opt_params[key]
                intermediate_results[f'{key}_grads'][iup] = gradient[key]

        losses[iup % convergence_window] = loss
        change = np.amax(losses) - np.amin(losses)

        logger.debug('Iteration %d: loss %f, last %d change %f',
                     iup, loss, convergence_window, change)

        if change < convergence:
            break

        opt_params = new_params

    num_updates = iup + 1

    logger.info('Done after %d steps.', num_updates)

    if return_intermediate:
        for key, value in intermediate_results.items():
            intermediate_results[key] = value[:num_updates]

    return np.array(opt_params['heff']), np.array(opt_params['offset']), intermediate_results
