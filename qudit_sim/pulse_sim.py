"""Pulse simulation frontend."""

from collections.abc import Callable, Sequence
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from functools import partial
from threading import Condition
from typing import Any, Optional, Union
import jax
import jax.numpy as jnp
from jax.experimental.ode import _odeint_wrapper
import numpy as np
from scipy.sparse import csr_array
import qutip as qtp

from rqutils import ArrayType
from rqutils.math import matrix_exp
from .frame import FrameSpec, QuditFrame, SystemFrame
from .hamiltonian import Hamiltonian, HamiltonianBuilder
from .parallel import parallel_map
from .sim_result import PulseSimResult, save_sim_result
from .unitary import closest_unitary

LOG = logging.getLogger(__name__)

Tlist = Union[np.ndarray, tuple[int, int], dict[str, Union[int, float]], float]
Array = Union[np.ndarray, csr_array]
QObject = Union[qtp.Qobj, Array, tuple[Array, ...], dict[str, Array]]


def pulse_sim(
    hgen: HamiltonianBuilder,
    tlist: Union[Tlist, list[Tlist]] = (10, 100),
    psi0: Optional[Union[QObject, list[QObject]]] = None,
    drive_args: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
    e_ops: Optional[Union[Sequence[QObject], list[Sequence[QObject]]]] = None,
    frame: Union[FrameSpec, list[FrameSpec]] = 'dressed',
    final_only: Union[bool, list[bool]] = False,
    reunitarize: Union[bool, list[bool]] = True,
    solver: str = 'qutip',
    solver_options: Optional[Union[dict, list[dict]]] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union['PulseSimResult', list['PulseSimResult']]:
    r"""Run a pulse simulation.

    Build the Hamiltonian terms from the HamiltonianBuilder, determine the time points for the
    simulation if necessary, and run ``qutip.sesolve``.

    Regardless of the setup of the input (initial state and expectation ops), the simulation is run
    in the lab frame with the identity operator as the initial state to obtain the time evolution
    operator for each time point. The evolution operators are then transformed into the specified
    frame, if any, or the default frame of the HamiltonianBuilder, before being applied to the
    initial state (or initial unitary). If expectation ops are given, their expectation values are
    computed from the resulting evolved states.

    All parameters except ``hgen``, ``save_result_to``, and ``log_level`` can be given as lists to
    trigger a parallel execution of ``sesolve``. If more than one parameter is a list, their lengths
    must be identical, and all parameters are "zipped" together to form the argument lists of
    individual single simulation jobs. Parallel jobs are run as separate processes if QuTiP
    (internally scipy)-based integrator is used, and as threads if JAX integrator is used.

    .. rubric:: Reunitarization of time evolution

    Due to the finite numerical precision in solving the Schrodinger equation, the computed time
    evolution matrices will stop being unitary at some point, with the rate of divergence dependent
    on the complexity of the Hamiltonian. In general, the closest unitary :math:`U`, i.e., one with
    the smallest 2-norm :math:`\lVert U-A \rVert`, to an operator :math:`A` can be calculated via a
    singular value decomposition:

    .. math::

        A & = V \Sigma W^{\dagger}, \\
        U & = V W^{\dagger}.

    This function has an option ``reunitarize`` to apply the above decomposition and recompute the
    time evolution matrices. Because reunitarizing at each time step can be rather slow (mostly
    because of the overhead in calling ``sesolve``), when this option is turned on (default),
    ``sesolve`` is called in time intervals of 512 time steps, and the entire array of
    evolution matrices from each interval is reunitarized (longer interval lengths result in larger
    reunitarization corrections).

    .. rubric:: Implementation notes (why we return an original object instead of the QuTiP result)

    When the coefficients of the time-dependent Hamiltonian are compiled (preferred method), QuTiP
    creates a transient python module with file name generated from the code hash, PID, and the
    current time. When running multiple simulations in parallel this is not strictly safe, and so we
    enclose ``sesolve`` in a context with a temporary directory in this function. The transient
    module is then deleted at the end of execution, but that in turn causes an error when this
    function is called in a subprocess and if we try to return the QuTiP result object directly
    through e.g. multiprocessing.Pipe. Somehow the result object tries to carry with it something
    defined in the transient module, which would therefore need to be pickled together with the
    returned object. But the transient module file is gone by the time the parent process receives
    the result from the pipe. So, the solution was to just return a "sanitized" object, consisting
    of plain ndarrays.

    Args:
        hgen: A HamiltonianBuilder.
        tlist: Time points to use in the simulation or a pair ``(points_per_cycle, num_cycles)``
            where in the latter case the cycle of the fastest oscillating term in the Hamiltonian
            will be used. A dict that can be unpacked as keyword arguments to the
            ``make_tlist`` method of HamiltonianBuilder is also accepted. Additionally, for
            ``solver='jax'``, a single float can be given, in which case the simulation will for
            the time interval ``[0, tlist]``.
        psi0: Initial state Qobj or array. Defaults to the identity operator appropriate for the
            given Hamiltonian.
        drive_args: Second parameter passed to drive amplitude functions (if callable).
        e_ops: list of observables passed to the QuTiP solver.
        frame: System frame to represent the simulation results in.
        final_only: Throw away intermediate states and expectation values to save memory. Due to
            implementation details, implies ``reunitarize=True``.
        reunitarize: Whether to reunitarize the time evolution matrices. Note that SVD can be very
            slow for large system sizes and ``final_only=False``.
        solver_options: If solver is qutip, QuTiP solver options. If solver is jax, a tuple
            representing ``(rtol, atol, mxstep, hmax)`` parameters of ``odeint``.
        save_result_to: File name (without the extension) to save the simulation result to.
        log_level: Log level.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = LOG.level
    LOG.setLevel(log_level)

    num_tasks = None

    if isinstance(drive_args, list):
        num_tasks = len(drive_args)

    # Put inputs to PulseSimParameters in a zipped list
    parallel_params = {
        'tlist': (tlist, (np.ndarray, tuple, dict, float)),
        'psi0': (psi0, (qtp.Qobj, np.ndarray, csr_array, tuple, dict)),
        'e_ops': (e_ops, Sequence),
        'frame': (frame, (str, dict, Sequence)),
        'final_only': (final_only, bool),
        'reunitarize': (reunitarize, bool),
        'solver_options': (solver_options, dict)
    }

    # Pack parallelizable arguments into per-job dicts
    singleton_params = []
    for key, (value, typ) in parallel_params.items():
        if (isinstance(value, list)
                and (typ is None or all(isinstance(elem, typ) for elem in value))):
            if num_tasks is None:
                num_tasks = len(value)
            elif num_tasks != len(value):
                raise ValueError('lists with inconsistent lengths passed as arguments')
        else:
            singleton_params.append(key)

    if num_tasks is None:
        hamiltonian = build_hamiltonian(hgen, solver=solver)

        raw_args = {key: value for key, (value, typ) in parallel_params.items()}
        parameters = compose_parameters(hgen, solver=solver, **raw_args)

        result = _run_single(hamiltonian, parameters, drive_args=drive_args, solver=solver,
                             save_result_to=save_result_to)

    else:
        raw_args_list = [{} for _ in range(num_tasks)]
        for key in singleton_params:
            for itask in range(num_tasks):
                raw_args_list[itask][key] = parallel_params[key][0]

        for key in set(parallel_params.keys()) - set(singleton_params):
            for itask in range(num_tasks):
                raw_args_list[itask][key] = parallel_params[key][0][itask]

        if not isinstance(drive_args, list):
            drive_args = [drive_args] * num_tasks

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            num_digits = int(np.log10(num_tasks)) + 1

            def save_result_path(itask):
                return os.path.join(save_result_to, f'%0{num_digits}d' % itask)
        else:
            def save_result_path(_):
                return None

        hamiltonian = build_hamiltonian(hgen, solver=solver)

        parameters_list = list(compose_parameters(hgen, solver=solver, **raw_args)
                               for raw_args in raw_args_list)

        if solver == 'jax':
            thread_based = True

            common_args = (hamiltonian,)
            args = parameters_list
            arg_position = 1
            kwarg_keys = ('drive_args', 'logger_name', 'save_result_to')
            kwarg_values = [(drive_args[itask], f'{__name__}.{itask}', save_result_path(itask))
                            for itask in range(num_tasks)]

        else:
            thread_based = False

            # Convert the functional Hamiltonian coefficients to arrays before passing them to
            # parallel_map
            hamiltonians = []
            for params, dargs in zip(parameters_list, drive_args):
                hamiltonians.append(hamiltonian.evaluate_coeffs(params.tlist, dargs, npmod=np))

            common_args = None
            args = list(zip(hamiltonians, parameters_list))
            arg_position = (0, 1)
            kwarg_keys = ('logger_name', 'save_result_to')
            kwarg_values = [(f'{__name__}.{itask}', save_result_path(itask))
                            for itask in range(num_tasks)]

        result = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys,
                              kwarg_values=kwarg_values, arg_position=arg_position,
                              common_args=common_args, common_kwargs={'solver': solver},
                              log_level=log_level, thread_based=thread_based)

    LOG.setLevel(original_log_level)

    return result


@dataclass
class PulseSimParameters:
    """Pulse simulation parameters."""

    frame: SystemFrame
    tlist: np.ndarray
    psi0: Union[np.ndarray, None] = None
    e_ops: Union[list[np.ndarray], None] = None
    final_only: bool = False
    reunitarize: bool = True
    solver_options: dict = field(default_factory=dict)


def build_hamiltonian(
    hgen: HamiltonianBuilder,
    solver: str = 'qutip'
) -> Any:
    """Build the Hamiltonian according to the pulse_sim_solver option.

    Args:
        hgen: A HamiltonianBuilder instance.
        solver: ``'qutip'`` or ``'jax'``
    """
    if solver == 'jax':
        h_terms = hgen.build(frame='lab', as_timefn=True)

        if len(h_terms) == 1 and isinstance(h_terms[0], qtp.Qobj):
            # Will just exponentiate
            hamiltonian = h_terms
        else:
            @jax.jit
            def hamiltonian(u_t, t, drive_args):
                h_t = h_terms[0].full()
                for qobj, fn in h_terms[1:]:
                    h_t += qobj.full() * fn(t, drive_args, npmod=jnp)

                return -1.j * jnp.matmul(h_t, u_t)
    else:
        hamiltonian = hgen.build(frame='lab')

    return hamiltonian


def compose_parameters(
    hgen: HamiltonianBuilder,
    tlist: Tlist = (10, 100),
    psi0: Optional[QObject] = None,
    e_ops: Optional[Sequence[QObject]] = None,
    frame: FrameSpec = 'dressed',
    final_only: bool = False,
    reunitarize: bool = True,
    solver: str = 'qutip',
    solver_options: Optional[dict] = None
) -> PulseSimParameters:
    """Generate the arguments for parallel jobs using the HamiltonianBuilder."""
    # Time points
    if isinstance(tlist, tuple):
        tlist = {'points_per_cycle': tlist[0], 'num_cycles': tlist[1]}
    if isinstance(tlist, dict):
        tlist = hgen.make_tlist(frame='lab', **tlist)
    if isinstance(tlist, (float, int)):
        tlist = np.array([0., tlist])

    if isinstance(psi0, qtp.Qobj):
        psi0 = np.squeeze(psi0.full())
    elif isinstance(psi0, csr_array):
        psi0 = psi0.todense()
    elif isinstance(psi0, tuple):
        qudit_psi0 = [qtp.Qobj(state) for state in psi0]
        psi0 = np.squeeze(qtp.tensor(qudit_psi0).full())
    elif isinstance(psi0, dict):
        qudit_psi0 = []
        for qid in hgen.qudit_ids():
            qudit_psi0.append(qtp.Qobj(psi0[qid]))
        psi0 = np.squeeze(qtp.tensor(qudit_psi0).full())

    if e_ops is not None:
        new_e_ops = []
        for qobj in e_ops:
            if isinstance(qobj, qtp.Qobj):
                new_e_ops.append(qobj.full())
            elif isinstance(qobj, csr_array):
                new_e_ops.append(qobj.todense())
            elif isinstance(qobj, tuple):
                qudit_obs = list(qtp.Qobj(obs) for obs in qobj)
                new_e_ops.append(qtp.tensor(qudit_obs).full())
            elif isinstance(qobj, dict):
                qudit_obs = []
                for qid in hgen.qudit_ids():
                    try:
                        qudit_obs.append(qtp.Qobj(qobj[qid]))
                    except KeyError:
                        qudit_obs.append(qtp.qeye(hgen.qudit_params(qid).num_levels))

                new_e_ops.append(qtp.tensor(qudit_obs).full())
            else:
                new_e_ops.append(qobj)

        e_ops = new_e_ops

    if not isinstance(frame, SystemFrame):
        frame = SystemFrame(frame, hgen)

    if solver_options is None:
        if solver == 'jax':
            solver_options = {'rtol': 1.e-8, 'atol': 1.e-8, 'mxstep': jnp.inf, 'hmax': jnp.inf,
                              'num_parallel': 64}
        else:
            solver_options = {'rtol': 1.e-6, 'atol': 1.e-8}

    return PulseSimParameters(frame, tlist, psi0, e_ops, final_only, reunitarize, solver_options)


def _run_single(
    hamiltonian: Any,
    parameters: PulseSimParameters,
    drive_args: Optional[dict[str, Any]] = None,
    solver: str = 'str',
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
) -> PulseSimResult:
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    logger.info('Using %d time points from %.3e to %.3e', parameters.tlist.shape[0],
                parameters.tlist[0], parameters.tlist[-1])

    # Run the simulation or exponentiate the static Hamiltonian
    sim_start = time.time()
    if (isinstance(hamiltonian, list) and len(hamiltonian) == 1
            and isinstance(hamiltonian[0], qtp.Qobj)):
        evolution = exponentiate_hstat(hamiltonian[0], parameters, logger_name)
    elif solver == 'jax':
        evolution = simulate_drive_odeint(hamiltonian, parameters, drive_args, logger_name)
    else:
        evolution = simulate_drive_sesolve(hamiltonian, parameters, drive_args, logger_name)

    # Compute the states and expectation values in the original frame
    transform_start = time.time()
    states, expect = transform_evolution(evolution, parameters)

    logger.debug('Calculation of states and expectation values completed in %.2f seconds.',
                 time.time() - transform_start)

    result = PulseSimResult(times=parameters.tlist, expect=expect, states=states,
                            frame=parameters.frame)

    logger.info('Done in %.2f seconds.', time.time() - sim_start)

    if save_result_to:
        logger.info('Saving the simulation result to %s.h5', save_result_to)
        save_sim_result(f'{save_result_to}.h5', result)

    return result


def simulate_drive_sesolve(
    hamiltonian: list[Union[qtp.Qobj, list]],
    parameters: PulseSimParameters,
    drive_args: Optional[dict[str, Any]] = None,
    logger_name: str = __name__
):
    """Integrate the time evolution using QuTiP sesolve."""
    logger = logging.getLogger(logger_name)

    # Convert functional coefficients to arrays
    if isinstance(hamiltonian, Hamiltonian):
        hamiltonian = hamiltonian.evaluate_coeffs(parameters.tlist, drive_args)

    if len(hamiltonian) == 1 and isinstance(hamiltonian[0], qtp.Qobj):
        return exponentiate_hstat(hamiltonian[0], parameters, logger_name)

    logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

    # Run sesolve in a temporary directory
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            os.chdir(tempdir)
            evolution = _run_sesolve(hamiltonian, parameters, logger)
        finally:
            os.chdir(cwd)

    return evolution


def _run_sesolve(hamiltonian, parameters, logger):
    # Create a QuTiP solver options object from the options dict
    solver_options = qtp.solver.Options()
    for attr in dir(solver_options):
        try:
            setattr(solver_options, attr, parameters.solver_options[attr])
        except KeyError:
            pass

    # sesolve generates O(1e-6) error in the global phase when the Hamiltonian is not traceless.
    # Since we run the simulation in the lab frame, the only traceful term is hdiag.
    # We therefore subtract the diagonals to make hdiag traceless, and later apply a global phase
    # according to the shift.
    hdiag = hamiltonian[0]
    global_phase = np.trace(hdiag.full()).real / hdiag.shape[0]
    hamiltonian[0] = hdiag - global_phase

    # If all coefficients in the Hamiltonian are constants, or string expressions, or functions,
    # we can reuse the compiled objects
    solver_options.rhs_reuse = all(not (isinstance(term, list) and isinstance(term[1], np.ndarray))
                                   for term in hamiltonian)

    # We'll always need the states
    solver_options.store_states = True

    # Simulation interval lengths for reunitarization (number of time points)
    if parameters.reunitarize:
        interval_len = parameters.solver_options.get('interval_len', 512)
    else:
        interval_len = parameters.tlist.shape[0]

    num_intervals = int(np.ceil((parameters.tlist.shape[0] - 1) / (interval_len - 1)))

    logger.debug('Integrating time evolution in %d intervals of %d steps.',
                 num_intervals, interval_len)

    # Initial unitary is the identity
    initial = np.eye(hdiag.shape[-1], dtype=complex)
    evolution_arrays = []

    for interval in range(num_intervals):
        interval_sim_start = time.time()

        start = interval * (interval_len - 1)
        end = start + interval_len

        local_hamiltonian = list()
        for h_term in hamiltonian:
            if isinstance(h_term, list) and isinstance(h_term[1], np.ndarray):
                local_hamiltonian.append([h_term[0], h_term[1][start:end]])
            else:
                local_hamiltonian.append(h_term)

        psi0 = qtp.Qobj(inpt=initial, dims=hdiag.dims)

        result = qtp.sesolve(local_hamiltonian, psi0, parameters.tlist[start:end],
                             options=solver_options)

        # Array of lab-frame evolution ops
        if parameters.final_only:
            evolution = result.states[-1].full()[None, ...]
        else:
            evolution = np.stack(list(state.full() for state in result.states))

        interval_sim_end = time.time()
        logger.debug('Integration of interval %d completed in %.2f seconds.',
                     interval, interval_sim_end - interval_sim_start)

        # Apply reunitarization
        if parameters.reunitarize:
            evolution = closest_unitary(evolution)
            logger.debug('Reunitarization of interval %d completed in %.2f seconds.',
                         interval, time.time() - interval_sim_end)

        if parameters.final_only:
            evolution_arrays = [evolution]
        else:
            evolution_arrays.append(evolution)

        initial = evolution[-1]

    # Restore the actual global phase
    evolution = np.concatenate(evolution_arrays)
    evolution *= np.exp(-1.j * global_phase * parameters.tlist[-evolution.shape[0]:, None, None])

    return evolution


def simulate_drive_odeint(
    hamiltonian: Callable,
    parameters: PulseSimParameters,
    drive_args: Optional[dict[str, Any]] = None,
    logger_name: str = __name__
):
    """Use JAX odeint to integrate the time evolution operator."""
    logger = logging.getLogger(logger_name)

    # parallel.parallel_map places the target function under jax.default_device()
    # pylint: disable-next=no-member
    logger.info('Starting simulation on JAX device %s.', jax.config.jax_default_device)

    tlists, tlist_indices = _stack_tlist(parameters.tlist,
                                         parameters.solver_options.get('num_parallel', 64))

    logger.debug('Shapes of tlists: original %s, stacked %s', parameters.tlist.shape, tlists.shape)

    compile_start = time.time()

    if drive_args is None:
        drive_args = {}

    state_dim = np.prod(parameters.frame.dim)
    integrator = _compile_vodeint(hamiltonian, parameters.solver_options, tlists.shape, state_dim,
                                  drive_args)

    compile_end = time.time()
    logger.debug('Compilation of integrator completed in %.2f seconds.',
                 compile_end - compile_start)

    # Run the evolution on a given device
    y0 = jnp.eye(state_dim, dtype=complex)
    tlists = jnp.array(tlists)

    evolution = integrator(y0, tlists, drive_args)
    # Pick the time points included in the original tlist
    evolution = evolution[tlist_indices]

    if parameters.final_only:
        evolution = evolution[-1:]

    sim_end = time.time()
    logger.debug('Integration completed in %.2f seconds.', sim_end - compile_end)

    if parameters.reunitarize:
        evolution = closest_unitary(evolution, npmod=jnp)
        logger.debug('Reunitarization completed in %.2f seconds.', time.time() - sim_end)

    return evolution


def _stack_tlist(tlist: np.ndarray, num_parallel: int):
    """Make an array of stacked tlists."""
    tlist_indices = np.arange(tlist.shape[0])
    # Add intermediate points until there are more tlist points than the vmap width
    while tlist.shape[0] - 1 < num_parallel:
        dt = np.diff(tlist)
        if not np.allclose(dt, dt[0]):
            raise ValueError('For JAX solver, tlist must be a uniformly increasing array')

        tlist_low = tlist[0] + np.cumsum(dt) * 0.5
        tlist_high = tlist_low[-1] + np.cumsum(dt) * 0.5
        tlist = np.concatenate([tlist[:1], tlist_low, tlist_high])
        tlist_indices *= 2

    # Number of time points in each thread
    interval_len = int(np.ceil((tlist.shape[0] - 1) / num_parallel)) + 1

    # Recalculate num_parallel: when ceil((T-1)/(N-k)) = ceil((T-1)/N), we should use N-k threads
    num_parallel = int(np.ceil((tlist.shape[0] - 1) / (interval_len - 1)))

    # The last thread may have extra points to simulate
    num_extra_points = (interval_len - 1) * num_parallel - (tlist.shape[0] - 1)
    if num_extra_points > 0:
        dt = tlist[-1] - tlist[-2]
        new_end = tlist[-1] + dt * num_extra_points
        tlist = np.concatenate([tlist[:-1], np.linspace(tlist[-1], new_end, num_extra_points + 1)])

    # Stacked tlists
    starts = np.arange(num_parallel) * (interval_len - 1)
    tlists = np.array(list(tlist[start:start + interval_len] for start in starts))

    return tlists, tlist_indices


odeint_cv = Condition()


def _compile_vodeint(
    hamiltonian: Callable,
    solver_options: dict[str, Any],
    tshape: tuple[int, int],
    ydim: int,
    drive_args: dict[str, Any],
    _compile: bool = True
):
    # vodeint must be lowered on the specific device - key on arg shape + device ID
    opt = (solver_options.get('rtol', 1.e-8), solver_options.get('atol', 1.e-8),
           solver_options.get('mxstep', jnp.inf), solver_options.get('hmax', jnp.inf))
    # pylint: disable-next=no-member
    h_key = (tshape, opt, jax.config.jax_default_device.id)

    with odeint_cv:
        if not hasattr(hamiltonian, 'compiled_vodeint'):
            hamiltonian.compiled_vodeint = {}

        try:
            integrator = hamiltonian.compiled_vodeint[h_key]
        except KeyError:
            hamiltonian.compiled_vodeint[h_key] = None
            odeint_cv.release()

            tlists = jnp.empty(tshape, dtype=float)
            y0 = jnp.empty((ydim, ydim), dtype=complex)
            converted, _ = jax.custom_derivatives.closure_convert(hamiltonian, y0, 0.,
                                                                  drive_args)
            lowered = vodeint.lower(converted, y0, tlists, drive_args, opt)

            if _compile:
                integrator = lowered.compile()
            else:
                integrator = lowered

            odeint_cv.acquire()
            hamiltonian.compiled_vodeint[h_key] = integrator
            odeint_cv.notify_all()
        else:
            if integrator is None:
                odeint_cv.wait_for(lambda: hamiltonian.compiled_vodeint[h_key] is not None)
                integrator = hamiltonian.compiled_vodeint[h_key]

    return integrator


@partial(jax.jit, static_argnums=(0, 4))
def vodeint(hamiltonian, y0, tlists, drive_args, opt):
    vwrapper = jax.vmap(_odeint_wrapper, in_axes=(None, None, None, None, None, None, 0, None))

    stacked_u = vwrapper(hamiltonian, opt[0], opt[1], opt[2], opt[3], y0, tlists, drive_args)

    def dot_last(ip, matrices):
        return matrices.at[ip].set(stacked_u[ip - 1, -1] @ matrices[ip - 1])

    num_pieces = stacked_u.shape[0]
    init = jnp.repeat(jnp.eye(stacked_u.shape[-1], dtype=complex)[None, :, :], num_pieces, axis=0)
    init = jax.lax.fori_loop(1, num_pieces, dot_last, init)

    matrix_shape = stacked_u.shape[-2:]
    evolution = jnp.einsum('ptij,pjk->ptik', stacked_u[:, :-1], init)
    evolution = jnp.concatenate([evolution.reshape((-1,) + matrix_shape),
                                 jnp.matmul(stacked_u[-1, -1], init[-1])[None, ...]], axis=0)

    return evolution


def exponentiate_hstat(
    hamiltonian: qtp.Qobj,
    parameters: PulseSimParameters,
    logger_name: str = __name__
):
    """Compute exp(-iHt)."""
    logger = logging.getLogger(logger_name)

    logger.info('Exponentiating hamiltonian with dimension %s.', hamiltonian.dims)

    hamiltonian = hamiltonian.full()

    if parameters.final_only:
        tlist = parameters.tlist[-1:]
    else:
        tlist = parameters.tlist

    sim_start = time.time()

    evolution = matrix_exp(-1.j * hamiltonian[None, ...] * tlist[:, None, None], hermitian=-1)

    sim_end = time.time()
    logger.debug('Exponentiation completed in %.2f seconds.', sim_end - sim_start)

    return evolution


def transform_evolution(
    evolution: ArrayType,
    parameters: PulseSimParameters
):
    """Compute the evolution operator and expectation values in the given frame."""
    if isinstance(evolution, jax.Array):
        npmod = jnp
    else:
        npmod = np

    if parameters.final_only:
        tlist = parameters.tlist[-1:]
    else:
        tlist = parameters.tlist

    # Change the frame back to original
    lab_frame = SystemFrame({qid: QuditFrame(np.zeros_like(qf.frequency), np.zeros_like(qf.phase))
                            for qid, qf in parameters.frame.items()})

    evolution = parameters.frame.change_frame(tlist, evolution, from_frame=lab_frame,
                                              objtype='evolution', t0=parameters.tlist[0],
                                              npmod=npmod)

    # State evolution in the original frame
    if parameters.psi0 is None:
        states = evolution
    else:
        states = evolution @ npmod.asarray(parameters.psi0)

    if not parameters.e_ops:
        return np.asarray(states), None

    # Fill the expectation values
    expect = list()
    for observable in parameters.e_ops:
        if len(states.shape) == 3:
            out = npmod.einsum('ij,tij->t', observable.conjugate(), states)
        else:
            out = npmod.einsum('ti,ij,tj->t', states.conjugate(), observable, states).real

        expect.append(np.asarray(out))

    return np.asarray(states), expect
