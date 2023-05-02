"""Pulse simulation frontend."""

import copy
import logging
import os
import tempfile
import time
from concurrent import futures
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from jax.experimental.ode import _odeint_wrapper
import numpy as np
from scipy.sparse import csr_array
from qutip import Qobj, sesolve, rhs_clear
from qutip.solver import Options

from rqutils.math import matrix_exp
from .config import config
from .frame import FrameSpec, QuditFrame, SystemFrame
from .hamiltonian import Hamiltonian, HamiltonianBuilder
from .parallel import parallel_map
from .sim_result import PulseSimResult, save_sim_result
from .unitary import closest_unitary

logger = logging.getLogger(__name__)

TList = Union[np.ndarray, Tuple[int, int], Dict[str, Union[int, float]]]
QObject = Union[qtp.Qobj, np.ndarray, csr_array]
SolverOptions = Union[Options, Tuple[float, float, float, float]]


def pulse_sim(
    hgen: HamiltonianBuilder,
    tlist: Union[TList, List[TList]] = (10, 100),
    psi0: Optional[Union[QObject, List[QObject]]] = None,
    drive_args: Union[Dict[str, Any], List[Dict[str, Any]]] = {},
    e_ops: Optional[Union[Sequence[QObject], List[Sequence[QObject]]]] = None,
    frame: Union[FrameSpec, List[FrameSpec]] = 'dressed',
    final_only: Union[bool, List[bool]] = False,
    reunitarize: Union[bool, List[bool]] = True,
    interval_len: Union[int, str, List[Union[int, str]]] = 'auto',
    solver: str = 'qutip',
    solver_options: Optional[Union[SolverOptions, List[SolverOptions]]] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union['PulseSimResult', List['PulseSimResult']]:
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
    individual single simulation jobs. Parallel jobs are run as separate processes if QuTiP (internally scipy)-based integrator is used, and as threads if JAX integrator is used.

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
    ``sesolve`` is called in time intervals of length ``interval_len``, and the entire array of
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
            will be used.
        psi0: Initial state Qobj or array. Defaults to the identity operator appropriate for the
            given Hamiltonian.
        drive_args: Second parameter passed to drive amplitude functions (if callable).
        e_ops: List of observables passed to the QuTiP solver.
        frame: System frame to represent the simulation results in.
        final_only: Throw away intermediate states and expectation values to save memory. Due to
            implementation details, implies ``reunitarize=True``.
        reunitarize: Whether to reunitarize the time evolution matrices. Note that SVD can be very
            slow for large system sizes and ``final_only=False``.
        interval_len: When integrating with QuTiP ``sesolve``, the number of time steps to run
            ``sesolve`` uninterrupted. In this case the parameter is ignored unless
            ``reunitarize=True`` and ``psi0`` is a unitary. When integrating with JAX ``odeint``,
            this parameter determines the size of tlist batches. If set to ``'auto'``, an
            appropriate value is automatically assigned. If the value is less than 2, simulation
            will be run in a single batch uninterrupted.
        solver_options: If solver is qutip, QuTiP solver options. If solver is jax, a tuple
            representing ``(rtol, atol, mxstep, hmax)`` parameters of ``odeint``.
        save_result_to: File name (without the extension) to save the simulation result to.
        log_level: Log level.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    num_tasks = None

    if isinstance(drive_args, list):
        num_tasks = len(drive_args)

    # Put inputs to PulseSimParameters in a zipped list
    zip_list = list()

    parallel_params = {
        'tlist': (tlist, (np.ndarray, tuple, dict)),
        'psi0': (psi0, (Qobj, np.ndarray, csr_array)),
        'e_ops': (e_ops, Sequence),
        'frame': (frame, (str, dict, Sequence)),
        'final_only': (final_only, bool),
        'reunitarize': (reunitarize, bool),
        'interval_len': (interval_len, (int, str)),
        'solver_options': (solver_options, (Options, tuple))
    }

    # Pack parallelizable arguments into per-job dicts
    singleton_params = list()
    for key, (value, typ) in parallel_params.items():
        if isinstance(value, list) and (typ is None or all(isinstance(elem, typ) for elem in value)):
            if num_tasks is None:
                num_tasks = len(value)
            elif num_tasks != len(value):
                raise ValueError('Lists with inconsistent lengths passed as arguments')
        else:
            singleton_params.append(key)

    if num_tasks is None:
        hamiltonian = build_hamiltonian(hgen, solver, drive_args)

        raw_args = {key: value for key, (value, typ) in parallel_params.items()}
        parameters = compose_parameters(hgen, solver=solver, **raw_args)

        result = _run_single(hamiltonian, parameters, drive_args=drive_args, solver=solver,
                             save_result_to=save_result_to)

    else:
        raw_args_list = list(dict() for _ in range(num_tasks))
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

            save_result_path = lambda itask: os.path.join(save_result_to, f'%0{num_digits}d' % itask)
        else:
            save_result_path = lambda itask: None

        hamiltonian = build_hamiltonian(hgen, solver, drive_args[0])

        parameters_list = list(compose_parameters(hgen, solver=solver, **raw_args)
                               for raw_args in raw_args_list)

        if solver == 'jax':
            thread_based = True

            # Precompile the hamiltonian function for all unique tlist-stack shapes
            tlist_shape_params = set((p.tlist.shape[0], p.interval_len, p.solver_options)
                                     for p in parameters_list)

            y0 = hgen.identity_op().full()

            # Compile in threads
            with futures.ThreadPoolExecutor() as pool:
                thread_jobs = dict()
                for tlist_len, interval_len, solver_options in tlist_shape_params:
                    num_intervals = int(np.ceil((tlist_len - 1) / (interval_len - 1)))

                    dummy_tlists = np.empty((num_intervals, interval_len))

                    lowered = vodeint.lower(hamiltonian, y0, dummy_tlists, drive_args[0], solver_options)
                    thread_jobs[(dummy_tlists.shape, solver_options)] = pool.submit(lowered.compile)

            hamiltonian.compiled_vodeint = {key: job.result() for key, job in thread_jobs.items()}

            common_args = (hamiltonian,)
            args = parameters_list
            arg_position = 1
            kwarg_keys = ('drive_args', 'logger_name', 'save_result_to')
            kwarg_values = list((drive_args[itask], f'{__name__}.{itask}', save_result_path(itask))
                                for itask in range(num_tasks))

        else:
            thread_based = False

            # Convert the functional Hamiltonian coefficients to arrays before passing them to parallel_map
            hamiltonians = list()
            for params, dargs in zip(parameters_list, drive_args):
                hamiltonians.append(hamiltonian.evaluate_coeffs(params.tlist, dargs, npmod=np))

            common_args = None
            args = list(zip(hamiltonians, parameters_list))
            arg_position = (0, 1)
            kwarg_keys = ('logger_name', 'save_result_to')
            kwarg_values = list((f'{__name__}.{itask}', save_result_path(itask))
                                for itask in range(num_tasks))

        result = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys, kwarg_values=kwarg_values,
                              arg_position=arg_position, common_args=common_args, common_kwargs={'solver': solver},
                              log_level=log_level, thread_based=thread_based)

    logger.setLevel(original_log_level)

    return result


@dataclass
class PulseSimParameters:
    frame: SystemFrame
    tlist: np.ndarray
    psi0: np.ndarray
    e_ops: Union[List[np.ndarray], None] = None
    final_only: bool = False
    reunitarize: bool = True
    interval_len: int = 1
    solver_options: SolverOptions = Options()

    def transform_evolution(
        self,
        evolution: np.ndarray,
        tlist: Optional[np.ndarray] = None
    ):
        if tlist is None:
            tlist = self.tlist

        # Change the frame back to original
        lab_frame = SystemFrame({qid: QuditFrame(np.zeros_like(qf.frequency), np.zeros_like(qf.phase))
                                for qid, qf in self.frame.items()})

        evolution = self.frame.change_frame(tlist, evolution, from_frame=lab_frame,
                                            objtype='evolution', t0=self.tlist[0])

        # State evolution in the original frame
        states = evolution @ self.psi0

        if not self.e_ops:
            return states, None

        # Fill the expectation values
        expect = list()
        for observable in self.e_ops:
            if len(states.shape) == 3:
                out = np.einsum('ij,tij->t', observable.conjugate(), states)
            else:
                out = np.einsum('ti,ij,tj->t', states.conjugate(), observable, states).real

            expect.append(out)

        return states, expect


def build_hamiltonian(
    hgen: HamiltonianBuilder,
    solver: str = 'qutip',
    sample_drive_args: Optional[Dict[str, Any]] = None
) -> Any:
    """Build the Hamiltonian according to the pulse_sim_solver option.

    Args:
        hgen: A HamiltonianBuilder instance.
        solver: ``'qutip'`` or ``'jax'``
        sample_drive_args: For ``solver='jax'``, an example argument dictionary (with valid keys and
            value types) to be used for compiling the pulse functions.
    """
    if solver == 'jax':
        h_terms = hgen.build(frame='lab', as_timefn=True)

        if len(h_terms) == 1 and isinstance(h_terms[0], Qobj):
            # Will just exponentiate
            hamiltonian = h_terms
        else:
            @jax.jit
            def du_dt(u_t, t, drive_args):
                h_t = h_terms[0].full()
                for qobj, fn in h_terms[1:]:
                    h_t += qobj.full() * fn(t, drive_args, npmod=jnp)

                return -1.j * jnp.matmul(h_t, u_t)

            if sample_drive_args is None:
                hamiltonian = du_dt
            else:
                hamiltonian, _ = jax.custom_derivatives.closure_convert(du_dt, hgen.identity_op().full(),
                                                                        0., sample_drive_args)

    else:
        hamiltonian = hgen.build(frame='lab')

    return hamiltonian


def compose_parameters(
    hgen: HamiltonianBuilder,
    tlist: TList = (10, 100),
    psi0: Optional[QObject] = None,
    e_ops: Optional[Sequence[QObject]] = None,
    frame: FrameSpec = 'dressed',
    final_only: bool = False,
    reunitarize: bool = True,
    interval_len: Union[int, str] = 'auto',
    solver: str = 'qutip',
    solver_options: Optional[SolverOptions] = None
) -> PulseSimParameters:
    """Generate the arguments for parallel jobs using the HamiltonianBuilder."""
    ## Time points
    if isinstance(tlist, tuple):
        tlist = {'points_per_cycle': tlist[0], 'num_cycles': tlist[1]}
    if isinstance(tlist, dict):
        tlist = hgen.make_tlist(frame='lab', **tlist)

    if psi0 is None:
        psi0 = hgen.identity_op().full()
    elif isinstance(psi0, Qobj):
        psi0 = np.squeeze(psi0.full())
    elif isinstance(psi0, csr_array):
        psi0 = psi0.todense()

    if e_ops is not None:
        new_e_ops = []
        for qobj in e_ops:
            if isinstance(qobj, Qobj):
                new_e_ops.append(qobj.full())
            elif isinstance(qobj, csr_array):
                new_e_ops.append(qobj.todense())
            else:
                new_e_ops.append(qobj)

        e_ops = new_e_ops

    if not isinstance(frame, SystemFrame):
        frame = SystemFrame(frame, hgen)

    if interval_len == 'auto':
        # Best determined by the representative frequency and simulation duration
        if solver == 'jax':
            interval_len = 101
        else:
            if reunitarize:
                interval_len = 501
            else:
                interval_len = tlist.shape[0]

    if interval_len <= 1:
        interval_len = tlist.shape[0]

    if solver_options is None:
        if solver == 'jax':
            solver_options = (1.e-8, 1.e-8, jnp.inf, jnp.inf)
        else:
            solver_options = Options()

    return PulseSimParameters(frame, tlist, psi0, e_ops, final_only, reunitarize, interval_len,
                              solver_options=solver_options)


def _run_single(
    hamiltonian: Any,
    parameters: PulseSimParameters,
    drive_args: Optional[Dict[str, Any]] = None,
    solver: str = 'str',
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
) -> PulseSimResult:
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    logger.info('Using %d time points from %.3e to %.3e', parameters.tlist.shape[0],
                parameters.tlist[0], parameters.tlist[-1])

    ## Run the simulation or exponentiate the static Hamiltonian
    if isinstance(hamiltonian, list) and len(hamiltonian) == 1 and isinstance(hamiltonian[0], Qobj):
        logger.info('Static Hamiltonian built. Exponentiating..')

        states, expect = exponentiate_hstat(hamiltonian[0], parameters, logger_name)

    elif solver == 'jax':
        logger.info('XLA-compiled Hamiltonian built. Starting simulation..')

        states, expect = simulate_drive_odeint(hamiltonian, parameters, drive_args, logger_name)

    else:
        logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

        states, expect = simulate_drive_sesolve(hamiltonian, parameters, drive_args, logger_name)

    ## Compile the result and return
    if parameters.final_only:
        tlist = parameters.tlist[-1:]
    else:
        tlist = parameters.tlist

    result = PulseSimResult(times=tlist, expect=expect, states=states, frame=parameters.frame)

    if save_result_to:
        logger.info('Saving the simulation result to %s.h5', save_result_to)
        save_sim_result(f'{save_result_to}.h5', result)

    return result


def simulate_drive_sesolve(
    hamiltonian: List[Union[Qobj, List]],
    parameters: PulseSimParameters,
    drive_args: Optional[Dict[str, Any]] = None,
    logger_name: str = __name__
):
    logger = logging.getLogger(logger_name)

    # Convert functional coefficients to arrays
    if isinstance(hamiltonian, Hamiltonian):
        hamiltonian = hamiltonian.evaluate_coeffs(parameters.tlist, drive_args)

    if len(hamiltonian) == 1 and isinstance(hamiltonian[0], Qobj):
        return exponentiate_hstat(hamiltonian[0], parameters, logger_name)

    # Take a copy of options because we change its values

    solver_options = copy.deepcopy(parameters.solver_options)

    # Reunitarization and interval running setting

    reunitarize = parameters.reunitarize
    if parameters.final_only and parameters.interval_len < parameters.tlist.shape[0]:
        # Final only option with a short interval len implies that we run sesolve in intervals.
        # The evolution matrices must be reunitarized in this case because an op psi0 argument
        # of sesolve must be unitary, but the evolution matrices diverge from unitary typically
        # within a few hundred time steps.
        reunitarize = True

    # sesolve generates O(1e-6) error in the global phase when the Hamiltonian is not traceless.
    # Since we run the simulation in the lab frame, the only traceful term is hdiag.
    # We therefore subtract the diagonals to make hdiag traceless, and later apply a global phase
    # according to the shift.
    hdiag = hamiltonian[0]
    global_phase = np.trace(hdiag.full()).real / hdiag.shape[0]
    hamiltonian[0] = hdiag - global_phase

    isarray = list(isinstance(term, list) and isinstance(term[1], np.ndarray)
                   for term in hamiltonian)
    if not any(isarray):
        # No interval dependency in the Hamiltonian (all constants or string expressions)
        # -> Can reuse the compiled objects
        solver_options.rhs_reuse = True

    # Arrays to store the unitaries and expectation values

    if not parameters.e_ops or solver_options.store_states:
        # squeezed shape
        state_shape = parameters.psi0.shape
        if parameters.final_only:
            states = np.empty((1,) + state_shape, dtype=np.complex128)
        else:
            states = np.empty(parameters.tlist.shape + state_shape, dtype=np.complex128)
    else:
        states = None

    if parameters.e_ops:
        num_e_ops = len(parameters.e_ops)
        if parameters.final_only:
            expect = list(np.empty(1) for _ in range(num_e_ops))
        else:
            expect = list(np.empty_like(parameters.tlist) for _ in range(num_e_ops))
    else:
        expect = None

    ## Run sesolve in a temporary directory

    # We'll always need the states
    solver_options.store_states = True

    sim_start = time.time()

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            os.chdir(tempdir)

            num_intervals = int(np.ceil((parameters.tlist.shape[0] - 1) / (parameters.interval_len - 1)))

            # Compute the time evolution in the lab frame -> initial unitary is identity

            last_unitary = np.eye(np.prod(hdiag.dims[0]))

            for interval in range(num_intervals):
                psi0 = Qobj(inpt=last_unitary, dims=hdiag.dims)

                start = interval * (parameters.interval_len - 1)
                end = start + parameters.interval_len

                local_hamiltonian = list()
                for h_term, isa in zip(hamiltonian, isarray):
                    if isa:
                        local_hamiltonian.append([h_term[0], h_term[1][start:end]])
                    else:
                        local_hamiltonian.append(h_term)

                tlist = parameters.tlist[start:end]

                qtp_result = sesolve(local_hamiltonian, psi0, tlist, options=solver_options)

                # Array of lab-frame evolution ops
                if parameters.final_only:
                    evolution = np.squeeze(qtp_result.states[-1].full())[None, ...]
                    tlist = tlist[-1:]
                else:
                    evolution = np.stack(list(np.squeeze(state.full()) for state in qtp_result.states))

                # Apply reunitarization
                if reunitarize:
                    evolution = closest_unitary(evolution)

                # evolution[-1] is the lab-frame evolution operator
                last_unitary = evolution[-1].copy()

                # Restore the actual global phase
                evolution *= np.exp(-1.j * global_phase * tlist[:, None, None])

                # Compute the states and expectation values in the original frame
                local_states, local_expect = parameters.transform_evolution(evolution, tlist)

                # Indices for the output arrays
                if parameters.final_only:
                    out_slice = slice(0, 1)
                else:
                    out_slice = slice(start, end)

                # Fill the states
                if states is not None:
                    states[out_slice] = local_states

                # Fill the expectation values
                if expect is not None:
                    for out, local in zip(expect, local_expect):
                        out[out_slice] = local

                logger.debug('Processed interval %d/%d. Cumulative simulation time %f seconds.',
                             interval, num_intervals, time.time() - sim_start)

        finally:
            os.chdir(cwd)
            # Clear the cached solver
            rhs_clear()

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect


odeint_lock = Lock()

def simulate_drive_odeint(
    hamiltonian: Callable,
    parameters: PulseSimParameters,
    drive_args: Dict[str, Any] = dict(),
    logger_name: str = __name__
):
    logger = logging.getLogger(logger_name)

    tlist = parameters.tlist

    interval_len = parameters.interval_len

    num_intervals = int(np.ceil((tlist.shape[0] - 1) / (interval_len - 1)))

    residual = (tlist.shape[0] - 1) % (interval_len - 1)
    if residual != 0:
        num_extra_points = interval_len - residual
        dt = tlist[-1] - tlist[-2]
        new_end = tlist[-1] + dt * num_extra_points
        tlist = np.concatenate([tlist[:-1], np.linspace(tlist[-1], new_end, num_extra_points + 1)])

    starts = np.arange(num_intervals) * (interval_len - 1)
    tlists = np.array(list(tlist[start:start + interval_len] for start in starts))

    sim_start = time.time()

    y0 = np.eye(parameters.psi0.shape[0])

    if hamiltonian.__name__ == 'du_dt':
        hamiltonian, _ = jax.custom_derivatives.closure_convert(hamiltonian, y0, 0., drive_args)

    h_key = (tlists.shape, parameters.solver_options)

    with odeint_lock:
        if not hasattr(hamiltonian, 'compiled_vodeint'):
            hamiltonian.compiled_vodeint = dict()

        try:
            integrator = hamiltonian.compiled_vodeint[h_key]
        except KeyError:
            integrator = vodeint.lower(hamiltonian, y0, tlists, drive_args, parameters.solver_options).compile()
            hamiltonian.compiled_vodeint[h_key] = integrator

    ## Run the evolution on a given device
    # parallel.parallel_map sets jax_devices[0] to the ID of the GPU to be used in this thread
    jax_device = jax.devices()[config.jax_devices[0]]

    y0 = jax.device_put(y0, device=jax_device)
    tlists = jax.device_put(tlists, device=jax_device)

    evolution = np.asarray(integrator(y0, tlists, drive_args))

    if tlist.shape[0] - parameters.tlist.shape[0] != 0:
        # Truncate back to the original tlist
        evolution = evolution[:parameters.tlist.shape[0]]
        tlist = tlist[:parameters.tlist.shape[0]]

    if parameters.final_only:
        evolution = evolution[-1:]
        tlist = tlist[-1:]

    # Apply reunitarization
    if parameters.reunitarize:
        # Note: Downloading the evolution matrices and using the numpy SVD because that seems faster
        evolution = closest_unitary(evolution, npmod=np)

    ## Restore the actual global phase
    #evolution *= np.exp(-1.j * global_phase * local_tlist[:, None, None])

    # Compute the states and expectation values in the original frame
    states, expect = parameters.transform_evolution(evolution, tlist)

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect

@partial(jax.jit, static_argnums=(0, 4))
def vodeint(hamiltonian, y0, tlists, drive_args, opt):
    vwrapper = jax.vmap(_odeint_wrapper, in_axes=(None, None, None, None, None, None, 0, None))

    stacked_u = vwrapper(hamiltonian, opt[0], opt[1], opt[2], opt[3], y0, tlists, drive_args)

    dot_last = lambda ip, matrices: matrices.at[ip].set(stacked_u[ip - 1, -1] @ matrices[ip - 1])

    num_pieces = stacked_u.shape[0]
    init = jnp.repeat(jnp.eye(stacked_u.shape[-1], dtype=complex)[None, :, :], num_pieces, axis=0)
    init = jax.lax.fori_loop(1, num_pieces, dot_last, init)

    matrix_shape = stacked_u.shape[-2:]
    evolution = jnp.einsum('ptij,pjk->ptik', stacked_u[:, :-1], init)
    evolution = jnp.concatenate([evolution.reshape((-1,) + matrix_shape),
                                 jnp.matmul(stacked_u[-1, -1], init[-1])[None, ...]], axis=0)

    return evolution


def exponentiate_hstat(
    hamiltonian: Qobj,
    parameters: PulseSimParameters,
    logger_name: str = __name__
):
    """Compute exp(-iHt)."""
    logger = logging.getLogger(__name__)

    hamiltonian = hamiltonian.full()

    if parameters.final_only:
        tlist = parameters.tlist[-1:]
    else:
        tlist = parameters.tlist

    evolution = matrix_exp(-1.j * hamiltonian[None, ...] * tlist[:, None, None], hermitian=-1)

    states, expect = parameters.transform_evolution(evolution, tlist)

    return states, expect
