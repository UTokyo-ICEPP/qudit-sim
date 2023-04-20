"""Pulse simulation frontend."""

from typing import Any, Dict, List, Tuple, Sequence, Iterable, Optional, Union, Callable
import os
import tempfile
import logging
import time
import copy
from dataclasses import dataclass

import numpy as np
from qutip import Qobj, qeye, sesolve, rhs_clear
from qutip.solver import Options
import jax
import jax.numpy as jnp
from jax.experimental.ode import _odeint_wrapper

from rqutils.math import matrix_exp

from .hamiltonian import HamiltonianBuilder
from .frame import FrameSpec, SystemFrame, QuditFrame
from .sim_result import PulseSimResult, save_sim_result
from .unitary import closest_unitary
from .parallel import parallel_map
from .config import config

logger = logging.getLogger(__name__)

TList = Union[np.ndarray, Tuple[int, int], Dict[str, Union[int, float]]]

def pulse_sim(
    hgen: HamiltonianBuilder,
    tlist: Union[TList, List[TList]] = (10, 100),
    psi0: Optional[Union[Qobj, List[Qobj]]] = None,
    drive_args: Union[Dict[str, Any], List[Dict[str, Any]]] = dict(),
    e_ops: Optional[Union[Sequence[Qobj], List[Sequence[Qobj]]]] = None,
    frame: Union[FrameSpec, List[FrameSpec]] = 'dressed',
    final_only: Union[bool, List[bool]] = False,
    reunitarize: Union[bool, List[bool]] = True,
    interval_len: Union[int, str, List[Union[int, str]]] = 'auto',
    sesolve_options: Union[Options, List[Options]] = Options(),
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union['PulseSimResult', List['PulseSimResult']]:
    r"""Run a pulse simulation.

    Build the Hamiltonian terms from the HamiltonianBuilder, determine the time points for the simulation if necessary,
    and run ``qutip.sesolve``.

    Regardless of the setup of the input (initial state and expectation ops), the simulation is run in
    the lab frame with the identity operator as the initial state to obtain the time evolution operator for each time point.
    The evolution operators are then transformed into the specified frame, if any, or the default frame of the
    HamiltonianBuilder, before being applied to
    the initial state (or initial unitary). If expectation ops are given, their expectation values are computed from the
    resulting evolved states.

    All parameters except ``hgen``, ``save_result_to``, and ``log_level`` can be given as lists to trigger a
    parallel execution of ``sesolve``. If more than one parameter is a list, their lengths must be identical, and all
    parameters are "zipped" together to form the argument lists of individual single simulation jobs. Parallel jobs
    are run as separate processes if QuTiP (internally scipy)-based integrator is used, and as threads if JAX integrator
    is used.

    .. rubric:: Reunitarization of time evolution

    Due to the finite numerical precision in solving the Schrodinger equation, the computed time evolution matrices will stop
    being unitary at some point, with the rate of divergence dependent on the complexity of the Hamiltonian. In general, the
    closest unitary :math:`U`, i.e., one with the smallest 2-norm :math:`\lVert U-A \rVert`, to an operator :math:`A` can be
    calculated via a singular value decomposition:

    .. math::

        A & = V \Sigma W^{\dagger}, \\
        U & = V W^{\dagger}.

    This function has an option ``reunitarize`` to apply the above decomposition and recompute the time evolution matrices.
    Because reunitarizing at each time step can be rather slow (mostly because of the overhead in calling ``sesolve``), when
    this option is turned on (default), ``sesolve`` is called in time intervals of length ``interval_len``, and the entire
    array of evolution matrices from each interval is reunitarized (longer interval lengths result in larger reunitarization
    corrections).

    .. rubric:: Implementation notes (why we return an original object instead of the QuTiP result)

    When the coefficients of the time-dependent Hamiltonian are compiled (preferred
    method), QuTiP creates a transient python module with file name generated from the code hash, PID, and the current time.
    When running multiple simulations in parallel this is not strictly safe, and so we enclose ``sesolve`` in a context with
    a temporary directory in this function. The transient module is then deleted at the end of execution, but that in turn
    causes an error when this function is called in a subprocess and if we try to return the QuTiP result object directly
    through e.g. multiprocessing.Pipe. Somehow the result object tries to carry with it something defined in the transient
    module, which would therefore need to be pickled together with the returned object. But the transient module file is
    gone by the time the parent process receives the result from the pipe.
    So, the solution was to just return a "sanitized" object, consisting of plain ndarrays.

    Args:
        hgen: A HamiltonianBuilder.
        tlist: Time points to use in the simulation or a pair ``(points_per_cycle, num_cycles)`` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
        psi0: Initial state Qobj. Defaults to the identity operator appropriate for the given Hamiltonian.
        drive_args: Second parameter passed to drive amplitude functions (if callable).
        e_ops: List of observables passed to the QuTiP solver.
        frame: System frame to represent the simulation results in.
        final_only: Throw away intermediate states and expectation values to save memory. Due to implementation details,
            implies ``unitarize_evolution=True``.
        reunitarize: Whether to reunitarize the time evolution matrices. Note that SVD can be very slow for large system
            sizes and ``final_only=False``.
        interval_len: When integrating with QuTiP ``sesolve``, the number of time steps to run ``sesolve`` uninterrupted.
            In this case the parameter is ignored unless ``reunitarize=True`` and ``psi0`` is a unitary. When integrating
            with JAX ``odeint``,
        sesolve_options: QuTiP solver options.
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

    parallel_params = [
        (tlist, (np.ndarray, tuple, dict)),
        (psi0, Qobj),
        (e_ops, Sequence),
        (frame, (str, dict, Sequence)),
        (final_only, bool),
        (reunitarize, bool),
        (interval_len, int),
        (sesolve_options, Options)
    ]

    # Pack parallelizable arguments into per-job tuples
    for param, typ in parallel_params:
        if isinstance(param, list) and (typ is None or all(isinstance(elem, typ) for elem in param)):
            if num_tasks is None:
                num_tasks = len(param)
            elif num_tasks != len(param):
                raise ValueError('Lists with inconsistent lengths passed as arguments')

            zip_list.append(param)

        else:
            zip_list.append(None)

    if num_tasks is None:
        hamiltonian = build_hamiltonian(hgen, drive_args)
        raw_args = tuple(param for param, _ in parallel_params)
        parameters = compose_parameters(hgen, raw_args)

        result = _run_single(hamiltonian, parameters, drive_args=drive_args, save_result_to=save_result_to)

    else:
        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            num_digits = int(np.log10(num_tasks)) + 1

            save_result_path = lambda itask: os.path.join(save_result_to, f'%0{num_digits}d' % itask)
        else:
            save_result_path = lambda itask: None

        if not isinstance(drive_args, list):
            drive_args = [drive_args] * num_tasks

        hamiltonian = build_hamiltonian(hgen, drive_args[0])

        for iparam, (param, _) in enumerate(parallel_params):
            if zip_list[iparam] is None:
                zip_list[iparam] = [param] * num_tasks

        parameters = compose_parameters(hgen, zip(*zip_list))

        if config.pulse_sim_solver == 'jax':
            thread_based = True

            # Precompile the hamiltonian function
            rtol, atol, mxstep, hmax = 1.e-8, 1.e-8, jnp.inf, jnp.inf
            y0 = hgen.identity_op().full()
            ts = parameters[0].tlist[None, :1]
            vodeint(hamiltonian, rtol, atol, mxstep, hmax, y0, ts, drive_args[0])

            common_args = (hamiltonian,)
            args = parameters
            arg_position = 1
            kwarg_keys = ('drive_args', 'logger_name', 'save_result_to')
            kwarg_values = list((drive_args[itask], f'{__name__}.{itask}', save_result_path(itask))
                                for itask in range(num_tasks))

        else:
            thread_based = False

            # Convert the functional Hamiltonian coefficients to arrays before passing them to parallel_map
            hamiltonians = list()
            for params, dargs in zip(parameters, drive_args):
                hamiltonians.append(hamiltonian.evaluate_coeffs(params.tlist, dargs))

            common_args = None
            args = list(zip(hamiltonians, parameters))
            arg_position = (0, 1)
            kwarg_keys = ('logger_name', 'save_result_to')
            kwarg_values = list((f'{__name__}.{itask}', save_result_path(itask)) for itask in range(num_tasks))

        result = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys, kwarg_values=kwarg_values,
                              arg_position=arg_position, common_args=common_args, log_level=log_level,
                              thread_based=thread_based)

    logger.setLevel(original_log_level)

    return result


@dataclass
class PulseSimParameters:
    frame: SystemFrame
    tlist: np.ndarray
    psi0: Qobj
    e_ops: Optional[Sequence[Qobj]] = None
    final_only: bool = False
    reunitarize: bool = True
    interval_len: Union[int, str] = 'auto'
    sesolve_options: Options = Options()

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
        states = evolution @ np.squeeze(self.psi0.full())

        if not self.e_ops:
            return states, None

        # Fill the expectation values
        expect = list()
        for observable in self.e_ops:
            if isinstance(observable, Qobj):
                op = observable.full()
                if len(states.shape) == 3:
                    out = np.einsum('ij,tij->t', op.conjugate(), states)
                else:
                    out = np.einsum('ti,ij,tj->t', states.conjugate(), op, states).real

            else:
                out = np.empty_like(tlist)
                for it, state in enumerate(states):
                    out[it] = observable(tlist[it], state)

            expect.append(out)

        return states, expect


def build_hamiltonian(
    hgen: HamiltonianBuilder,
    sample_drive_args: Optional[Dict[str, Any]] = None
) -> Any:
    """Build the Hamiltonian according to the pulse_sim_solver option."""
    if config.pulse_sim_solver == 'jax':
        h_terms = hgen.build(frame='lab', as_timefn=True)

        if len(h_terms) == 1 and isinstance(h_terms[0], Qobj):
            # Will just exponentiate
            hamiltonian = h_terms
        else:
            @jax.jit
            def du_dt(u_t, t, drive_args):
                h_t = h_terms[0].full()
                for qobj, fn in h_terms[1:]:
                    h_t += qobj.full() * fn(t, drive_args)

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
    raw_args_list: Union[Tuple[Any, ...], Iterable[Tuple[Any, ...]]]
) -> List[Tuple[Any, ...]]:
    """Generate the arguments for parallel jobs using the HamiltonianBuilder."""
    singleton_input = False
    if isinstance(raw_args_list, tuple) and isinstance(raw_args_list[-1], Options):
        singleton_input = True
        raw_args_list = [raw_args_list]

    parameters_list = list()

    for tlist, psi0, e_ops, frame, final_only, reunitarize, interval_len, sesolve_options in raw_args_list:
        ## Time points
        if isinstance(tlist, tuple):
            tlist = {'points_per_cycle': tlist[0], 'num_cycles': tlist[1]}
        if isinstance(tlist, dict):
            tlist = dict(tlist)
            tlist['frame'] = 'lab'
            tlist = hgen.make_tlist(**tlist)

        if psi0 is None:
            psi0 = hgen.identity_op()

        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, hgen)

        parameters = PulseSimParameters(frame, tlist, psi0, e_ops=e_ops, final_only=final_only,
                                        reunitarize=reunitarize, interval_len=interval_len,
                                        sesolve_options=sesolve_options)

        parameters_list.append(parameters)

    if singleton_input:
        return parameters_list[0]
    else:
        return parameters_list


def _run_single(
    hamiltonian: Any,
    parameters: PulseSimParameters,
    drive_args: Optional[Dict[str, Any]] = None,
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
) -> PulseSimResult:
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    logger.info('Using %d time points from %.3e to %.3e', parameters.tlist.shape[0], parameters.tlist[0],
                parameters.tlist[-1])

    ## Run the simulation or exponentiate the static Hamiltonian
    if isinstance(hamiltonian, list) and len(hamiltonian) == 1 and isinstance(hamiltonian[0], Qobj):
        logger.info('Static Hamiltonian built. Exponentiating..')

        states, expect = exponentiate_hstat(hamiltonian, parameters, logger_name)

    elif config.pulse_sim_solver == 'jax':
        logger.info('XLA-compiled Hamiltonian built. Starting simulation..')

        states, expect = simulate_drive_odeint(hamiltonian, parameters, drive_args, logger_name)

    else:
        logger.info('Hamiltonian with %d terms built. Starting simulation..', len(h))

        states, expect = simulate_drive_sesolve(hamiltonian, parameters, drive_args, logger_name)

    ## Compile the result and return
    if parameters.final_only:
        tlist = parameters.tlist[-1:]
    else:
        tlist = parameters.tlist

    if parameters.e_ops and not parameters.sesolve_options.store_states:
        states = None

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
        return exponentiate_hstat(hamiltonian, parameters, logger_name)

    # Take a copy of options because we change its values

    sesolve_options = copy.deepcopy(parameters.sesolve_options)

    # Reunitarization and interval running setting

    reunitarize = parameters.reunitarize
    if parameters.final_only and parameters.interval_len < parameters.tlist.shape[0]:
        # Final only option with a short interval len implies that we run sesolve in intervals.
        # The evolution matrices must be reunitarized in this case because an op psi0 argument
        # of sesolve must be unitary, but the evolution matrices diverge from unitary typically
        # within a few hundred time steps.
        reunitarize = True

    interval_len = parameters.interval_len
    if interval_len == 'auto':
        if reunitarize:
            interval_len = 500
        else:
            interval_len = parameters.tlist.shape[0]

    # sesolve generates O(1e-6) error in the global phase when the Hamiltonian is not traceless.
    # Since we run the simulation in the lab frame, the only traceful term is hdiag.
    # We therefore subtract the diagonals to make hdiag traceless, and later apply a global phase
    # according to the shift.
    hdiag = hamiltonian[0]
    global_phase = np.trace(hdiag.full()).real / hdiag.shape[0]
    hamiltonian[0] = hdiag - global_phase

    isarray = list(isinstance(term, list) and isinstance(term[1], np.ndarray) for term in hamiltonian)
    if not any(isarrayd):
        # No interval dependency in the Hamiltonian (all constants or string expressions)
        # -> Can reuse the compiled objects
        sesolve_options.rhs_reuse = True

    # Arrays to store the unitaries and expectation values

    if not parameters.e_ops or sesolve_options.store_states:
        # squeezed shape
        state_shape = tuple(s for s in parameters.psi0.shape if s != 1)
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
    sesolve_options.store_states = True

    sim_start = time.time()

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            os.chdir(tempdir)

            start = 0
            # Compute the time evolution in the lab frame -> initial unitary is identity
            local_psi0 = qeye(hdiag.dims[0])
            while True:
                local_hamiltonian = list()
                for h_term, isa in zip(hamiltonian, isarray):
                    if isa:
                        local_hamiltonian.append([h_term[0], h_term[1][start:start + interval_len]])
                    else:
                        local_hamiltonian.append(h_term)

                local_tlist = parameters.tlist[start:start + interval_len]

                qtp_result = sesolve(local_hamiltonian,
                                     local_psi0,
                                     local_tlist,
                                     options=sesolve_options)

                # Array of lab-frame evolution ops
                if parameters.final_only:
                    evolution = np.squeeze(qtp_result.states[-1].full())[None, ...]
                    local_tlist = local_tlist[-1:]
                else:
                    evolution = np.stack(list(np.squeeze(state.full()) for state in qtp_result.states))

                # Apply reunitarization
                if reunitarize:
                    evolution = closest_unitary(evolution)

                last_unitary = evolution[-1].copy()

                # Restore the actual global phase
                evolution *= np.exp(-1.j * global_phase * local_tlist[:, None, None])

                # Compute the states and expectation values in the original frame
                local_states, local_expect = parameters.transform_evolution(evolution, local_tlist)

                # Indices for the output arrays
                if parameters.final_only:
                    out_slice = slice(0, 1)
                else:
                    out_slice = slice(start, start + interval_len)

                # Fill the states
                if states is not None:
                    states[out_slice] = local_states

                # Fill the expectation values
                if expect is not None:
                    for out, local in zip(expect, local_expect):
                        out[out_slice] = local

                # Update start and local_psi0
                start += interval_len - 1

                if start < parameters.tlist.shape[0] - 1:
                    # evolution[-1] is the lab-frame evolution operator
                    local_psi0 = Qobj(inpt=last_unitary, dims=local_psi0.dims)
                else:
                    break

                logger.debug('Processed interval %d-%d. Cumulative simulation time %f seconds.',
                             start - interval_len + 1, start + 1, time.time() - sim_start)

        finally:
            os.chdir(cwd)
            # Clear the cached solver
            rhs_clear()

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect


def simulate_drive_odeint(
    hamiltonian: Callable,
    parameters: PulseSimParameters,
    drive_args: Dict[str, Any] = dict(),
    logger_name: str = __name__
):
    logger = logging.getLogger(logger_name)

    interval_len = parameters.interval_len
    if interval_len == 'auto':
        # Best determined by the representative frequency and simulation duration
        interval_len = 100

    tlist = parameters.tlist
    num_intervals = (tlist.shape[0] - 1) // interval_len
    num_extra_points = tlist.shape[0] - 1 - interval_len * num_intervals
    if num_extra_points != 0:
        num_intervals += 1
        num_extra_points = interval_len * num_intervals - (tlist.shape[0] - 1)
        dt = tlist[-1] - tlist[-2]
        new_end = tlist[-1] + dt * num_extra_points
        tlist = np.concatenate([tlist[:-1], np.linspace(tlist[-1], new_end, num_extra_points + 1)])

    if parameters.final_only:
        tlists = np.array(list([tlist[iv * interval_len], tlist[(iv + 1) * interval_len]]
                               for iv in range(num_intervals)))
    else:
        tlists = np.array(list(tlist[iv * interval_len:(iv + 1) * interval_len + 1]
                               for iv in range(num_intervals)))

    sim_start = time.time()

    y0 = qeye(parameters.psi0.dims[0]).full()

    if hamiltonian.__name__ == 'du_dt':
        hamiltonian, _ = jax.custom_derivatives.closure_convert(hamiltonian, y0, 0., drive_args)

    rtol, atol, mxstep, hmax = 1.e-8, 1.e-8, jnp.inf, jnp.inf
    stacked_u = vodeint(hamiltonian, rtol, atol, mxstep, hmax, y0, tlists, drive_args)

    evolution = serialize_evolution(stacked_u)

    if num_extra_points != 0:
        # Truncate back to the original tlist
        evolution = evolution[:-num_extra_points]
        tlist = tlist[:-num_extra_points]

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

vodeint = jax.vmap(_odeint_wrapper, in_axes=(None, None, None, None, None, None, 0, None))

@jax.jit
def serialize_evolution(stacked_u):
    dot_last = lambda ip, matrices: matrices.at[ip].set(stacked_u[ip, -1] @ matrices[ip - 1])

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
