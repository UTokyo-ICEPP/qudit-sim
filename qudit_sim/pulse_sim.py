"""Pulse simulation frontend."""

from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import os
import tempfile
import logging
import time
import copy

import numpy as np
import qutip as qtp
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from rqutils.math import matrix_exp

from .hamiltonian import HamiltonianBuilder, Frame
from .sim_result import PulseSimResult, save_sim_result
from .unitary import closest_unitary
from .parallel import parallel_map
from .config import config

logger = logging.getLogger(__name__)

TList = Union[np.ndarray, Tuple[int, int], Dict[str, Union[int, float]]]
EOps = Sequence[qtp.Qobj]

def pulse_sim(
    hgen: HamiltonianBuilder,
    tlist: Union[TList, List[TList]] = (10, 100),
    psi0: Optional[Union[qtp.Qobj, List[qtp.Qobj]]] = None,
    args: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    e_ops: Optional[Union[EOps, List[EOps]]] = None,
    final_only: bool = False,
    reunitarize: bool = True,
    interval_len: int = 1000,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union['PulseSimResult', List['PulseSimResult']]:
    r"""Run a pulse simulation.

    Build the Hamiltonian terms from the HamiltonianBuilder, determine the time points for the simulation if necessary,
    and run ``qutip.sesolve``.

    Regardless of the setup of the input (Hamiltonian frame, initial state, and expectation ops), the simulation is run in
    the lab frame with the identity operator as the initial state to obtain the time evolution operator for each time point.
    The evolution operators are then frame-changed to the original frame of the input Hamiltonian before being applied to
    the initial state (or initial unitary). If expectation ops are given, their expectation values are computed from the
    resulting evolved states.

    All parameters except ``options``, ``save_result_to``, and ``log_level`` can be given as lists to trigger a parallel
    (multiprocess) execution of ``sesolve``. If more than one parameter is a list, their lengths must be identical, and all
    parameters are "zipped" together to form the argument lists of individual single simulation jobs.

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
        hgen: A HamiltonianBuilder or a list thereof.
        tlist: Time points to use in the simulation or a pair ``(points_per_cycle, num_cycles)`` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
        psi0: Initial state Qobj. Defaults to the identity operator appropriate for the given Hamiltonian.
        args: Second parameter passed to drive amplitude functions (if callable).
        e_ops: List of observables passed to the QuTiP solver.
        final_only: Throw away intermediate states and expectation values to save memory. Due to implementation details,
            implies ``unitarize_evolution=True``.
        reunitarize: Whether to reunitarize the time evolution matrices.
        interval_len: Number of time steps to run ``sesolve`` uninterrupted. Only used when ``reunitarize=True``
            and ``psi0`` is a unitary.
        options: QuTiP solver options.
        save_result_to: File name (without the extension) to save the simulation result to.
        log_level: Log level.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    num_tasks = None
    zip_list = []

    parallel_params = [
        (tlist, (np.ndarray, tuple, dict)),
        (psi0, qtp.Qobj),
        (args, dict),
        (e_ops, Sequence),
        (final_only, bool),
        (reunitarize, bool),
        (interval_len, int)
    ]

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
        args = tuple(param[0] for param in parallel_params)
        kwargs = {'options': options, 'save_result_to': save_result_to}

        result = _run_single(*args, **kwargs)

    else:
        for iparam, (param, _) in enumerate(parallel_params):
            if zip_list[iparam] is None:
                zip_list[iparam] = [param] * num_tasks

        args = list(zip(*zip_list))
        arg_position = list(range(1, len(zip_list) + 1))
        common_args = (hgen,)
        common_kwargs = {'options': options}

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            num_digits = int(np.log10(num_tasks)) + 1

            save_result_path = lambda itask: os.path.join(save_result_to, f'%0{num_digits}d' % itask)
        else:
            save_result_path = lambda itask: None

        kwarg_keys = ('logger_name', 'save_result_to')
        kwarg_values = list()
        for itask in range(num_tasks):
            kwarg_values.append((f'{__name__}.{itask}', save_result_path(itask)))

        thread_based = config.pulse_sim_solver == 'jax'

        result = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys, kwarg_values=kwarg_values,
                              common_args=common_args, common_kwargs=common_kwargs,
                              arg_position=arg_position, log_level=log_level, thread_based=thread_based)

    logger.setLevel(original_log_level)

    return result


def _run_single(
    hgen: HamiltonianBuilder,
    tlist: TList,
    psi0: Union[qtp.Qobj, None],
    args: Dict[str, Any],
    e_ops: Union[EOps, None],
    final_only: bool,
    reunitarize: bool,
    interval_len: int,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
) -> PulseSimResult:
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    ## Create a lab-frame copy of hgen

    lab_hgen = hgen.copy()
    lab_hgen.set_global_frame('lab')

    ## Define the time points if necessary

    if isinstance(tlist, tuple):
        tlist = {'points_per_cycle': tlist[0], 'num_cycles': tlist[1]}

    if isinstance(tlist, dict):
        tlist = lab_hgen.make_tlist(**tlist)

    logger.info('Using %d time points from %.3e to %.3e', tlist.shape[0], tlist[0], tlist[-1])

    ## Initial state

    if psi0 is None:
        psi0 = hgen.identity_op()

    ## Set up the states and expectation values calculator

    calculator = StatesAndExpectations(hgen, psi0, e_ops, tlist[0])

    ## Run the simulation or exponentiate the static Hamiltonian

    if any(len(d) > 0 for d in hgen.drive().values()):
        states, expect = _simulate_drive(lab_hgen, tlist, calculator, args,
                                         final_only, reunitarize, interval_len, options, logger)

    else:
        states, expect = _exponentiate(lab_hgen, tlist, calculator, final_only, logger)

        if expect and (options is None or not options.store_states):
            states = None

    ## Compile the result and return

    if final_only:
        tlist = tlist[-1:]

    dim = (hgen.num_levels,) * hgen.num_qudits
    frame_tuple = tuple(hgen.frame().values())

    result = PulseSimResult(times=tlist, expect=expect, states=states, dim=dim, frame=frame_tuple)

    if save_result_to:
        logger.info('Saving the simulation result to %s.h5', save_result_to)
        save_sim_result(f'{save_result_to}.h5', result)

    return result


class StatesAndExpectations:
    def __init__(self, hgen, psi0, e_ops, t0):
        ## Define the frame reverting function
        original_frame = hgen.frame()

        if any((np.any(frame.frequency) or np.any(frame.phase))
               for frame in original_frame.values()):
            # original frame is not lab
            self.revert_frame = lambda e, t: hgen.change_frame(t, e, from_frame='lab',
                                                               objtype='evolution', t0=t0)
        else:
            self.revert_frame = lambda e, t: e

        self.psi0_arr = np.squeeze(psi0.full())
        self.e_ops = e_ops

    def calculate(self, evolution, tlist):
        # Change the frame back to original
        evolution = self.revert_frame(evolution, tlist)

        # State evolution in the original frame
        states = evolution @ self.psi0_arr

        if not self.e_ops:
            return states, None

        # Fill the expectation values
        expect = list()
        for observable in self.e_ops:
            if isinstance(observable, qtp.Qobj):
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


def _simulate_drive(
    hgen: HamiltonianBuilder,
    tlist: np.ndarray,
    calculator: StatesAndExpectations,
    args: Dict[str, Any],
    final_only: bool,
    reunitarize: bool,
    interval_len: int,
    options: Union[None, qtp.solver.Options],
    logger: logging.Logger
):
    """Simulate the time evolution under a dynamic drive using qutip.sesolve."""
    ## Build the Hamiltonian
    hamiltonian = hgen.build(tlist=tlist, args=args)

    identity = hgen.identity_op()

    logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

    if config.pulse_sim_solver == 'jax':
        return _simulate_drive_odeint(hamiltonian, identity, tlist, calculator, final_only, reunitarize, logger)
    else:
        return _simulate_drive_sesolve(hamiltonian, identity, tlist, calculator, args, final_only, reunitarize,
                                       interval_len, options, logger)

def _simulate_drive_sesolve(
    hamiltonian: List[Union[qtp.Qobj, List]],
    identity: qtp.Qobj,
    tlist: np.ndarray,
    calculator: StatesAndExpectations,
    args: Any,
    final_only: bool,
    reunitarize: bool,
    interval_len: int,
    options: Union[None, qtp.solver.Options],
    logger: logging.Logger
):
    ## Options that are actually used in sesolve

    if options is None:
        solve_options = qtp.solver.Options()
    else:
        solve_options = copy.deepcopy(options)

    ## Reunitarization and interval running setting

    if final_only and interval_len < tlist.shape[0]:
        # Final only option with a short interval len implies that we run sesolve in intervals.
        # The evolution matrices must be reunitarized in this case because an op psi0 argument
        # of sesolve must be unitary, but the evolution matrices diverge from unitary typically
        # within a few hundred time steps.
        reunitarize = True

    if not reunitarize:
        interval_len = tlist.shape[0]

    # sesolve generates O(1e-6) error in the global phase when the Hamiltonian is not traceless.
    # Since we run the simulation in the lab frame, the only traceful term is hdiag.
    # We therefore subtract the diagonals to make hdiag traceless, and later apply a global phase
    # according to the shift.
    hdiag = hamiltonian[0]
    global_phase = np.trace(hdiag.full()).real / hdiag.shape[0]
    hamiltonian[0] -= global_phase

    array_terms = list()

    if interval_len < tlist.shape[0]:
        local_hamiltonian = copy.deepcopy(hamiltonian)

        for iterm, term in enumerate(hamiltonian):
            if isinstance(term, list) and isinstance(term[1], np.ndarray):
                array_terms.append(iterm)

        if not array_terms:
            # We can reuse the same Hamiltonian for multiple intervals
            solve_options.rhs_reuse = True

    else:
        local_hamiltonian = hamiltonian

    ## Arrays to store the unitaries and expectation values

    if not calculator.e_ops or solve_options.store_states:
        state_shape = calculator.psi0_arr.shape
        if final_only:
            states = np.empty((1,) + state_shape, dtype=np.complex128)
        else:
            states = np.empty(tlist.shape + state_shape, dtype=np.complex128)
    else:
        states = None

    if calculator.e_ops:
        num_e_ops = len(calculator.e_ops)
        if final_only:
            expect = list(np.empty(1) for _ in range(num_e_ops))
        else:
            expect = list(np.empty_like(tlist) for _ in range(num_e_ops))
    else:
        expect = None

    ## Run sesolve in a temporary directory

    # We'll always need the states
    solve_options.store_states = True

    sim_start = time.time()

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            os.chdir(tempdir)

            start = 0
            # Compute the time evolution in the lab frame -> initial unitary is identity
            local_psi0 = identity
            while True:
                for iterm in array_terms:
                    local_hamiltonian[iterm][1] = hamiltonian[iterm][1][start:start + interval_len]

                local_tlist = tlist[start:start + interval_len]

                qtp_result = qtp.sesolve(local_hamiltonian,
                                         local_psi0,
                                         local_tlist,
                                         args=args,
                                         options=solve_options)

                # Array of lab-frame evolution ops
                if final_only:
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
                local_states, local_expect = calculator.calculate(evolution, local_tlist)

                # Indices for the output arrays
                if final_only:
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

                if start < tlist.shape[0] - 1:
                    # evolution[-1] is the lab-frame evolution operator
                    local_psi0 = qtp.Qobj(inpt=last_unitary, dims=local_psi0.dims)
                else:
                    break

                logger.debug('Processed interval %d-%d. Cumulative simulation time %f seconds.',
                             start - interval_len + 1, start + 1, time.time() - sim_start)

        finally:
            os.chdir(cwd)
            # Clear the cached solver
            qtp.rhs_clear()

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect

def _simulate_drive_odeint(
    hamiltonian: List[Union[qtp.Qobj, List]],
    identity: qtp.Qobj,
    tlist: np.ndarray,
    calculator: StatesAndExpectations,
    final_only: bool,
    reunitarize: bool,
    logger: logging.Logger
):
    @jax.jit
    def du_dt(u_t, t):
        h_t = hamiltonian[0].full()
        for qobj, fn in hamiltonian[1:]:
            h_t += qobj.full() * fn(t, None)

        return -1.j * jnp.matmul(h_t, u_t)

    if final_only:
        tlist = jnp.array([tlist[0], tlist[-1]], dtype='float64')

    sim_start = time.time()

    evolution = odeint(du_dt, identity.full(), tlist, rtol=1.e-8, atol=1.e-8)

    # Apply reunitarization
    if reunitarize:
        evolution = closest_unitary(evolution)

    ## Restore the actual global phase
    #evolution *= np.exp(-1.j * global_phase * local_tlist[:, None, None])

    # Compute the states and expectation values in the original frame
    states, expect = calculator.calculate(evolution, tlist)

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect

def _exponentiate(
    hgen: HamiltonianBuilder,
    tlist: np.ndarray,
    calculator: StatesAndExpectations,
    final_only: bool,
    logger: logging.Logger
):
    """Compute exp(-iHt)."""
    hamiltonian = hgen.build()[0].full()

    if final_only:
        evolution = matrix_exp(-1.j * hamiltonian[None, ...] * tlist[-1:, None, None], hermitian=-1)
        tlist = tlist[-1:]
    else:
        evolution = matrix_exp(-1.j * hamiltonian[None, ...] * tlist[:, None, None], hermitian=-1)

    return calculator.calculate(evolution, tlist)
