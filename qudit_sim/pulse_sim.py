"""Pulse simulation frontend."""

from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable
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

from .hamiltonian import HamiltonianBuilder
from .frame import FrameSpec, SystemFrame, QuditFrame
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
    drive_args: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    e_ops: Optional[Union[EOps, List[EOps]]] = None,
    frame: Optional[Union[FrameSpec, List[FrameSpec]]] = None,
    final_only: Optional[Union[bool, List[bool]]] = False,
    reunitarize: Optional[Union[bool, List[bool]]] = True,
    interval_len: Optional[Union[int, List[int]]] = 1000,
    options: Optional[qtp.solver.Options] = None,
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

    All parameters except ``hgen``, ``options``, ``save_result_to``, and ``log_level`` can be given as lists to trigger a
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
        (drive_args, dict),
        (e_ops, Sequence),
        (frame, (str, dict, Sequence)),
        (final_only, bool),
        (reunitarize, bool),
        (interval_len, int)
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
        raw_args = [tuple(param for param, _ in parallel_params)]
        args = _prepare_args(hgen, raw_args)[0]
        kwargs = {'options': options, 'save_result_to': save_result_to}

        result = _run_single(*args, **kwargs)

    else:
        for iparam, (param, _) in enumerate(parallel_params):
            if zip_list[iparam] is None:
                zip_list[iparam] = [param] * num_tasks

        args = _prepare_args(hgen, list(zip(*zip_list)))

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
                              common_kwargs=common_kwargs,
                              log_level=log_level, thread_based=thread_based)

    logger.setLevel(original_log_level)

    return result


def _prepare_args(
    hgen: HamiltonianBuilder,
    raw_args_list: List[Tuple[Any, ...]]
) -> List[Tuple[Any, ...]]:
    """Generate the arguments for parallel jobs using the HamiltonianBuilder."""
    args = list()

    for tlist, psi0, drive_args, e_ops, frame, final_only, reunitarize, interval_len in raw_args_list:
        ## Time points
        if isinstance(tlist, tuple):
            tlist = {'points_per_cycle': tlist[0], 'num_cycles': tlist[1]}
        if isinstance(tlist, dict):
            tlist = dict(tlist)
            tlist['frame'] = 'lab'
            tlist = hgen.make_tlist(**tlist)

        ## Drive args
        if drive_args is None:
            drive_args = dict()

        ## Build the Hamiltonian
        if config.pulse_sim_solver == 'jax':
            if len(args):
                # All threads will use the same hamiltonian function
                hamiltonian = args[0][0]
            else:
                h_terms = hgen.build(frame='lab', as_timefn=True)

                if len(h_terms) == 1 and isinstance(h_terms[0], qtp.Qobj):
                    # Will just exponentiate
                    hamiltonian = h_terms
                else:
                    @jax.jit
                    def hamiltonian(u_t, t, drive_args):
                        h_t = h_terms[0].full()
                        for qobj, fn in h_terms[1:]:
                            h_t += qobj.full() * fn(t, drive_args)

                        return -1.j * jnp.matmul(h_t, u_t)

        else:
            hamiltonian = hgen.build(frame='lab').evaluate_coeffs(tlist, drive_args)

        ## Initial state
        if psi0 is None:
            psi0 = hgen.identity_op()

        identity = hgen.identity_op()

        ## Set up the states and expectation values calculator
        calculator = StatesAndExpectations(frame, psi0, tlist, e_ops=e_ops, hgen=hgen)

        args.append((hamiltonian, calculator, identity, drive_args,
                     final_only, reunitarize, interval_len))

    if config.pulse_sim_solver == 'jax' and len(args) > 1:
        # Precompile the hamiltonian function
        hamiltonian, _, identity, drive_args = args[0][:4]
        odeint(hamiltonian, identity.full(), tlist[:1], drive_args, rtol=1.e-8, atol=1.e-8)

    return args


class StatesAndExpectations:
    def __init__(
        self,
        frame: FrameSpec,
        psi0: qtp.Qobj,
        tlist: np.ndarray,
        e_ops: Optional[EOps] = None,
        hgen: Optional[HamiltonianBuilder] = None
    ):
        if frame is None:
            frame = 'dressed'

        self.frame = SystemFrame(frame, hgen)
        self.psi0 = psi0
        self.e_ops = e_ops
        self.tlist = tlist

    def calculate(self, evolution, tlist=None):
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


def _run_single(
    hamiltonian: Union[List[Union[qtp.Qobj, List]], Callable],
    calculator: StatesAndExpectations,
    identity: qtp.Qobj,
    drive_args: Dict[str, Any],
    final_only: bool,
    reunitarize: bool,
    interval_len: int,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
) -> PulseSimResult:
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    logger.info('Using %d time points from %.3e to %.3e', calculator.tlist.shape[0], calculator.tlist[0],
                calculator.tlist[-1])

    ## Run the simulation or exponentiate the static Hamiltonian
    if isinstance(hamiltonian, list) and len(hamiltonian) == 1 and isinstance(hamiltonian[0], qtp.Qobj):
        logger.info('Static Hamiltonian built. Exponentiating..')

        states, expect = exponentiate_hstat(hamiltonian[0], calculator, final_only, logger)

    else:
        if config.pulse_sim_solver == 'jax':
            logger.info('XLA-compiled Hamiltonian built. Starting simulation..')

            states, expect = simulate_drive_odeint(hamiltonian, identity, calculator, drive_args, final_only,
                                                   reunitarize, logger)
        else:
            logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

            states, expect = simulate_drive_sesolve(hamiltonian, identity, calculator, final_only, reunitarize,
                                                    interval_len, options, logger)

    ## Compile the result and return
    if final_only:
        tlist = calculator.tlist[-1:]
    else:
        tlist = calculator.tlist

    if calculator.e_ops and options and not options.store_states:
        states = None

    result = PulseSimResult(times=tlist, expect=expect, states=states, frame=calculator.frame)

    if save_result_to:
        logger.info('Saving the simulation result to %s.h5', save_result_to)
        save_sim_result(f'{save_result_to}.h5', result)

    return result


def simulate_drive_sesolve(
    hamiltonian: List[Union[qtp.Qobj, List]],
    identity: qtp.Qobj,
    calculator: StatesAndExpectations,
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

    if final_only and interval_len < calculator.tlist.shape[0]:
        # Final only option with a short interval len implies that we run sesolve in intervals.
        # The evolution matrices must be reunitarized in this case because an op psi0 argument
        # of sesolve must be unitary, but the evolution matrices diverge from unitary typically
        # within a few hundred time steps.
        reunitarize = True

    if not reunitarize:
        interval_len = calculator.tlist.shape[0]

    # sesolve generates O(1e-6) error in the global phase when the Hamiltonian is not traceless.
    # Since we run the simulation in the lab frame, the only traceful term is hdiag.
    # We therefore subtract the diagonals to make hdiag traceless, and later apply a global phase
    # according to the shift.
    hdiag = hamiltonian[0]
    global_phase = np.trace(hdiag.full()).real / hdiag.shape[0]
    hamiltonian[0] -= global_phase

    array_terms = list()

    if interval_len < calculator.tlist.shape[0]:
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
        # squeezed shape
        state_shape = tuple(s for s in calculator.psi0.shape if s != 1)
        if final_only:
            states = np.empty((1,) + state_shape, dtype=np.complex128)
        else:
            states = np.empty(calculator.tlist.shape + state_shape, dtype=np.complex128)
    else:
        states = None

    if calculator.e_ops:
        num_e_ops = len(calculator.e_ops)
        if final_only:
            expect = list(np.empty(1) for _ in range(num_e_ops))
        else:
            expect = list(np.empty_like(calculator.tlist) for _ in range(num_e_ops))
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

                local_tlist = calculator.tlist[start:start + interval_len]

                qtp_result = qtp.sesolve(local_hamiltonian,
                                         local_psi0,
                                         local_tlist,
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

                if start < calculator.tlist.shape[0] - 1:
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

def simulate_drive_odeint(
    hamiltonian: Callable,
    identity: qtp.Qobj,
    calculator: StatesAndExpectations,
    drive_args: Dict[str, Any],
    final_only: bool,
    reunitarize: bool,
    logger: logging.Logger
):
    if final_only:
        tlist = jnp.array([calculator.tlist[0], calculator.tlist[-1]], dtype='float64')
    else:
        tlist = calculator.tlist

    sim_start = time.time()

    evolution = odeint(hamiltonian, identity.full(), tlist, drive_args, rtol=1.e-8, atol=1.e-8)

    # Apply reunitarization
    if reunitarize:
        evolution = closest_unitary(evolution)

    ## Restore the actual global phase
    #evolution *= np.exp(-1.j * global_phase * local_tlist[:, None, None])

    # Compute the states and expectation values in the original frame
    states, expect = calculator.calculate(evolution, tlist)

    logger.info('Done in %f seconds.', time.time() - sim_start)

    return states, expect

def exponentiate_hstat(
    hamiltonian: qtp.Qobj,
    calculator: StatesAndExpectations,
    final_only: bool,
    logger: logging.Logger
):
    """Compute exp(-iHt)."""
    hamiltonian = hamiltonian.full()

    if final_only:
        tlist = calculator.tlist[-1:]
    else:
        tlist = calculator.tlist

    evolution = matrix_exp(-1.j * hamiltonian[None, ...] * tlist[:, None, None], hermitian=-1)

    states, expect = calculator.calculate(evolution, tlist)

    return states, expect
