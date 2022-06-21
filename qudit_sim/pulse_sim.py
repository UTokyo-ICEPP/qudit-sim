"""Pulse simulation frontend."""

from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import typing
import os
import tempfile
import logging
import time
import copy
import collections

import numpy as np
import h5py
import qutip as qtp

from .hamiltonian import HamiltonianBuilder
from .util import PulseSimResult, save_sim_result
from .parallel import parallel_map

logger = logging.getLogger(__name__)

TList = Union[np.ndarray, Tuple[int, int], Dict[str, Union[int, float]]]
EOps = Sequence[qtp.Qobj]

def pulse_sim(
    hgen: Union[HamiltonianBuilder, List[HamiltonianBuilder]],
    tlist: Union[TList, List[TList]] = (10, 100),
    psi0: Optional[Union[qtp.Qobj, List[qtp.Qobj]]] = None,
    args: Optional[Any] = None,
    e_ops: Optional[Union[EOps, List[EOps]]] = None,
    keep_callable: Union[bool, List[bool]] = False,
    final_only: bool = False,
    reunitarize: bool = True,
    interval_len: int = 1000,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union[PulseSimResult, List[PulseSimResult]]:
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
        keep_callable: Keep callable time-dependent Hamiltonian coefficients. Otherwise all callable coefficients
            are converted to arrays before simulation execution for efficiency (no loss of accuracy observed so far).
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
        (hgen, HamiltonianBuilder),
        (tlist, (np.ndarray, tuple, dict)),
        (psi0, qtp.Qobj),
        (args, None),
        (e_ops, Sequence),
        (keep_callable, bool),
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
        kwargs = {'options': options, 'save_result_to': save_result_to, 'log_level': log_level}

        result = _run_single(*args, **kwargs)

    else:
        for iparam, (param, _) in enumerate(parallel_params):
            if zip_list[iparam] is None:
                zip_list[iparam] = [param] * num_tasks

        args = list(zip(*zip_list))

        common_kwargs = {'options': options, 'log_level': log_level}

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

        result = parallel_map(_run_single, args=args, kwarg_keys=kwarg_keys, kwarg_values=kwarg_values,
                              common_kwargs=common_kwargs, log_level=log_level)

    logger.setLevel(original_log_level)

    return result


def _run_single(
    hgen: HamiltonianBuilder,
    tlist: TList,
    psi0: Union[qtp.Qobj, None],
    args: Any,
    e_ops: Union[EOps, None],
    keep_callable: bool,
    final_only: bool,
    reunitarize: bool,
    interval_len: int,
    options: Optional[qtp.solver.Options] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING,
    logger_name: str = __name__
):
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    ## Move to the lab frame

    original_frame = hgen.frame()

    nonlab_original_frame = any((np.any(frame.frequency) or np.any(frame.phase))
                                for frame in original_frame.values())

    if nonlab_original_frame:
        hgen = hgen.copy()
        hgen.set_global_frame('lab')

    ## Define the time points if necessary

    if isinstance(tlist, tuple):
        tlist = hgen.make_tlist(points_per_cycle=tlist[0], num_cycles=tlist[1])
    elif isinstance(tlist, dict):
        tlist = hgen.make_tlist(**tlist)

    logger.info('Using %d time points from %.3e to %.3e', tlist.shape[0], tlist[0], tlist[-1])

    ## Options that are actually used in sesolve

    if options is None:
        solve_options = qtp.solver.Options()
    else:
        solve_options = copy.deepcopy(options)

    ## Build the Hamiltonian

    if keep_callable:
        tlist_arg = dict()
    else:
        tlist_arg = {'tlist': tlist, 'args': args}

    hamiltonian = hgen.build(rwa=False, **tlist_arg)

    ## Reunitarization and interval running setting

    if final_only and interval_len < tlist.shape[0]:
        # Final only option with a short interval len implies that we run sesolve in intervals.
        # The evolution matrices must be reunitarized in this case because an op psi0 argument
        # of sesolve must be unitary, but the evolution matrices diverge from unitary typically
        # within a few hundred time steps.
        reunitarize = True

    if not reunitarize:
        interval_len = tlist.shape[0]

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

    ## Initial state

    identity = qtp.tensor([qtp.qeye(hgen.num_levels)] * hgen.num_qudits)

    if psi0 is None:
        psi0 = identity

    ## Arrays to store the states and expectation values

    if not e_ops or solve_options.store_states:
        if final_only:
            states = np.empty((1,) + np.squeeze(psi0.full()).shape, dtype=np.complex128)
        else:
            states = np.empty(tlist.shape + np.squeeze(psi0.full()).shape, dtype=np.complex128)
    else:
        states = None

    if e_ops:
        if final_only:
            expect = list(np.empty(1) for _ in range(len(e_ops)))
        else:
            expect = list(np.empty_like(tlist) for _ in range(len(e_ops)))
    else:
        expect = None

    ## Run sesolve in a temporary directory

    logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

    # We'll always need the states
    solve_options.store_states = True
    # psi0 in the original frame as an array
    psi0_arr = np.squeeze(psi0.full())

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
                    v, _, wdag = np.linalg.svd(evolution)
                    evolution = v @ wdag

                # Keep the last one - to be used as the local_psi0 of the next interval
                lab_frame_final = evolution[-1]

                # Change the frame back to original
                if nonlab_original_frame:
                    evolution = hgen.change_frame(local_tlist, evolution,
                                                  from_frame='lab', to_frame=original_frame,
                                                  objtype='evolution', t0=tlist[0])

                # State evolution in the original frame
                local_states = evolution @ psi0_arr

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
                    for observable, out in zip(e_ops, expect):
                        if isinstance(observable, qtp.Qobj):
                            op = observable.full()
                            if psi0.isoper:
                                out[out_slice] = np.einsum('ij,tij->t', op.conjugate(), local_states)
                            else:
                                out[out_slice] = np.einsum('ti,ij,tj->t',
                                                           local_states.conjugate(),
                                                           op,
                                                           local_states).real

                        else:
                            for it, out_idx in enumerate(np.r_[out_slice]):
                                out[out_idx] = observable(local_tlist[it], local_states[it])

                # Update start and local_psi0
                start += interval_len - 1

                if start < tlist.shape[0] - 1:
                    local_psi0 = qtp.Qobj(inpt=lab_frame_final, dims=identity.dims)
                else:
                    break

                logger.debug('Processed interval %d-%d. Cumulative simulation time %f seconds.',
                             start - interval_len + 1, start + 1, time.time() - sim_start)

        finally:
            os.chdir(cwd)
            # Clear the cached solver
            qtp.rhs_clear()

    logger.info('Done in %f seconds.', time.time() - sim_start)

    ## Truncate the tlist
    if final_only:
        tlist = tlist[-1:]

    dim = (hgen.num_levels,) * hgen.num_qudits
    frame_tuple = tuple(original_frame[qudit_id] for qudit_id in hgen.qudit_ids())

    result = PulseSimResult(times=tlist, expect=expect, states=states, dim=dim, frame=frame_tuple)

    if save_result_to:
        logger.info('Saving the simulation result to %s.h5', save_result_to)
        save_sim_result(f'{save_result_to}.h5', result)

    return result
