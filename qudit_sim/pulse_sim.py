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
import qutip as qtp

from .hamiltonian import HamiltonianBuilder
from .util import PulseSimResult
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
    options: Optional[qtp.solver.Options] = None,
    progress_bar: Optional[qtp.ui.progressbar.BaseProgressBar] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Union[PulseSimResult, List[PulseSimResult]]:
    """Run a pulse simulation.

    Build the Hamiltonian terms from the HamiltonianBuilder, determine the time points for the simulation
    if necessary, and run ``qutip.sesolve``.

    All parameters except ``options``, ``progress_bar``, ``save_result_to``, and ``log_level`` can be given as lists to
    trigger a parallel (multiprocess) execution of ``sesolve``. If more than one parameter is a list, their lengths must
    be identical, and all parameters are "zipped" together to form the argument lists of individual single simulation jobs.

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
        options: QuTiP solver options.
        progress_bar: QuTiP progress bar.
        save_result_to: File name (without the extension) to save the simulation result to.
        log_level: Log level.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    num_tasks = None
    zip_list = []

    parallel_params = [hgen, tlist, psi0, args, e_ops, keep_callable]
    element_types = [HamiltonianBuilder, (np.ndarray, tuple, dict), qtp.Qobj, None, Sequence, bool, bool]

    for param, typ in zip(parallel_params, element_types):
        if isinstance(param, list) and (typ is None or all(isinstance(elem, typ) for elem in param)):
            if num_tasks is None:
                num_tasks = len(param)
            elif num_tasks != len(param):
                raise ValueError('Lists with inconsistent lengths passed as arguments')

            zip_list.append(param)

        else:
            zip_list.append(None)

    if num_tasks is None:
        result = _run_single(hgen, tlist, psi0, args, e_ops, keep_callable, options=options,
                             progress_bar=progress_bar, save_result_to=save_result_to, log_level=log_level)

    else:
        for iparam, param in enumerate(parallel_params):
            if zip_list[iparam] is None:
                zip_list[iparam] = [param] * num_tasks

        args = list(zip(*zip_list))

        common_kwargs = {'options': options, 'log_level': log_level}

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            save_result_path = lambda itask: os.path.join(save_result_to, f'sim_{itask}')
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
    e_ops: EOps,
    keep_callable: bool,
    options: Optional[qtp.solver.Options] = None,
    progress_bar: Optional[qtp.ui.progressbar.BaseProgressBar] = None,
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

    ## Define the initial state if necessary

    identity = qtp.tensor([qtp.qeye(hgen.num_levels)] * hgen.num_qudits)

    if psi0 is None:
        psi0 = identity

    if psi0.isket:
        objtype = 'state'
    else:
        objtype = 'evolution'

    psi0_data = hgen.change_frame(tlist[:1], psi0, from_frame=original_frame, to_frame='lab', objtype=objtype)
    dims = psi0.dims
    psi0 = qtp.Qobj(inpt=psi0_data[0], dims=dims)

    ## Change the e_ops frame if necessary

    if nonlab_original_frame and e_ops is not None:
        en_diagonal, offset_diagonal = hgen.frame_change_operator(from_frame=original_frame, to_frame='lab')

        def make_op_fn(observable):
            if isinstance(observable, qtp.Qobj):
                obs_arr = observable.full()
                def op_fn(t, state):
                    cof_op_diag = np.exp(1.j * (en_diagonal * t + offset_diagonal))
                    frame_obs = cof_op_diag[:, None] * obs_arr * cof_op_diag[None, :].conjugate()

                    return state.dag().full() @ frame_obs @ state.full()

            else:
                def op_fn(t, state):
                    cof_op_diag = np.exp(1., * (en_diagonal * t + offset_diagonal))
                    obs_arr = observable(t, identity)
                    frame_obs = cof_op_diag[:, None] * obs_arr * cof_op_diag[None, :].conjugate()

                    return state.dag().full() @ frame_obs @ state.full()

            return op_fn

        e_ops_orig = e_ops
        e_ops = []
        for observable in e_ops_orig:
            e_ops.append(make_op_fn(observable))

    ## Arguments to sesolve

    kwargs = {'args': args, 'e_ops': e_ops, 'options': options, 'progress_bar': progress_bar}

    ## Build the Hamiltonian

    if keep_callable:
        tlist_arg = dict()
    else:
        tlist_arg = {'tlist': tlist, 'args': args}

    hamiltonian = hgen.build(rwa=False, **tlist_arg)

    ## Run sesolve in a temporary directory

    logger.info('Hamiltonian with %d terms built. Starting simulation..', len(hamiltonian))

    start = time.time()

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            os.chdir(tempdir)
            qtp_result = qtp.sesolve(hamiltonian, psi0, tlist, **kwargs)
        finally:
            os.chdir(cwd)

    stop = time.time()

    logger.info('Done in %f seconds.', stop - start)

    if save_result_to:
        logger.info('Saving the simulation result to %s.qu', save_result_to)
        qtp.fileio.qsave(qtp_result, save_result_to)

    ## Bring the hgen frame back and change the frame of the states

    if qtp_result.states:
        states = np.stack(list(state.full() for state in qtp_result.states))
        states = hgen.change_frame(tlist, states, from_frame='lab', to_frame=original_frame,
                                   objtype=objtype)
    else:
        states = None

    expect = list(exp.copy() for exp in qtp_result.expect)
    dim = (hgen.num_levels,) * hgen.num_qudits
    frame_tuple = tuple(original_frame[qudit_id] for qudit_id in hgen.qudit_ids())

    return PulseSimResult(times=tlist, expect=expect, states=states, dim=dim, frame=frame_tuple)
