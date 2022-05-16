"""Pulse simulation frontend."""

from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import os
import tempfile
import logging
import time
import copy
import collections

import numpy as np
import qutip as qtp

from .hamiltonian import HamiltonianGenerator
from .util import PulseSimResult
from .parallel import parallel_map

logger = logging.getLogger(__name__)

def pulse_sim(
    hgen: Union[HamiltonianGenerator, List[HamiltonianGenerator]],
    psi0: Optional[qtp.Qobj] = None,
    tlist: Union[np.ndarray, Tuple[int, int]] = (10, 100),
    args: Optional[Any] = None,
    rwa: bool = True,
    keep_callable: bool = False,
    e_ops: Optional[Sequence[Any]] = None,
    options: Optional[qtp.solver.Options] = None,
    progress_bar: Optional[qtp.ui.progressbar.BaseProgressBar] = None,
    save_result_to: Optional[str] = None,
    num_cpus: int = 0,
    log_level: int = logging.WARNING
) -> Union[PulseSimResult, List[PulseSimResult]]:
    """Run a pulse simulation.

    Generate the Hamiltonian terms from the HamiltonianGenerator, determine the time points for the simulation
    if necessary, and run `qutip.sesolve`.

    .. rubric:: Implementation notes (why we return an original object instead of the QuTiP result)

    When the coefficients of the time-dependent Hamiltonian are compiled (preferred
    method), QuTiP creates a transient python module with file name generated from the code hash, PID, and the current time.
    When running multiple simulations in parallel this is not strictly safe, and so we enclose `sesolve` in a context with
    a temporary directory in this function. The transient module is then deleted at the end of execution, but that in turn
    causes an error when this function is called in a subprocess and if we try to return the QuTiP result object directly
    through e.g. multiprocessing.Pipe. Somehow the result object tries to carry with it something defined in the transient
    module, which would therefore need to be pickled together with the returned object. But the transient module file is
    gone by the time the parent process receives the result from the pipe.
    So, the solution was to just return a "sanitized" object, consisting of plain ndarrays.

    Args:
        hgen: Fully set up Hamiltonian generator or a list thereof.
        psi0: Initial state Qobj. Defaults to the identity operator appropriate for the given Hamiltonian.
        tlist: Time points to use in the simulation or a pair `(points_per_cycle, num_cycles)` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
        args: Second parameter passed to drive amplitude functions (if callable).
        rwa: Whether to use the rotating-wave approximation.
        keep_callable: Keep callable time-dependent Hamiltonian coefficients. Otherwise all callable coefficients
            are converted to arrays before simulation execution for efficiency (no loss of accuracy observed so far).
        e_ops: List of observables passed to the QuTiP solver.
        options: QuTiP solver options.
        progress_bar: QuTiP progress bar.
        save_result_to: File name (without the extension) to save the simulation result to.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    ## kwargs passed directly to sesolve

    kwargs = {'e_ops': e_ops, 'options': options, 'progress_bar': progress_bar}
    if args is not None:
        kwargs['args'] = args

    if isinstance(hgen, list):
        common_kwargs = {'psi0': psi0, 'tlist': tlist, 'rwa': rwa, 'keep_callable': keep_callable, 'kwargs': kwargs}

        num_tasks = len(hgen)

        kwarg_keys = ('logger_name',)
        kwarg_values = list((f'{__name__}.{itask}',) for itask in range(num_tasks))

        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)

            kwarg_keys += ('save_result_to',)
            for itask in range(num_tasks):
                kwarg_values[itask] += (os.path.join(save_result_to, f'sim_{itask}'),)

        result = parallel_map(_run_single, args=hgen, kwarg_keys=kwarg_keys, kwarg_values=kwarg_values,
                              common_kwargs=common_kwargs, num_cpus=num_cpus, log_level=log_level)

    else:
        result = _run_single(hgen, psi0, tlist, rwa, keep_callable, kwargs, save_result_to)

    logger.setLevel(original_log_level)

    return result


def _run_single(
    hgen: HamiltonianGenerator,
    psi0: Union[qtp.Qobj, None],
    tlist: Union[np.ndarray, Tuple[int, int]],
    rwa: bool,
    keep_callable: bool,
    kwargs: dict,
    save_result_to: Optional[str] = None,
    logger_name: str = __name__
):
    """Run one pulse simulation."""
    logger = logging.getLogger(logger_name)

    ## Define the initial state if necessary

    if psi0 is None:
        psi0 = qtp.tensor([qtp.qeye(hgen.num_levels)] * hgen.num_qudits)

    ## Define the time points if necessary

    if isinstance(tlist, tuple):
        # Need to generate Hint and Hdrive once to get the max frequencies
        hgen.generate_hint()
        hgen.generate_hdrive(rwa=rwa)
        tlist = hgen.make_tlist(*tlist)

    logger.info('Using %d time points from %.3e to %.3e', tlist.shape[0], tlist[0], tlist[-1])

    ## Generate the Hamiltonian

    if keep_callable:
        tlist_arg = dict()
    else:
        tlist_arg = {'tlist': tlist, 'args': kwargs.get('args')}

    hamiltonian = hgen.generate(rwa=rwa, **tlist_arg)

    ## Run sesolve in a temporary directory

    logger.info('Hamiltonian with %d terms generated. Starting simulation..', len(hamiltonian))

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

    if qtp_result.states:
        states = np.stack(list(state.full() for state in qtp_result.states))
    else:
        states = None

    expect = list(exp.copy() for exp in qtp_result.expect)
    dim = (hgen.num_levels,) * hgen.num_qudits

    return PulseSimResult(times=tlist, expect=expect, states=states, dim=dim)
