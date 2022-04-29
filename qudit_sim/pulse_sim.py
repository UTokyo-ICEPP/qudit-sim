from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import os
import tempfile
import logging
import time
import copy
import collections

import numpy as np
import qutip as qtp

import rqutils.paulis as paulis

from .hamiltonian import HamiltonianGenerator

logger = logging.getLogger(__name__)

DriveDef = Dict[Union[int, str], Dict[str, Any]]

PulseSimResult = collections.namedtuple('PulseSimResult', ['times', 'expect', 'states'])

def run_pulse_sim(
    hgen: HamiltonianGenerator,
    psi0: qtp.Qobj = qtp.basis(2, 0),
    tlist: Union[np.ndarray, Tuple[int, int]] = (10, 100),
    args: Optional[Any] = None,
    rwa: bool = True,
    force_array: bool = False,
    e_ops: Optional[Sequence[Any]] = None,
    options: Optional[qtp.solver.Options] = None,
    progress_bar: Optional[qtp.ui.progressbar.BaseProgressBar] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> PulseSimResult:
    """Run a pulse simulation.
    
    Generates the Hamiltonian terms from the HamiltonianGenerator, determine the time points for the simulation
    if necessary, and run `qutip.sesolve`.
    
    ** Implementation notes (why we return an original object instead of the QuTiP result) **
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
        hgen: Fully set up Hamiltonian generator.
        psi0: Initial state Qobj.
        tlist: Time points to use in the simulation or a pair `(points_per_cycle, num_cycles)` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
        args: Second parameter passed to drive amplitude functions (if callable).
        rwa: Whether to use the rotating-wave approximation.
        force_array: Use an array-based Hamiltonian for the simulation. May run faster but potentially give
            inaccurate results, depending on the Hamiltonian.
        e_ops: List of observables passed to the QuTiP solver.
        options: QuTiP solver options.
        progress_bar: QuTiP progress bar.
        save_result_to: File name (without the extension) to save the simulation result to.

    Returns:
        Result of the pulse simulation.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    assert len(psi0.dims[0]) == hgen.num_qudits
    
    ## Make the Hamiltonian
    
    num_sim_levels = psi0.dims[0][0]
    
    ## kwargs passed directly to sesolve
    
    kwargs = {'e_ops': e_ops, 'options': options, 'progress_bar': progress_bar}
    if args is not None:
        kwargs['args'] = args
        
    ## Define the time points if necessary

    if isinstance(tlist, tuple):
        tlist = hgen.make_tlist(*tlist)
        
    logger.info('Using %d time points from %.3e to %.3e', tlist.shape[0], tlist[0], tlist[-1])
    
    ## Generate the Hamiltonian

    if force_array or hgen.need_tlist:
        tlist_arg = {'tlist': tlist, 'args': args}
    else:
        tlist_arg = dict()
        
    hamiltonian = hgen.generate(rwa=rwa, compile_hint=True, **tlist_arg)
    
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
        
    logger.setLevel(original_log_level)
    
    return PulseSimResult(times=tlist, expect=expect, states=states)
