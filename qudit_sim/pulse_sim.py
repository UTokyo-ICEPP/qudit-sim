from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import os
import tempfile
import logging
import time
import copy
import collections

import numpy as np
import qutip as qtp

from .hamiltonian import RWAHamiltonianGenerator

logger = logging.getLogger(__name__)

DriveDef = Dict[Union[int, str], Dict[str, Any]]

PulseSimResult = collections.namedtuple('PulseSimResult', ['times', 'expect', 'states'])

def run_pulse_sim(
    qubits: Union[Sequence[int], int],
    params: Dict[str, Any],
    drive_def: Dict,
    phase_offsets: Optional[Dict[int, float]] = None,
    psi0: qtp.Qobj = qtp.basis(2, 0),
    tlist: Union[np.ndarray, Tuple[int, int]] = (10, 100),
    force_array: bool = False,
    e_ops: Optional[Sequence[Any]] = None,
    options: Optional[qtp.solver.Options] = None,
    progress_bar: Optional[qtp.ui.progressbar.BaseProgressBar] = None,
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> PulseSimResult:
    """Run a pulse simulation.
    
    Sets up an RWAHamiltonianGenerator object from the given parameters, determine the time points for the simulation
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
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of RWAHamiltonian for details.
        drive_def: Drive definition dict. Keys are qubit ids and values are dicts of format
            `{'frequency': frequency, 'amplitude': amplitude}`. Can optionally include a key `'args'` where the
            corresponding value is then passed to the `args` argument of `sesolve`.
        phase_offsets: Model of phase offsets between the room temperature electronics and the qudit system. Drive
            signal is sent to the qudits with the given offsets.
        psi0: Initial state Qobj.
        tlist: Time points to use in the simulation or a pair `(points_per_cycle, num_cycles)` where in the latter
            case the cycle of the fastest oscillating term in the Hamiltonian will be used.
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
    
    if isinstance(qubits, int):
        qubits = (qubits,)
    
    ## Collect kwargs passed directly to sesolve
    
    kwargs = {'e_ops': e_ops, 'options': options, 'progress_bar': progress_bar}
    
    ## Make the Hamiltonian
    
    num_sim_levels = psi0.dims[0][0]
    
    if phase_offsets is not None:
        params = copy.deepcopy(params)
        
        for q, offset in phase_offsets.items():
            params[f'omegad{q}'] *= np.exp(1.j * offset)
    
    # When using the array Hamiltonian, Hint terms should be kept as python functions which
    # yield the arrays upon calling array_hamiltonian().
    hgen = RWAHamiltonianGenerator(qubits, params, num_sim_levels, compile_hint=(not force_array))
    
    logger.info('Instantiated a Hamiltonian generator for %d qubits and %d levels', len(qubits), num_sim_levels)
    logger.info('Number of interaction terms: %d', len(hgen.hint))
    
    for key, value in drive_def.items():
        if key == 'args':
            kwargs['args'] = value
            continue
            
        logger.info('Adding a drive with frequency %f and envelope %s', value['frequency'], value['amplitude'])

        hgen.add_drive(key, frequency=value['frequency'], amplitude=value['amplitude'])
        
    logger.info('Number of drive terms: %d', len(hgen.hdrive))

    if isinstance(tlist, tuple):
        tlist = hgen.make_tlist(*tlist)
        
    logger.info('Using %d time points from %.3e to %.3e', tlist.shape[0], tlist[0], tlist[-1])

    if force_array or hgen.need_tlist:
        hamiltonian = hgen.array_generate(tlist)
    else:
        hamiltonian = hgen.generate()
        
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
        
    logger.setLevel(original_log_level)
    
    if qtp_result.states:
        states = np.stack(list(state.full() for state in qtp_result.states))
    else:
        states = None
    
    return PulseSimResult(times=qtp_result.times, expect=qtp_result.expect, states=states)
