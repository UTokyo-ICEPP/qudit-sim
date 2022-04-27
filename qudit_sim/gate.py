from typing import Any, Dict, List, Sequence, Optional, Union
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import qutip as qtp

from rqutils.paulis as paulis
from rqutils.math import matrix_angle
from rqutils.qprint import QPrintPauli

from .pulse_sim import run_pulse_sim, DriveDef
from .parallel import parallel_map

logger = logging.getLogger(__name__)

def identify_gate(
    qubits: Union[Sequence[int], int],
    params: Dict[str, Any],
    drive_def: Union[List[DriveDef], DriveDef],
    phase_offsets: Optional[Dict[int, float]] = None,
    num_sim_levels: int = 2,
    num_time_steps: int = 100,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None,
    sim_num_cpus: int = 0,
    log_level: int = logging.WARNING
) -> np.ndarray:
    """Run the pulse simulation and identify the resulting unitary."""
    
    original_log_level = logger.level
    logger.setLevel(log_level)
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)
    
    if phase_offsets is not None:
        phase_offset_array = np.zeros(num_qubits, dtype=np.float)
        for iq, qubit in enumerate(qubits):
            try:
                phase_offset_array[iq] = phase_offsets[qubit]
            except KeyError:
                pass
    
    logger.info('Running a pulse simulation for %d time steps', num_time_steps)
    
    if isinstance(drive_def, list):
        num_jobs = len(drive_def)
        
        kwarg_keys = ('tlist',)
        kwarg_values_tuple = ([],)
        
        for ddef in drive_def:
            tlist = np.linspace(0., _get_drive_duration(ddef), num_time_steps)
            kwarg_values_tuple[0].append(tlist)
            
        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)
                
            kwarg_keys += ('save_result_to',)
            kwarg_values_tuple +=  list(os.path.join(save_result_to, f'sim_{i}') for i in range(num_jobs))
            
        kwarg_values = list(tuple(t[i] for t in kwarg_values_tuple) for i in range(num_jobs))
        
        common_args = (qubits, params)
        common_kwargs = {'phase_offsets': phase_offsets, 'psi0': psi0, 'log_level': log_level}
        
        results = parallel_map(
            run_pulse_sim,
            args=drive_def,
            kwarg_keys=kwarg_keys,
            kwarg_values=kwarg_values,
            arg_position=2,
            common_args=common_args,
            common_kwargs=common_kwargs,
            num_cpus=sim_num_cpus,
            log_level=log_level
        )

        unitary = np.stack(list(result.states[-1] for result in results))

    else:
        tlist = np.linspace(0., _get_drive_duration(drive_def), num_time_steps)
        
        result = run_pulse_sim(
            qubits,
            params,
            drive_def,
            phase_offsets=phase_offsets,
            psi0=psi0,
            tlist=tlist,
            #save_result_to=save_result_to,
            log_level=log_level)
        
        unitary = result.states[-1]
        
    ilogu = matrix_ufunc(lambda u: -np.angle(u), unitary)
    ilogu_compos = paulis.components(ilogu, (num_sim_levels,) * num_qubits)
    ilogu_compos_trunc = paulis.truncate(ilogu_compos, (comp_dim,) * num_qubits)
    
    if phase_offsets is not None:
        for iq, offset in enumerate(phase_offset_array):
            dim = iq + 1 if isinstance(drive_def, list) else iq
            ilogu_compos_trunc = shift_phase(ilogu_compos_trunc, offset, dim=dim)

    if save_result_to:
        if isinstance(drive_def, list):
            for idef, result in enumerate(results):
                filename = os.path.join(save_result_to, f'heff_{idef}')
                with h5py.File(f'{filename}.h5', 'w') as out:
                    out.create_dataset('num_qubits', data=num_qubits)
                    out.create_dataset('num_sim_levels', data=num_sim_levels)
                    out.create_dataset('comp_dim', data=comp_dim)
                    out.create_dataset('time_evolution', data=result.states)
                    out.create_dataset('tlist', data=result.times)
                    if num_sim_levels != comp_dim:
                        out.create_dataset('ilogu_compos_original', data=ilogu_compos[idef])
                    out.create_dataset('ilogu_compos', data=iogu_compos_trunc[idef])
                    
        else:
            with h5py.File(f'{save_result_to}.h5', 'w') as out:
                out.create_dataset('num_qubits', data=num_qubits)
                out.create_dataset('num_sim_levels', data=num_sim_levels)
                out.create_dataset('comp_dim', data=comp_dim)
                out.create_dataset('time_evolution', data=result.states)
                out.create_dataset('tlist', data=result.times)
                if num_sim_levels != comp_dim:
                    out.create_dataset('ilogu_compos_original', data=ilogu_compos)
                out.create_dataset('ilogu_compos', data=ilogu_compos_trunc)
                
    logger.setLevel(original_log_level)
    
    return ilogu_compos_trunc

def _get_drive_duration(ddef):
    duration = 0.
    for key, value in ddef.items():
        if key == 'args':
            continue

        try:
            drive_end = value['start'] + value['duration']
        except KeyError:
            envelope = value['amplitude']
            try:
                drive_end = envelope.end
            except AttributeError:
                raise RuntimeError(f'Unknown end time for drive {key}: {value}')

        duration = max(duration, drive_end)

    return duration


def gate_expr(
    components: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = 0.01
) -> str:
    hamiltonian = QPrintPauli(components, symbol=symbol, epsilon=threshold)
    return fr'\exp\left[-i \left({hamiltonian}\right)\right]'
