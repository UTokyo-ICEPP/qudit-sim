from typing import Any, Dict, List, Sequence, Optional, Union
import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import h5py
import qutip as qtp

from .paulis import make_generalized_paulis, make_prod_basis, truncate_coefficients
from .pulse_sim import run_pulse_sim, DriveDef
from .parallel import parallel_map

logger = logging.getLogger(__name__)

def identify_gate(
    qubits: Union[Sequence[int], int],
    params: Dict[str, Any],
    drive_def: Union[List[DriveDef], DriveDef],
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
    
    logger.info('Running a pulse simulation for %d time steps', num_time_steps)
    
    if isinstance(drive_def, list):
        kwarg_keys = ('tlist',)
        kwarg_values = []
        
        for ddef in drive_def:
            tlist = np.linspace(0., _get_drive_duration(ddef), num_time_steps)
            kwarg_values.append((tlist,))
        
        if save_result_to:
            if not (os.path.exists(save_result_to) and os.path.isdir(save_result_to)):
                os.makedirs(save_result_to)
                
            kwarg_keys += ('save_result_to',)
            kwarg_values = list(v + (os.path.join(save_result_to, f'sim_{i}'),) for i, v in enumerate(kwarg_values))
        
        results = parallel_map(
            run_pulse_sim,
            args=drive_def,
            kwarg_keys=kwarg_keys,
            kwarg_values=kwarg_values,
            arg_position=2,
            common_args=(qubits, params),
            common_kwargs={'psi0': psi0, 'log_level': log_level},
            num_cpus=sim_num_cpus,
            log_level=log_level
        )

        unitary = np.empty((len(drive_def),) + psi0.shape, dtype=np.complex128)
        for ires, (time_evolution, _) in enumerate(results):
            unitary[ires] = time_evolution[-1]

    else:
        tlist = np.linspace(0., _get_drive_duration(drive_def), num_time_steps)
        
        time_evolution, _ = run_pulse_sim(
            qubits,
            params,
            drive_def,
            psi0=psi0,
            tlist=tlist,
            save_result_to=save_result_to,
            log_level=log_level)
        
        unitary = time_evolution[-1]
        
    ilogu = matrix_ufunc(lambda u: -np.angle(u), unitary)
    
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    ilogu_coeffs = np.tensordot(ilogu, basis, ((-2, -1), (-1, -2))).real / 2.
        
    ilogu_coeffs_trunc = truncate_coefficients(ilogu_coeffs, num_sim_levels, comp_dim, num_qubits)

    if save_result_to:
        if isinstance(drive_def, list):
            for idef in range(len(drive_def)):
                filename = os.path.join(save_result_to, f'heff_{idef}')
                with h5py.File(f'{filename}.h5', 'w') as out:
                    out.create_dataset('num_qubits', data=num_qubits)
                    out.create_dataset('num_sim_levels', data=num_sim_levels)
                    out.create_dataset('comp_dim', data=comp_dim)
                    out.create_dataset('unitary', data=unitary[idef])
                    out.create_dataset('tlist', data=results[idef][1])
                    if num_sim_levels != comp_dim:
                        out.create_dataset('ilogu_coeffs_original', data=ilogu_coeffs[idef])
                    out.create_dataset('ilogu_coeffs', data=iogu_coeffs_trunc[idef])
                    
        else:
            with h5py.File(f'{save_result_to}.h5', 'w') as out:
                out.create_dataset('num_qubits', data=num_qubits)
                out.create_dataset('num_sim_levels', data=num_sim_levels)
                out.create_dataset('comp_dim', data=comp_dim)
                out.create_dataset('unitary', data=unitary)
                out.create_dataset('tlist', data=tlist)
                if num_sim_levels != comp_dim:
                    out.create_dataset('ilogu_coeffs_original', data=ilogu_coeffs)
                out.create_dataset('ilogu_coeffs', data=iogu_coeffs_trunc)
                out.create_dataset('heff_coeffs', data=heff_coeffs_trunc)
                
    logger.setLevel(original_log_level)
    
    return ilogu_coeffs_trunc

def _get_drive_duration(ddef):
    duration = 0.
    for key, value in ddef.values():
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