from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import csr_array
import qutip as qtp

import rqutils.paulis as paulis
from .find_heff import add_drive_for_heff
from ...hamiltonian import HamiltonianBuilder
from ...frame import FrameSpec

Array = Union[np.ndarray, csr_array]
QObject = Union[qtp.Qobj, Array, Tuple[Array, ...], Dict[str, Array]]
QuditSpec = Union[str, Tuple[str, ...]]
FrequencySpec = Union[float, Tuple[float, ...]]
AmplitudeSpec = Union[float, complex, Tuple[Union[float, complex], ...]]

def hamiltonian_tomography(
    hgen: HamiltonianBuilder,
    qudit: QuditSpec,
    frequency: FrequencySpec,
    amplitude: AmplitudeSpec,
    meas_qubit: str,
    psi0: QObject,
    pulse_shape: Tuple[float, float, float],
    num_points: int,
    frame: FrameSpec = 'dressed',
    pulse_sim_solver: str = 'jax'
):
    original_log_level = logger.level
    logger.setLevel(log_level)

    hgen_drv, tlist, drive_args, time_range = add_drive_for_heff(hgen, qudit, frequency, amplitude,
                                                                 pulse_shape, use_cycles=False,
                                                                 num_flattop_points=num_points)

    e_ops = list({meas_qubit: pauli}
                 for pauli in paulis.paulis(hgen.qudit_params(meas_qubit).num_levels)[1:4])

    sim_result = pulse_sim(hgen_drv, tlist, psi0=psi0, e_ops=e_ops, drive_args=drive_args, frame=frame,
                           solver=pulse_sim_solver, save_result_to=save_result_to,
                           log_level=log_level)
