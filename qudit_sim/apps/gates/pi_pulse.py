"""Calibrate a π pulse."""

from typing import Hashable, Tuple
import logging
import numpy as np
import scipy.optimize as sciopt
import qutip as qtp

import rqutils.paulis as paulis
from rqutils.math import matrix_angle

from ...hamiltonian import HamiltonianBuilder
from ...pulse import Drag
from ...pulse_sim import pulse_sim
from ...basis import change_basis
from .components import gate_components

logger = logging.getLogger(__name__)

unit_time = 0.2e-9

def pi_pulse(
    hgen: HamiltonianBuilder,
    qudit_id: Hashable,
    level: int,
    duration: float = unit_time * 160,
    sigma: int = unit_time * 40,
    log_level: int = logging.WARNING
) -> Tuple[float, Drag]:
    r"""Find the :math:`\pi` pulse for the given level of the given qudit by numerical optimization.

    This function finds a DRAG pulse with the specified duration and sigma that minimizes the Euclidean
    difference between the components array of the final state and the ideal :math:`\pi` gate. The target
    array is zero everywhere except for the single component corresponding to the "X" that excites ``level``
    of the ``qudit_id`` to ``level+1``. The final state is allowed to have arbitrary phase shifts in the
    unaddressed subspace.

    Targeting of a specific level is implemented using the ``'pauli<n>'`` basis, which cyclically shifts
    the "core" subspace of the generalized Gell-Mann matrices. To find the :math:`\pi` pulse for level ``j``,
    the Pauli components of the generator of the pulse gate is interpreted in the ``paulij`` basis, in
    which the components target generator reads (*, :math:`\pi/2`, 0, 0, 0, 0, 0, 0, *, 0, ...) where *
    indicates that the component is unconstrained because the corresponding basis matrix is diagonal and
    commutes with the target "X".

    Args:
        hgen: Hamiltonian of the system.
        qudit_id: ID of the qudit to find the :math:`\pi` pulse for.
        level: Index of the lower level in the transition.
        duration: DRAG pulse duration.
        sigma: DRAG pulse sigma.
        log_level: Logging level.

    Returns:
        The frequency of the drive and the DRAG pulse that implements the :math:`\pi` pulse.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    hgen = hgen.copy(clear_drive=True)

    components_shape = (hgen.num_levels ** 2,) * hgen.num_qudits
    qudit_index = hgen.qudit_index(qudit_id)

    # The target - Pulse is optimized to minimize the distance of the gate generator components to this array
    target = np.zeros(components_shape)
    target_index = tuple(1 if i == qudit_index else 0 for i in range(hgen.num_qudits))
    target[target_index] = np.pi / 2. * np.power(2. * hgen.num_levels, (hgen.num_qudits - 1) / 2.)

    # Components mask - We unconstrain the commuting diagonals of the gate
    components_mask = np.ones(components_shape)
    diag_indices = [0] + list(np.square(np.arange(3, hgen.num_levels)) - 1)
    target_indices = tuple(diag_indices if i == qudit_index else np.zeros_like(diag_indices)
                           for i in range(hgen.num_qudits))
    components_mask[target_indices] = 0

    # Pulse frequency
    drive_frequency = hgen.dressed_frequencies(qudit_id)[level]

    # Initial amplitude estimate
    # Solve exp(-i pi/2 X) = exp(-i ∫H(t)dt)
    # Approximating H(t) as a pure X triangle ((A/half_duration)*X*t in the first half),
    # ∫H(t)dt = A * duration/2 * X
    # For a resonant drive, A = drive_base * amplitude * drive_weight / 2
    # Therefore amplitude = 2pi / (duration * drive_base * drive_weight)
    params = hgen.qudit_params(qudit_id)
    amp_estimate = 2. * np.pi / duration / params.drive_amplitude / params.drive_weight[level]

    # Initial beta estimate (ref. Gambetta et al. PRA 83 012308 eqn. 5.12 & text after 5.13)
    beta_estimate = 0.
    if level < hgen.num_levels - 1:
        beta_estimate += (params.drive_weight[level + 1] / params.drive_weight[level]) ** 2 / params.anharmonicity
    if level > 0:
        beta_estimate -= (params.drive_weight[level - 1] / params.drive_weight[level]) ** 2 / params.anharmonicity
    beta_estimate *= -1. / 4.

    # Target for gate_components (must be in pauli0 basis)
    target_pauli0 = change_basis(target, to_basis='pauli0', from_basis=f'pauli{level}')

    icall = 0

    def fun(params):
        nonlocal icall

        logger.debug('COBYLA fun call %d: %s', icall, params)
        icall += 1

        # params are O(1) -> normalize
        amp = params[0] * amp_estimate
        beta = params[1] * beta_estimate
        pulse = Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)

        hgen.clear_drive()
        hgen.add_drive(qudit_id, frequency=drive_frequency, amplitude=pulse)

        sim_result = pulse_sim(hgen,
                               tlist={'points_per_cycle': 10, 'duration': pulse.duration},
                               final_only=True)

        components = change_basis(gate_components(sim_result, target_pauli0), to_basis=f'pauli{level}')
        components *= components_mask

        return np.sum(np.square(components - target))

    logger.info('Starting pi pulse identification..')

    optres = sciopt.minimize(fun, (1., 1.), method='COBYLA', tol=5.e-5, options={'rhobeg': 0.5})

    logger.info('Done after %d function calls', icall)

    amp = optres.x[0] * amp_estimate
    beta = optres.x[1] * beta_estimate

    logger.setLevel(original_log_level)

    return drive_frequency, Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)
