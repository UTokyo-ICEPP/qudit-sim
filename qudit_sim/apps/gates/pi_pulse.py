"""Calibration of a π pulse."""

import logging
from typing import Tuple
import numpy as np
import scipy.optimize as sciopt

from rqutils.math import matrix_exp, matrix_angle
import rqutils.paulis as paulis

from .components import gate_components
from ...basis import change_basis
from ...expression import Parameter
from ...hamiltonian import HamiltonianBuilder
from ...pulse import Drag
from ...pulse_sim import (build_hamiltonian, compose_parameters, simulate_drive_odeint,
                          simulate_drive_sesolve)

logger = logging.getLogger(__name__)

unit_time = 0.2e-9

def pi_pulse(
    hgen: HamiltonianBuilder,
    qudit_id: str,
    level: int,
    angle: float = np.pi,
    duration: float = unit_time * 160,
    sigma: int = unit_time * 40,
    pulse_sim_solver: str = 'qutip',
    fit_tol: float = 1.e-5,
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
        angle: Target Rx rotation angle.
        duration: DRAG pulse duration.
        sigma: DRAG pulse sigma.
        log_level: Logging level.

    Returns:
        The frequency of the drive and the DRAG pulse that implements the :math:`\pi` pulse.
    """
    original_log_level = logger.level
    logger.setLevel(log_level)

    hgen = hgen.copy(clear_drive=True)

    qudit_index = hgen.qudit_index(qudit_id)
    target_params = hgen.qudit_params(qudit_id)

    spectator_indices = tuple(iq for iq in range(hgen.num_qudits) if iq != qudit_index)
    spectator_size = np.prod(np.array(hgen.system_dim())[list(spectator_indices)])

    target_single = np.eye(target_params.num_levels, dtype=complex)
    pauli_x = paulis.paulis(2)[1]
    target_single[level:level + 2, level:level + 2] = matrix_exp(0.5j * angle * pauli_x, hermitian=-1)
    target = 1.
    for iq in range(hgen.num_qudits):
        if iq == qudit_index:
            target = np.kron(target, target_single)
        else:
            num_levels = hgen.qudit_params(hgen.qudit_id(iq)).num_levels
            target = np.kron(target, np.eye(num_levels))

    free_diags = list(range(level)) + list(range(level + 2, target_params.num_levels))

    # Pulse frequency
    drive_frequency = hgen.dressed_frequencies(qudit_id)[level]

    # Initial amplitude estimate
    # Solve exp(-i pi/2 X) = exp(-i ∫H(t)dt)
    # Approximating H(t) as a pure X triangle ((A/half_duration)*X*t in the first half),
    # ∫H(t)dt = A * duration/2 * X
    # For a resonant drive, A = drive_base * amplitude * drive_weight / 2
    # Therefore amplitude = 2pi / (duration * drive_base * drive_weight)
    amp_estimate = 2. * angle / duration / target_params.drive_amplitude
    amp_estimate /= target_params.drive_weight[level]

    # Initial beta estimate (ref. Gambetta et al. PRA 83 012308 eqn. 5.12 & text after 5.13)
    beta_estimate = 0.
    if level < target_params.num_levels - 1:
        weight_ratio = target_params.drive_weight[level + 1] / target_params.drive_weight[level]
        beta_estimate += weight_ratio ** 2 / target_params.anharmonicity
    if level > 0:
        weight_ratio = target_params.drive_weight[level - 1] / target_params.drive_weight[level]
        beta_estimate -= weight_ratio ** 2 / target_params.anharmonicity
    beta_estimate *= -1. / 4.

    amp = Parameter('amp')
    beta = Parameter('beta')
    x_pulse = Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)
    hgen.add_drive(qudit_id, frequency=drive_frequency, amplitude=x_pulse)

    if pulse_sim_solver == 'jax':
        tlist = duration
    else:
        tlist = hgen.make_tlist(points_per_cycle=10, duration=duration)

    hamiltonian = build_hamiltonian(hgen, pulse_sim_solver, {'amp': 0., 'beta': 0.})
    parameters = compose_parameters(hgen, tlist, final_only=True, reunitarize=False,
                                    solver=pulse_sim_solver)

    if pulse_sim_solver == 'jax':
        simulate_drive = simulate_drive_odeint
    else:
        simulate_drive = simulate_drive_sesolve

    icall = 0

    def fun(params):
        nonlocal icall

        logger.debug('COBYLA fun call %d: %s', icall, params)
        icall += 1

        # params are O(1) -> normalize
        drive_args = {'amp': params[0] * amp_estimate, 'beta': params[1] * beta_estimate}

        states, _ = simulate_drive(hamiltonian, parameters, drive_args)

        # Take the prod with the target and extract the diagonals
        diag_prod = np.diag(states[-1] @ target).reshape(hgen.system_dim())
        # Integrate out the non-participating qudits
        norm_ptr = np.sum(diag_prod, axis=spectator_indices) / spectator_size
        # Compute the fidelity (sum of abs-squared of commuting diagonals + abs-square of the trace
        # of target levels)
        fidelity = np.sum(np.square(np.abs(norm_ptr[free_diags])))
        fidelity += np.square(np.abs(np.sum(norm_ptr[level:level + 2]))) / 2.

        logger.debug('COBYLA diag_prod %s fidelity %f', diag_prod, fidelity)

        return -fidelity

    logger.info('Starting pi pulse identification..')

    optres = sciopt.minimize(fun, (1., 1.), method='COBYLA', tol=fit_tol, options={'rhobeg': 0.5})

    logger.info('Done after %d function calls. Final fidelity %f.', icall, -optres.fun)

    amp = optres.x[0] * amp_estimate
    beta = optres.x[1] * beta_estimate

    logger.setLevel(original_log_level)

    return drive_frequency, Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)
