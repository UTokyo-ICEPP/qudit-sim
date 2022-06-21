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

    ## System parameters
    qudit_index = hgen.qudit_index(qudit_id)
    dim = (hgen.num_levels,) * hgen.num_qudits
    matrix_dim = dim[0]

    # We need to perform a change of basis among the diagonal Pauli components in order to identiy the
    # "Z" operator for the relevant subspace while masking out the orthogonal operators.
    basis = paulis.paulis(dim)
    if level > 0:
        num_paulis = matrix_dim ** 2
        num_diagonal_paulis = matrix_dim

        # Separate the single-qudit Pauli set to diagonal and offdiagonal elements
        qudit_basis = paulis.paulis(matrix_dim)
        symmetry = paulis.symmetry(matrix_dim)
        diagonal_paulis = qudit_basis[np.nonzero(symmetry == 0)]
        offdiagonal_paulis = qudit_basis[np.nonzero(symmetry)]

        # Roll the diagonal -> diagonal_paulis[1] is now the Z of the desired subspace
        diagonals = np.diagonal(diagonal_paulis, axis1=-2, axis2=-1)
        diagonals = np.roll(diagonals, level, axis=-1)
        # Trick for making an array of diagonal matrices: construct Nx(N+1) matrices with the first column filled
        diagonal_paulis = np.concatenate((diagonals[..., None], np.zeros_like(diagonal_paulis)), axis=-1)
        diagonal_paulis = diagonal_paulis.reshape(num_diagonal_paulis, -1)[:, :matrix_dim ** 2]
        diagonal_paulis = diagonal_paulis.reshape(-1, matrix_dim, matrix_dim)

        # Redefine qudit_basis and obtain the change-of-basis matrix (of Pauli components)
        qudit_basis[np.nonzero(symmetry == 0)] = diagonal_paulis
        change_of_basis = paulis.components(qudit_basis, dim=matrix_dim)

        # tensordot(.., (1, qudit_index)) will place the change-of-basis axis at 0
        # -> tensordot on the first axis and then move it to the correct position
        basis = np.moveaxis(np.tensordot(change_of_basis, basis, (1, 0)), 0, qudit_index)

    # The third component at qudit_index of the product-trace with basis is the Z component of the desired subspace

    ## Define the target components array
    components_mask = np.ones(np.square(dim))
    is_diagonal = (paulis.symmetry(matrix_dim) == 0)
    is_diagonal[3] = False
    diagonals = np.nonzero(is_diagonal)
    diagonals += (np.zeros_like(diagonals[0]),) * (hgen.num_qudits - 1)
    components_mask[diagonals] = 0.
    components_mask = np.moveaxis(components_mask, 0, qudit_index)

    # The "X" Pauli is the third from the last in the list of Paulis for the given level
    resonant_component = (level + 2) ** 2 - 3
    target_component_value = np.pi / 2. * np.power(2. * hgen.num_levels, (hgen.num_qudits - 1) / 2.)
    target_components = np.zeros(np.square(dim))
    target_index = [0] * hgen.num_qudits
    target_index[qudit_index] = resonant_component
    target_components[tuple(target_index)] = target_component_value

    ## Initialize the Hamiltonian
    drive_frequency = hgen.dressed_frequencies(qudit_id)[level]

    ## Make the tlist
    hgen.add_drive(qudit_id, frequency=drive_frequency, amplitude=1.)
    tlist = hgen.make_tlist(10, duration=duration)
    hgen.clear_drive()

    ## Get an initial estimate of the amplitude
    # Solve exp(-i pi/2 X) = exp(-i ∫H(t)dt)
    # Approximating H(t) as a pure X triangle ((A/half_duration)*X*t in the first half),
    # ∫H(t)dt = A * duration/2 * X
    # For a resonant drive, A = drive_base * amplitude * sqrt(level+1) / 2
    # Therefore amplitude = 2pi / (duration * drive_base * sqrt(level+1))
    drive_amplitude = hgen.qudit_params(qudit_id).drive_amplitude
    rough_amp_estimate = 2. * np.pi / duration / drive_amplitude / np.sqrt(level + 1)

    icall = 0

    def fun(params):
        nonlocal icall

        logger.debug('COBYLA fun call %d: %s', icall, params)
        icall += 1

        # params are O(1) -> normalize
        amp = params[0] * rough_amp_estimate
        beta = params[1] * sigma
        pulse = Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)

        hgen.clear_drive()
        hgen.add_drive(qudit_id, frequency=drive_frequency, amplitude=pulse)

        sim_result = pulse_sim(hgen, tlist, final_only=True)

        # Need to compute the components "by hand" because our basis definition may be nonstandard
        generator = -matrix_angle(sim_result.states[-1])
        components = np.trace(basis @ generator, axis1=-2, axis2=-1).real * (2 ** (hgen.num_qudits - 2))
        components *= components_mask

        return np.sum(np.square(components - target_components))

    logger.info('Starting pi pulse identification..')

    optres = sciopt.minimize(fun, (1., 0.), method='COBYLA', options={'rhobeg': 0.1})

    logger.info('Done after %d function calls', icall)

    amp = optres.x[0] * rough_amp_estimate
    beta = optres.x[1] * sigma

    logger.setLevel(original_log_level)

    return drive_frequency, Drag(duration=duration, amp=amp, sigma=sigma, beta=beta)
