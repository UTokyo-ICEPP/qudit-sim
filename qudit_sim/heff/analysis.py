import numpy as np
import rqutils.paulis as paulis

from .common import heff_fidelity

def fidelity_loss(
    time_evolution: np.ndarray,
    components: np.ndarray,
    tlist: np.ndarray
) -> np.ndarray:
    """Compute the loss in the mean fidelity when the components are set to zero.

    Args:
        time_evolution: Time evolution unitary as a function of time.
        components: Pauli components.
        tlist: Time points.

    Returns:
        An array with the same shape as components, with elements corresponding to the loss in mean
        fidelity when the respective components are set to zero.
    """
    pauli_dim = np.around(np.sqrt(components.shape)).astype(int)
    basis = paulis.paulis(pauli_dim)

    best_fidelity = np.mean(heff_fidelity(states, components, basis, tlist))

    fid_loss = np.zeros_like(components)

    for idx in np.ndindex(components.shape):
        test_compos = components.copy()
        test_compos[idx] = 0.
        test_fidelity = np.mean(heff_fidelity(states, test_compos, basis, tlist))

        fid_loss[idx] = best_fidelity - test_fidelity

    return fid_loss
