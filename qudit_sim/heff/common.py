import numpy as np

from ..paulis import get_num_paulis, get_l0_projection

def truncate_heff(
    heff_coeffs: np.ndarray,
    num_sim_levels: int,
    comp_dim: int,
    num_qubits: int
) -> np.ndarray:
    if comp_dim == num_sim_levels:
        return heff_coeffs

    ## Truncate the hamiltonian to the computational subspace
    l0_projection = get_l0_projection(comp_dim, num_sim_levels)

    for iq in range(num_qubits):
        indices = [slice(None)] * num_qubits
        indices[iq] = 0
        indices = tuple(indices)

        heff_coeffs[indices] = np.tensordot(heff_coeffs, l0_projection, (iq, 0))

    num_comp_paulis = get_num_paulis(comp_dim)
    return heff_coeffs[(slice(num_comp_paulis),) * num_qubits]
    