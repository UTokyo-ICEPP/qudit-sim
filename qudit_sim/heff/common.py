import numpy as np

from ..utils import matrix_ufunc
from ..paulis import get_num_paulis, get_l0_projection

def get_ilogus_and_valid_it(unitaries):
    ## Compute ilog(U(t))
    ilogus, ilogvs = matrix_ufunc(lambda u: -np.angle(u), unitaries, with_diagonals=True)

    ## Find the first t where an eigenvalue does a 2pi jump
    last_valid_it = ilogus.shape[0]
    for ilogv_ext in [np.amin(ilogvs, axis=1), -np.amax(ilogvs, axis=1)]:
        margin = 0.1
        hits_minus_pi = np.asarray(ilogv_ext < -np.pi + margin).nonzero()[0]
        if len(hits_minus_pi) != 0:
            last_valid_it = min(last_valid_it, hits_minus_pi[0])
            
    return ilogus, ilogvs, last_valid_it


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
