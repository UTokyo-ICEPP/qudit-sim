from typing import Union

import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    array_type = np.ndarray
else:
    array_type = Union[np.ndarray, jnp.DeviceArray]

from ..utils import matrix_ufunc
from ..paulis import (get_num_paulis, make_generalized_paulis,
                      make_prod_basis, get_l0_projection)

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


def make_heff(
    heff_coeffs: array_type,
    basis_or_dim: Union[array_type, int],
    num_qubits: int = 0,
    npmod=np
) -> array_type:
    if npmod is np:
        if isinstance(basis_or_dim, int):
            basis_dim = basis_or_dim
        else:
            basis = basis_or_dim
            basis_dim = basis.shape[-1]

        if num_qubits <= 0:
            num_qubits = len(heff_coeffs.shape)
            if num_qubits > get_num_paulis(basis_dim):
                raise RuntimeError('Need to specify number of qubits for a flat-list input')

        if isinstance(basis_or_dim, int):
            paulis = make_generalized_paulis(basis_dim)
            basis = make_prod_basis(paulis, num_qubits)
    else:
        basis = basis_or_dim
            
    heff_coeffs = heff_coeffs.reshape(-1)
    basis_list = basis.reshape(-1, *basis.shape[-2:])
    return npmod.tensordot(basis_list, heff_coeffs, (0, 0)) / (2 ** (num_qubits - 1))


def make_heff_t(
    heff_coeffs: array_type,
    basis_or_dim: Union[array_type, int],
    tlist: Union[array_type, float],
    num_qubits: int = 0,
    npmod=np
) -> array_type:
    heff = make_heff(heff_coeffs, basis_or_dim, num_qubits=num_qubits, npmod=npmod)
    tlist = npmod.asarray(tlist)
    return tlist.reshape(tlist.shape + (1, 1)) * heff.reshape(tlist.shape + heff.shape)


def make_ueff(
    heff_coeffs: array_type,
    basis_or_dim: Union[array_type, int],
    tlist: Union[array_type, float] = 1.,
    num_qubits: int = 0,
    phase_factor: float = -1.,
    npmod=np
) -> array_type:
    heff_t = make_heff_t(heff_coeffs, basis_or_dim, tlist, num_qubits=num_qubits, npmod=npmod)
    return matrix_ufunc(lambda v: npmod.exp(phase_factor * 1.j * v), heff_t, hermitian=True, npmod=npmod)


def heff_fidelity(
    time_evolution: array_type,
    heff_coeffs: array_type,
    basis_or_dim: Union[array_type, int],
    tlist: array_type,
    num_qubits: int = 0,
    npmod=np
) -> array_type:
    heff_t = make_heff_t(heff_coeffs, basis_or_dim, tlist, num_qubits=num_qubits, npmod=npmod)
    ueffdag_t = matrix_ufunc(lambda v: npmod.exp(1.j * v), heff_t, hermitian=True, npmod=npmod)

    tr_u_ueffdag = npmod.trace(npmod.matmul(time_evolution, ueffdag_t), axis1=1, axis2=2)
    fidelity = (npmod.square(tr_u_ueffdag.real) + npmod.square(tr_u_ueffdag.imag)) / (ueffdag_t.shape[-1] ** 2)
    
    return fidelity


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
