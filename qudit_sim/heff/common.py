from typing import Union

import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    array_type = np.ndarray
else:
    array_type = Union[np.ndarray, jnp.DeviceArray]

from rqutils.math import matrix_exp, matrix_angle
import rqutils.paulis as paulis

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
    heff_compos: array_type,
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
            num_qubits = len(heff_compos.shape)
            if num_qubits > basis_dim ** 2:
                raise RuntimeError('Need to specify number of qubits for a flat-list input')

        if isinstance(basis_or_dim, int):
            basis = paulis.paulis((basis_dim,) * num_qubits)
    else:
        basis = basis_or_dim
            
    heff_compos = heff_compos.reshape(-1)
    basis_list = basis.reshape(-1, *basis.shape[-2:])
    return npmod.tensordot(basis_list, heff_compos, (0, 0))


def make_heff_t(
    heff: array_type,
    tlist: Union[array_type, float],
    npmod=np
) -> array_type:
    tlist = npmod.asarray(tlist)
    tdims = (1,) * len(tlist.shape)
    return tlist[..., None, None] * heff.reshape(tdims + heff.shape)


def compose_ueff(
    heff_compos: array_type,
    basis_list: array_type,
    tlist: Union[array_type, float] = 1.,
    phase_factor: float = -1.,
    npmod=np
) -> array_type:
    basis_list = basis_list.reshape(-1, *basis_list.shape[-2:])
    heff_compos = heff_compos.reshape(-1)
    heff = npmod.tensordot(basis_list, heff_compos, (0, 0))

    heff_t = make_heff_t(heff, tlist, npmod=npmod)
    
    return matrix_exp(phase_factor * 1.j * heff_t, hermitian=-1, npmod=npmod)


def heff_fidelity(
    time_evolution: array_type,
    heff_compos: array_type,
    basis_or_dim: Union[array_type, int],
    tlist: array_type,
    num_qubits: int = 0,
    npmod=np
) -> array_type:
    heff_t = make_heff_t(heff_compos, basis_or_dim, tlist, num_qubits=num_qubits, npmod=npmod)
    ueffdag_t = matrix_ufunc(lambda v: npmod.exp(1.j * v), heff_t, hermitian=True, npmod=npmod)

    tr_u_ueffdag = npmod.trace(npmod.matmul(time_evolution, ueffdag_t), axis1=1, axis2=2)
    fidelity = (npmod.square(tr_u_ueffdag.real) + npmod.square(tr_u_ueffdag.imag)) / (ueffdag_t.shape[-1] ** 2)
    
    return fidelity
