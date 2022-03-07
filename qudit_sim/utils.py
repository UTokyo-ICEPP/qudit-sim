from typing import Callable, Union

import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    array_type = np.ndarray
else:
    array_type = Union[np.ndarray, jnp.DeviceArray]

def matrix_ufunc(
    op: Callable,
    mat: array_type,
    hermitian: bool = False,
    with_diagonals: bool = False,
    npmod=np
) -> np.ndarray:
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.
    
    The argument `mat` must be an array of normal matrices (in the last two dimensions). This function
    unitary-diagonalizes the matrices, applies `op` to the diagonals, and inverts the diagonalization.
    
    Args:
        op: Unary operator to be applied to the diagonals of `mat`.
        mat: Array of normal matrices (shape (..., n, n)). No check on normality is performed.
        with_diagonals: If True, also return the array `op(eigenvalues)`.

    Returns:
        An array corresponding to `op(mat)`. If `diagonals==True`, another array corresponding to `op(eigvals)`.
    """
    if hermitian:
        eigvals, eigcols = npmod.linalg.eigh(mat)
    else:
        eigvals, eigcols = npmod.linalg.eig(mat)
        
    eigrows = npmod.conjugate(npmod.moveaxis(eigcols, -2, -1))

    op_eigvals = op(eigvals)
    
    op_mat = npmod.matmul(eigcols * op_eigvals[..., None, :], eigrows)

    if with_diagonals:
        return op_mat, op_eigvals
    else:
        return op_mat
    

def make_heff(
    heff_coeffs: array_type,
    basis: array_type,
    num_qubits: int,
    npmod=np
) -> array_type:
    heff_coeffs = heff_coeffs.reshape(-1)
    basis_list = basis.reshape(-1, *basis.shape[-2:])
    return npmod.tensordot(basis_list, heff_coeffs, (0, 0)) / (2 ** (num_qubits - 1))


def make_heff_t(
    heff_coeffs: array_type,
    basis: array_type,
    num_qubits: int,
    tlist: array_type,
    npmod=np
) -> array_type:
    heff = make_heff(heff_coeffs, basis, num_qubits, npmod=npmod)
    return tlist[:, None, None] * heff[None, ...]


def make_ueff(
    heff_coeffs: array_type,
    basis: array_type,
    num_qubits: int,
    tlist: array_type,
    phase_factor: float = -1.,
    npmod=np
) -> array_type:
    heff_t = make_heff_t(heff_coeffs, basis, num_qubits, tlist, npmod=npmod)
    return matrix_ufunc(lambda v: npmod.exp(phase_factor * 1.j * v), heff_t, hermitian=True, npmod=npmod)


def heff_fidelity(
    time_evolution: array_type,
    heff_coeffs: array_type,
    basis: array_type,
    num_qubits: int,
    tlist: array_type,
    npmod=np
) -> array_type:
    heff_t = make_heff_t(heff_coeffs, basis, num_qubits, tlist, npmod=npmod)
    ueffdag_t = matrix_ufunc(lambda v: npmod.exp(1.j * v), heff_t, hermitian=True, npmod=npmod)

    tr_u_ueffdag = npmod.trace(npmod.matmul(time_evolution, ueffdag_t), axis1=1, axis2=2)
    fidelity = (npmod.square(tr_u_ueffdag.real) + npmod.square(tr_u_ueffdag.imag)) / (basis.shape[-1] ** 2)
    
    return fidelity
