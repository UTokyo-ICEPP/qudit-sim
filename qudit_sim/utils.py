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
