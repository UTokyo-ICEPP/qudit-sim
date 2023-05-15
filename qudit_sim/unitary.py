from types import ModuleType
from typing import Optional, Tuple, Union
import numpy as np

from rqutils import ArrayType
from rqutils.math import matrix_angle, matrix_exp

def truncate_matrix(
    matrix: ArrayType,
    original_dim: Union[int, Tuple[int, ...]],
    reduced_dim: Union[int, Tuple[int, ...]]
) -> ArrayType:
    """Truncate a multi-qudit operator matrix.

    The input matrix can have extra dimensions in front. The last two dimensions must have
    dimensions ``(prod(original_dim), prod(original_dim))``, and it will be truncated to
    ``(prod(reduced_dim), prod(reduced_dim))`` taking the first ``reduced_dim`` levels in each qudit
    space.

    Args:
        matrix: The input matrix or an array of matrices.
        original_dim: Original numbers of levels of each single-qudit space.
        reduced_dim: Truncated numbers of levels.

    Returns:
        A truncated ndarray.
    """
    if original_dim == reduced_dim:
        return matrix

    if isinstance(original_dim, int):
        original_dim = (original_dim,)
    if isinstance(reduced_dim, int):
        reduced_dim = (reduced_dim,)

    if any(orig < reduc for orig, reduc in zip(original_dim, reduced_dim)):
        raise ValueError(f'Cannot expand matrix dimension from {original_dim} to {reduced_dim}')
    if matrix.shape[-2:] != (np.prod(original_dim),) * 2:
        raise ValueError('Matrix shape is inconsistent')

    extra_dims = matrix.shape[:-2]

    matrix = matrix.reshape((-1,) + original_dim + original_dim)

    truncation = tuple(slice(0, dim) for dim in reduced_dim) * 2
    trunc_matrix = matrix[(slice(None),) + truncation]
    trunc_matrix_shape = (np.prod(reduced_dim),) * 2
    trunc_matrix = trunc_matrix.reshape(extra_dims + trunc_matrix_shape)

    return trunc_matrix


def closest_unitary(
    matrix: ArrayType,
    with_fidelity: bool = False,
    npmod: ModuleType = np
) -> Union[ArrayType, Tuple[ArrayType, float]]:
    r"""Find the closest unitary to a matrix.

    In general, the closest unitary :math:`U`, i.e., one with the smallest 2-norm :math:`\lVert U-A \rVert`,
    to an operator :math:`A` can be calculated via a singular value decomposition:

    .. math::

        A & = V \Sigma W^{\dagger}, \\
        U & = V W^{\dagger}.

    The input matrix can have extra dimensions in front.

    Args:
        matrix: The input matrix or an array of matrices.
        with_fidelity: Return the fidelity of the computed unitary with respect to the original matrix.

    Returns:
        A matrix or an array of matrices corresponding to the closest unitary to the input. If with_fidelity
        is True, the fidelity of the computed unitary with respect to the input matrix is appended.
    """
    v, _, wdag = npmod.linalg.svd(matrix)

    unitary = v @ wdag

    if with_fidelity:
        conjugate_unitary = npmod.moveaxis(unitary.conjugate(), -1, -2)
        norm_tr = npmod.trace(matrix @ conjugate_unitary, axis1=-2, axis2=-1) / unitary.shape[-1]
        fidelity = npmod.square(norm_tr.real) + npmod.square(norm_tr.imag)
        return unitary, fidelity

    return unitary


def remove_global_phase(
    matrix: np.ndarray,
    block: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Remove the global phase from unitaries or blocks within.

    The first argument can be a unitary matrix or an array thereof. If the block argument is
    present, phase factor is applied to the entire matrix in order to eliminate the local phase of
    the specific block. The function takes the log of the unitaries, subtract the trace component,
    and re-exponentiates the result.

    Args:
        matrix: A unitary or an array of unitaries.
        block: The start and end (inclusive) rows of the subunitary.

    Returns:
        Exponential(s) of traceless Hermitian matrix(ces).
    """
    hermitian = matrix_angle(matrix)
    if block is None:
        norm_trace = np.trace(hermitian, axis1=-2, axis2=-1) / hermitian.shape[-1]
    else:
        subunitary = closest_unitary(matrix[..., block[0]:block[1] + 1, block[0]:block[1] + 1])
        norm_trace = np.trace(matrix_angle(subunitary), axis1=-2, axis2=-1) / subunitary.shape[-1]

    hermitian -= norm_trace[..., None, None] * np.eye(hermitian.shape[-1], dtype=complex)
    return matrix_exp(1.j * hermitian, hermitian=-1)
