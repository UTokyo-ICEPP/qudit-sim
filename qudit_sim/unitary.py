from typing import Tuple, Union
from types import ModuleType
import numpy as np

from rqutils import ArrayType

def truncate_matrix(
    matrix: np.ndarray,
    num_qudits: int,
    original_dim: int,
    reduced_dim: int,
) -> np.ndarray:
    """Truncate a multi-qudit operator matrix.

    The input matrix can have extra dimensions in front. The last two dimensions must have dimensions
    ``(original_dim ** num_qudits, original_dim ** num_qudits)``, and it will be truncated to
    ``(reduced_dim ** num_qudits, reduced_dim ** num_qudits)`` taking the first ``reduced_dim`` levels
    in each qudit space.

    Args:
        matrix: The input matrix or an array of matrices.
        original_dim: Original number of levels of each single-qudit space.
        reduced_dim: Truncated number of levels.

    Returns:
        A truncated ndarray.
    """
    if original_dim == reduced_dim:
        return matrix

    elif original_dim < reduced_dim:
        raise ValueError(f'Cannot expand matrix dimension from {original_dim} to {reduced_dim}')

    if matrix.shape[-2:] != (original_dim ** num_qudits,) * 2:
        raise ValueError('Matrix shape is inconsistent')

    extra_dims = matrix.shape[:-2]

    truncation = (slice(0, reduced_dim),) * (num_qudits * 2)

    matrix = matrix.reshape((-1,) + (original_dim,) * (num_qudits * 2))
    trunc_matrix = matrix[(slice(None),) + truncation]
    trunc_matrix = trunc_matrix.reshape(extra_dims + (reduced_dim ** num_qudits,) * 2)

    return trunc_matrix


def closest_unitary(
    matrix: np.ndarray,
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
    else:
        return unitary
