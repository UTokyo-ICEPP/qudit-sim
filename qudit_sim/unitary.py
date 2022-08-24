from typing import Tuple, Union
import numpy as np

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

#     flattened_grid_indices = np.indices(reduced_dim_tuple).reshape(num_qudits, -1)
#     trunc_indices_single = np.ravel_multi_index(flattened_grid_indices, original_dim_tuple)
#     # example: (4, 4) -> (3, 3): [0, 1, 2, 4, 5, 6, 8, 9, 10]

#     trunc_indices = np.ix_(trunc_indices_single, trunc_indices_single)

#     # matrix may have extra dimensions in front
#     matrix = np.moveaxis(matrix, (-2, -1), (0, 1))
#     trunc_matrix = matrix[trunc_indices]
#     return np.moveaxis(matrix, (0, 1), (-2, -1))


def closest_unitary(
    matrix: np.ndarray,
    with_fidelity: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
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
    v, _, wdag = np.linalg.svd(matrix)

    unitary = v @ wdag

    if with_fidelity:
        fidelity = np.square(np.abs(np.trace(matrix @ unitary.conjugate().T))
                         / unitary.shape[-1])

        return unitary, fidelity
    else:
        return unitary
