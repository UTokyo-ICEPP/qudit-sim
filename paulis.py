from typing import Optional
import string
import numpy as np

def make_generalized_paulis(
    comp_dim: int = 2,
    matrix_dim: Optional[int] = None
) -> np.ndarray:
    """Return a list of normalized generalized Pauli matrices of given dimension as a numpy array.
    
    Args:
        comp_dim: Dimension of the Pauli matrices.
        matrix_dim: Dimension of the containing matrix, which may be greater than `comp_dim`.
        
    Returns:
        The full list of Pauli matrices as an array of dtype `complex128` and shape
            `(comp_dim ** 2, matrix_dim, matrix_dim)`.
    """
    
    if matrix_dim is None:
        matrix_dim = comp_dim
        
    assert matrix_dim >= comp_dim, 'Matrix dimension cannot be smaller than Pauli dimension'
    
    num_paulis = comp_dim ** 2
    
    paulis = np.zeros((num_paulis, matrix_dim, matrix_dim), dtype=np.complex128)
    paulis[0, :comp_dim, :comp_dim] = np.diagflat(np.ones(comp_dim)) / np.sqrt(comp_dim)
    ip = 1
    for idim in range(1, comp_dim):
        for irow in range(idim):
            paulis[ip, irow, idim] = 1. / np.sqrt(2.)
            paulis[ip, idim, irow] = 1. / np.sqrt(2.)
            ip += 1
            paulis[ip, irow, idim] = -1.j / np.sqrt(2.)
            paulis[ip, idim, irow] = 1.j / np.sqrt(2.)
            ip += 1

        paulis[ip, :idim + 1, :idim + 1] = np.diagflat(np.array([1.] * idim + [-idim]))
        paulis[ip] /= np.sqrt(np.trace(paulis[ip] * paulis[ip]))
        ip += 1
        
    return paulis


def make_prod_basis(
    basis: np.ndarray,
    num_qubits: int
) -> np.ndarray:
    """Return a list of basis matrices of multi-qubit operators.
    
    Args:
        basis: Basis of single-qubit operators. Array with shape
            `(num_basis, matrix_dim, matrix_dim)`.
        num_qubits: Number of qubits.
        
    Returns:
        The full list of basis matrices as a single `num_qubits + 2`-dimensional array.
            The first `num_qubits` dimensions are size `num_basis`, and the last two are
            size `matrix_dim ** num_qubits`.
    """
    # Use of einsum implies that we can deal with at most 52 // 3 = 17 qubits
    al = string.ascii_letters
    indices_in = []
    indices_out = [''] * 3
    for il in range(0, num_qubits * 3, 3):
        indices_in.append(al[il:il + 3])
        indices_out[0] += al[il]
        indices_out[1] += al[il + 1]
        indices_out[2] += al[il + 2]

    indices = f'{",".join(indices_in)}->{"".join(indices_out)}'
    
    shape = [basis.shape[0]] * num_qubits + [basis.shape[1] ** num_qubits] * 2
    
    return np.einsum(indices, *([basis] * num_qubits)).reshape(*shape)
