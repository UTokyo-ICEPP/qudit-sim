from typing import Optional, Any
import string
import numpy as np

def get_num_paulis(dim: int) -> int:
    return dim ** 2


def make_generalized_paulis(
    dim: int = 2,
    matrix_dim: Optional[int] = None
) -> np.ndarray:
    """Return a list of generalized Pauli matrices of given dimension as a numpy array.

    **Note on normalization**
    
    Matrices are normalized so that :math:`\mathrm{Tr}(\lambda_i \lambda_j) = 2\delta_{ij}`.
    Therefore, to extract the coefficient :math:`\nu_{i_1 \dots i_n}` from a linear combination
    :math:`H = \sum_{\{j_1 \dots j_n\}} \nu_{j_1 \dots j_n} (\lambda_{i_1} \dots \lambda_{i_n})/2^{n-1}`,
    one needs to compute
    
    .. math::
    
    \nu_{i_1 \dots i_n} = \mathrm{Tr}\left( H \frac{\lambda_{i_1} \dots \lambda_{i_n}}{2} \right).
    
    Also, this normalization implies that for n-dimensions
    
    .. math::
    \lambda^{(n)}_0 = \sqrt{\frac{2}{n}} I_n.
    
    Args:
        dim: Dimension of the Pauli matrices.
        matrix_dim: Dimension of the containing matrix, which may be greater than `dim`.
        
    Returns:
        The full list of Pauli matrices as an array of dtype `complex128` and shape
            `(dim ** 2, matrix_dim, matrix_dim)`.
    """
    
    if matrix_dim is None:
        matrix_dim = dim
        
    assert matrix_dim >= dim, 'Matrix dimension cannot be smaller than Pauli dimension'
    
    num_paulis = get_num_paulis(dim)
    
    paulis = np.zeros((num_paulis, matrix_dim, matrix_dim), dtype=np.complex128)
    paulis[0, :dim, :dim] = np.diagflat(np.ones(dim))
    ip = 1
    for idim in range(1, dim):
        for irow in range(idim):
            paulis[ip, irow, idim] = 1.
            paulis[ip, idim, irow] = 1.
            ip += 1
            paulis[ip, irow, idim] = -1.j
            paulis[ip, idim, irow] = 1.j
            ip += 1

        paulis[ip, :idim + 1, :idim + 1] = np.diagflat(np.array([1.] * idim + [-idim]))
        ip += 1

    # Normalization
    norm = np.trace(paulis @ paulis, axis1=1, axis2=2)
    paulis *= np.sqrt(2. / norm)[:, np.newaxis, np.newaxis]
        
    return paulis


def get_l0_projection(reduced_dim: int, original_dim: int) -> np.ndarray:
    """Return the vector corresponding to lambda_0 in reduced_dim in the original_dim space.
    
    Args:
        reduced_dim: Matrix dimension of the target subspace.
        original_dim: Matrix dimension of the full space.
        
    Returns:
        Coefficient vector v(m, n) such that v(m, n) . paulis(n) = lambda_0(m).
    """
    assert reduced_dim <= original_dim
    
    coeffs_l0 = np.zeros(original_dim ** 2)
    coeffs_l0[0] = 1.
    
    ## Making use of the recursion relation 
    #   lambda_0^{d-n-1;d} = 1/sqrt(d-n) (sqrt(d-n-1)*lambda_0^{d-n;d} + lambda_{(d-n)^2-1})
    # where lambda_0^{k;d} is the k-dimensional lambda_0 expanded (zero-padded) to d dimensions
    
    for dim in range(original_dim, reduced_dim, -1):
        coeffs_l0 *= np.sqrt(dim - 1)
        coeffs_l0[dim ** 2 - 1] = 1.
        coeffs_l0 /= np.sqrt(dim)
        
    return coeffs_l0
    

def make_prod_basis(
    paulis: np.ndarray,
    num_qubits: int
) -> np.ndarray:
    """Return a list of basis matrices of multi-qubit operators.
    
    Args:
        paulis: Basis of single-qubit operators. Array with shape
            `(num_paulis, matrix_dim, matrix_dim)`.
        num_qubits: Number of qubits.
        
    Returns:
        The full list of basis matrices as a single `num_qubits + 2`-dimensional array.
            The first `num_qubits` dimensions are size `num_paulis`, and the last two are
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
    
    shape = [paulis.shape[0]] * num_qubits + [paulis.shape[1] ** num_qubits] * 2
    
    return np.einsum(indices, *([paulis] * num_qubits)).reshape(*shape)


def unravel_basis_index(
    indices: Any,
    dim: int,
    num_qubits: int
) -> np.ndarray:
    """Compute the index into prod_basis from a flat list index.
    """
    num_paulis = get_num_paulis(dim)
    shape = [num_paulis] * num_qubits
    return np.unravel_index(indices, shape)


def pauli_labels(num_paulis: int, symbol: Optional[str] = None):
    if symbol is None:
        if num_paulis == 4:
            labels = ['I', 'X', 'Y', 'Z']
        else:
            labels = list((r'\lambda_{%d}' % i) for i in range(num_paulis))
    else:
        labels = list((r'%s_{%d}' % (symbol, i)) for i in range(num_paulis))
        
    return np.array(labels)


def prod_basis_labels(num_paulis: int, num_qubits: int, symbol: Optional[str] = None):
    labels = pauli_labels(num_paulis, symbol=symbol)
    out = labels
    for _ in range(1, num_qubits):
        out = np.char.add(np.repeat(out[..., None], num_paulis, axis=-1), labels)
        
    return out
