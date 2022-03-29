from typing import Optional, Any
import string
import numpy as np

from .utils import matrix_ufunc

def get_num_paulis(pauli_dim: int) -> int:
    return pauli_dim ** 2


def get_pauli_dim(num_paulis: int) -> int:
    pauli_dim = np.sqrt(num_paulis)
    assert abs(np.around(pauli_dim) - pauli_dim) < 1.e-9
    return int(pauli_dim)

    
def make_generalized_paulis(
    pauli_dim: int = 2,
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
        pauli_dim: Dimension of the Pauli matrices.
        matrix_dim: Dimension of the containing matrix, which may be greater than `pauli_dim`.
        
    Returns:
        The full list of Pauli matrices as an array of dtype `complex128` and shape
            `(pauli_dim ** 2, matrix_dim, matrix_dim)`.
    """
    
    if matrix_dim is None:
        matrix_dim = pauli_dim
        
    assert matrix_dim >= pauli_dim, 'Matrix dimension cannot be smaller than Pauli dimension'
    
    num_paulis = get_num_paulis(pauli_dim)
    
    paulis = np.zeros((num_paulis, matrix_dim, matrix_dim), dtype=np.complex128)
    paulis[0, :pauli_dim, :pauli_dim] = np.diagflat(np.ones(pauli_dim))
    ip = 1
    for idim in range(1, pauli_dim):
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


def extract_coefficients(
    hermitian: np.ndarray,
    num_qubits: int = 1
) -> np.ndarray:
    """Extract the Pauli coefficients"""
    pauli_dim = np.around(np.power(hermitian.shape[-1], 1. / num_qubits)).astype(int)
    basis = make_prod_basis(make_generalized_paulis(pauli_dim), num_qubits)
    return np.tensordot(hermitian, basis, ((-2, -1), (-1, -2))).real / 2.
    
    
def get_l0_projection(reduced_dim: int, original_dim: int) -> np.ndarray:
    """Return the vector corresponding to lambda_0 in reduced_dim in the original_dim space.
    
    Args:
        reduced_dim: Matrix dimension of the target subspace.
        original_dim: Matrix dimension of the full space.
        
    Returns:
        Coefficient vector v(m, n) such that v(m, n)[:] . paulis(n)[:] = lambda_0(m).
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


def truncate_coefficients(
    coeffs: np.ndarray,
    original_dim: int,
    reduced_dim: int,
    num_qubits: Optional[int] = None
) -> np.ndarray:
    """Truncate a Pauli coefficient array into a reduced set of Pauli operators.
    
    When the matrix dimensions are truncated, the diagonal matrices in the higher dimension are projected
    onto the identity in the lower dimension.
    """
    assert reduced_dim <= original_dim

    coeffs = coeffs.copy()
    
    if reduced_dim == original_dim:
        return coeffs

    # Coefficient for lambda_0 projection
    l0_projection = get_l0_projection(reduced_dim, original_dim)
    
    if num_qubits is None:
        num_qubits = len(coeffs.shape)

    ## Allow additional dimensions of coeffs
    num_pre_dims = len(coeffs.shape) - num_qubits
    pre_slices = (slice(None),) * num_pre_dims

    for iq in range(num_qubits):
        # For each qubit, set the coefficient at index 0 to projection . coeffs
        indices = tuple(0 if i == iq else slice(None) for i in range(num_qubits))
        coeffs[pre_slices + indices] = np.tensordot(coeffs, l0_projection, (num_pre_dims + iq, 0))

    num_reduced_paulis = get_num_paulis(reduced_dim)
    slices = (slice(num_reduced_paulis),) * num_qubits
    return coeffs[pre_slices + slices]
    

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
    pauli_dim: int,
    num_qubits: int
) -> np.ndarray:
    """Compute the index into prod_basis from a flat list index.
    """
    num_paulis = get_num_paulis(pauli_dim)
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


def matrix_symmetry(pauli_dim: int):
    num_paulis = get_num_paulis(pauli_dim)

    symmetry = np.zeros(num_paulis, dtype=int)

    ip = 1
    for idim in range(1, pauli_dim):
        for irow in range(idim):
            symmetry[ip] = 1
            ip += 1
            symmetry[ip] = -1
            ip += 1

        ip += 1
        
    return symmetry


def shift_phase(coeffs: np.ndarray, phase: float, dim: int = 0):
    pauli_dim = get_pauli_dim(coeffs.shape[dim])
    num_paulis = get_num_paulis(pauli_dim)

    perm = list(range(len(coeffs.shape)))
    perm.remove(dim)
    perm.insert(0, dim)
    
    transposed = coeffs.transpose(*perm)
    shifted = coeffs.copy().transpose(*perm)

    symmetry = matrix_symmetry(pauli_dim)
    sym_idx = np.asarray(symmetry == 1).nonzero()[0]
    asym_idx = np.asarray(symmetry == -1).nonzero()[0]
    
    shifted[sym_idx] = np.cos(phase) * transposed[sym_idx] + np.sin(phase) * transposed[asym_idx]
    shifted[asym_idx] = -np.sin(phase) * transposed[sym_idx] + np.cos(phase) * transposed[asym_idx]
    
    rperm = list(range(len(coeffs.shape)))
    rperm.remove(0)
    rperm.insert(dim, 0)

    return shifted.transpose(*rperm)
