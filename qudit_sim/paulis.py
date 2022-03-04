from typing import Optional
import string
import numpy as np

def make_generalized_paulis(
    comp_dim: int = 2,
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
    paulis[0, :comp_dim, :comp_dim] = np.diagflat(np.ones(comp_dim))
    ip = 1
    for idim in range(1, comp_dim):
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


def heff_expr(
    coefficients: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = None
) -> str:
    """Generate a LaTeX expression of the effective Hamiltonian from the Pauli coefficients.
    
    The dymamic range of the numerical values of the coefficients is set by the maximum absolute
    value. For example, if the maximum absolute value is between 1.e+6 and 1.e+9, the coefficients
    are expressed in MHz, with the minimum of 0.001 MHz. Pauli terms whose coefficients have
    absolute values below the threshold are ommitted.
    
    Args:
        heff: Array of Pauli coefficients returned by find_heff
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        threshold: Ignore terms with absolute coefficients below this value.
        
    Returns:
        A LaTeX expression string for the effective Hamiltonian.
    """
    labels = pauli_labels(coefficients.shape[0], symbol)
        
    maxval = np.amax(np.abs(coefficients))
    for base, unit in [(1.e+9, 'GHz'), (1.e+6, 'MHz'), (1.e+3, 'kHz'), (1., 'Hz')]:
        norm = 2. * np.pi * base
        if maxval > norm:
            if threshold is None:
                threshold = norm * 1.e-3
            break
            
    if threshold is None:
        raise RuntimeError(f'Passed coefficients with maxabs = {maxval}')
            
    expr = ''
    
    for index in np.ndindex(coefficients.shape):
        coeff = coefficients[index]
        if abs(coeff) < threshold:
            continue
            
        if coeff < 0.:
            expr += ' - '
        elif expr:
            expr += ' + '
            
        if len(coefficients.shape) == 1:
            expr += f'{abs(coeff) / norm:.3f}{labels[index[0]]}'
        else:
            oper = ''.join(labels[i] for i in index)
            if len(coefficients.shape) == 2:
                denom = '2'
            else:
                denom = '2^{%d}' % (len(coefficients.shape) - 1)
                
            expr += f'{abs(coeff) / norm:.3f}' + (r'\frac{%s}{%s}' % (oper, denom))
        
    return (r'\frac{H_{\mathrm{eff}}}{2 \pi \mathrm{%s}} = ' % unit) + expr


def pauli_labels(num_paulis: int, symbol: Optional[str] = None):
    if symbol is None:
        if num_paulis == 4:
            labels = ['I', 'X', 'Y', 'Z']
        else:
            labels = list((r'\lambda_{%d}' % i) for i in range(num_paulis))
    else:
        labels = list((r'%s_{%d}' % (symbol, i)) for i in range(num_paulis))
        
    return labels