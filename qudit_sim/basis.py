"""Functions to manipulate Pauli matrix bases."""

from collections import defaultdict
from typing import List, Union, Optional
import numpy as np

import rqutils.paulis as paulis


def change_basis(
    components: np.ndarray,
    to_basis: Union[str, np.ndarray],
    from_basis: str = "gell-mann",
    num_qudits: Optional[int] = None
) -> np.ndarray:
    r"""Return the Pauli components in the given basis.

    By default, the Pauli components are given in the Gell-Mann (orthogonal) basis. However,
    different generator decompositions are more convenient in certain cases. For example,
    when discussing qudit gates, the "qudit" basis with X, Y, Z defined for each level
    transition can make the connection between the generator and the pulse control clearer.
    This function converts the Gell-Mann basis components into components in other bases.

    For a matrix dimension :math:`n`, the :math:`i`-th matrix :math:`M_i` of each basis
    is defined as follows:

    :``qudit``:
        :math:`M_0 = I`

        :math:`M_{3j+\{1,2,3\}} = \{X, Y, Z\}_{j\,j+1}` for :math:`0 \leq j \leq n-2`

        :math:`(M_{N_d+2j+\{1,2\}})_{k,l} = \{1,-i\}\delta_{k,j}\delta_{l,j+d} + \{1,i\}\delta_{k,j+d}\delta_{l,j}`
        where :math:`2 \leq d \leq n-1`, :math:`N_d = 1 + 3(n-1) + 2\sum_{p=2}^{d-1}(n+1-p)`, and
        :math:`0 \leq j \leq n-d-1`.

    :``pauli<j>`` (``<j>``=0,..,n-1):
        Cyclically shifted Gell-Mann basis. :math:`M_0 = \sqrt{2/n}I` is the same for any
        value of ``j``, and :math:`M_{1,2,3}` corresponds to the Pauli X, Y, and Z matrices
        where the :math:`|j\rangle \leftrightarrow |j+1\rangle` transition is considered
        as the "qubit space". Higher-numbered matrices :math:`M_{\ge 4}` are defined by
        extending the matrix dimension from this qubit space cyclically. The last matrix
        is :math:`M_{n^2-1}_{k,l} = \sqrt{\frac{2}{n(n-1)}}\delta_{kl}(1 - n\delta_{k\,j-1})`.

    It is also possible to pass a custom transformation matrix as the ``to_basis`` argument.
    The convention is for the transformation matrix to convert the components column vector by left
    multiplication. In other words, the original basis matrices :math:`\lambda_{i}` and the
    new basis :math:`\lambda'_{j}` are related through the transformation matrix :math:`V_{ij}` by

    .. math::

        \lambda_{i} = \sum_{j} \lambda'_{j} V_{ji}.

    The original and new components :math:`\nu_{i}` and :math:`\nu'_{j}` satisfy
    :math:`\sum_{i} \lambda_{i} \nu_{i} = \sum_{j} \lambda'_{j} \nu'_{j}`, which implies that

    .. math::

        \nu'_{j} = \sum_{i} V_{ji} \nu_{i}.

    The components array can have extra dimensions (e.g. the time axis). In such cases, the
    last ``num_qudits`` dimensions are converted.

    Args:
        components: Components array given in the Pauli basis.
        to_basis: Name of the basis to convert to, or the transformation matrix.
        from_basis: Name of the basis to convert from. Ignored if ``to_basis`` is an array.
        num_qudits: If components has extra dimensions, specify the number of qudits.

    Returns:
        Basis-converted Pauli components.
    """
    if to_basis == from_basis:
        return components

    if isinstance(to_basis, (np.ndarray, list)):
        conversions = to_basis

    else:
        if to_basis != 'gell-mann' and from_basis != 'gell-mann':
            components = change_basis(components, 'gell-mann', from_basis, num_qudits)
            from_basis = 'gell-mann'

        if num_qudits is None:
            num_qudits = len(components.shape)

        # Input sanity check
        dim = np.around(np.sqrt(components.shape[-num_qudits:])).astype(int)
        if not np.allclose(np.square(dim), components.shape[-num_qudits:]):
            raise ValueError(f'Invalid shape of components array {components.shape}')

        if to_basis == 'gell-mann':
            conversion_id = f'from_{from_basis}'
            basis = from_basis
        else:
            conversion_id = f'to_{to_basis}'
            basis = to_basis

        conversions = list()
        for nlevels in dim:
            try:
                conversion = _conversion_matrices[conversion_id][nlevels]
            except KeyError:
                conversion = _make_conversion_matrix(basis, nlevels)

                _conversion_matrices[f'to_{basis}'][nlevels] = conversion
                _conversion_matrices[f'from_{basis}'][nlevels] = np.linalg.inv(conversion)

            conversions.append(conversion)

    converted = components
    for iq in range(-num_qudits, 0):
        converted = np.moveaxis(np.tensordot(conversions[iq], converted, (1, iq)), 0, iq)

    if components.dtype.kind == 'f':
        return converted.real

    return converted

_conversion_matrices = defaultdict(dict)


def _make_conversion_matrix(basis, nlevels):
    """Make the matrix that maps the given basis to gell-mann through right-multiplication."""

    if basis == 'qudit':
        # Matrix ordering and number of matrices in each sector
        # I (XYZ)_{01} (XYZ)_{12} ... (XY)_{02} (XY)_{13} ... (XY)_{03} (XY)_{14} ...
        # - ------------------------- ----------------------- -----------------------
        # 1                    3(n-1)                  2(n-2)                  2(n-3) ..

        new_basis = np.zeros((nlevels ** 2, nlevels, nlevels), dtype=complex)

        new_basis[0] = np.eye(nlevels)

        ib = 1

        for level in range(nlevels - 1):
            new_basis[ib, level, level + 1] = 1.
            new_basis[ib, level + 1, level] = 1.
            ib += 1
            new_basis[ib, level, level + 1] = -1.j
            new_basis[ib, level + 1, level] = 1.j
            ib += 1
            new_basis[ib, level, level] = 1.
            new_basis[ib, level + 1, level + 1] = -1.
            ib += 1

        for gap in range(2, nlevels):
            for level in range(nlevels - gap):
                new_basis[ib, level, level + gap] = 1.
                new_basis[ib, level + gap, level] = 1.
                ib += 1
                new_basis[ib, level, level + gap] = -1.j
                new_basis[ib, level + gap, level] = 1.j
                ib += 1

    elif basis.startswith('pauli'):
        shift = _get_pauli_shift(basis, nlevels)
        if shift == 0:
            return np.eye(nlevels ** 2)

        new_basis = np.roll(paulis.paulis(nlevels), shift, axis=(1, 2))

    conversion = np.linalg.inv(paulis.components(new_basis, dim=nlevels).T)

    return conversion


def diagonals(basis: Union[str, None, np.ndarray], dim: int) -> np.ndarray:
    if isinstance(basis, np.ndarray):
        offdiagonals = np.nonzero(paulis.symmetry(dim))
        indices = np.nonzero(np.all(basis.T[offdiagonals] == 0., axis=0))[0]

    elif basis is None or basis == 'gell-mann' or basis.startswith('pauli'):
        indices = np.nonzero(paulis.symmetry(dim) == 0)[0]

    elif basis == 'qudit':
        indices = np.arange(dim) * 3

    else:
        raise ValueError(f'Unknown basis name {basis}')

    return indices


def matrix_labels(basis: str, dim: int) -> List[str]:
    if basis == 'gell-mann':
        symbol = list(fr'\lambda_{{{i}}}' for i in range(dim ** 2))

    elif basis == 'qudit':
        symbol = ['I']
        for l in range(dim - 1):
            symbol += [f'{s}_{{{l}{l + 1}}}' for s in ['X', 'Y', 'Z']]
        for d in range(dim - 1, 1, -1):
            for b in range(d - 1):
                symbol += [f'{s}_{{{b}{b + d}}}' for s in ['X', 'Y']]

    elif basis.startswith('pauli'):
        shift = _get_pauli_shift(basis, dim)
        symbol = list(fr'\lambda^{{({shift})}}_{{{i}}}' for i in range(dim ** 2))

    else:
        raise ValueError(f'Unknown basis name {basis}')

    return symbol


def _get_pauli_shift(basis: str, dim: int) -> int:
    try:
        shift = int(basis[len('pauli'):])
        if shift < 0 or shift >= dim:
            raise ValueError()
    except:
        raise ValueError(f'Invalid basis name {basis}')

    return shift
