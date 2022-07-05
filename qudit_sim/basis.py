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
    when discussing qudit gates, often the more convenient basis is the "qudit" basis with
    X, Y, Z defined for each level transition. This function converts the Pauli-basis
    components into components in other bases.

    For a Pauli matrix dimension :math:`n`, the :math:`i`-th matrix :math:`M_i` of each basis
    is defined as follows:

    :``qudit``:
        :math:`M_0 = I`

        :math:`M_{3j+\{1,2,3\}} = \{X, Y, Z\}_{j\,j+1}` for :math:`0 \leq j \leq n-2`

        :math:`(M_{N_d+2j+\{1,2\}})_{k,l} = \{1,-i\}\delta_{k,j}\delta_{l,j+d} + \{1,i\}\delta_{k,j+d}\delta_{l,j}`
        where :math:`2 \leq d \leq n-1`, :math:`N_d = 1 + 3(n-1) + 2\sum_{p=2}^{d-1}(n+1-p)`, and
        :math:`0 \leq j \leq n-d-1`.

    There is only one basis implemented currently.

    The components array can have extra dimensions (e.g. the time axis). In such cases, the last ``num_qudits``
    dimensions are converted.

    Args:
        components: Components array given in the Pauli basis.
        to_basis: Name of the basis to convert to, or the change-of-basis matrix.
        from_basis: Name of the basis to convert from. Ignored if ``to_basis`` is an array.
        num_qudits: If components has extra dimensions, specify the number of qudits.

    Returns:
        Basis-converted Pauli components.
    """
    if isinstance(to_basis, np.ndarray):
        conversion = to_basis

    else:
        if to_basis != 'gell-mann' and from_basis != 'gell-mann':
            components = change_basis(components, 'gell-mann', from_basis, num_qudits)
            from_basis = 'gell-mann'

        if num_qudits is None:
            num_qudits = len(components.shape)

        # Input sanity check
        dim = np.around(np.sqrt(components.shape[-num_qudits:])).astype(int)
        nlevels = dim[0]
        if np.any(dim != nlevels) or not np.allclose(np.square(dim), components.shape[-num_qudits:]):
            raise ValueError(f'Invalid shape of components array {components.shape}')

        if to_basis == 'gell-mann':
            conversion_id = f'from_{from_basis}'
            basis = from_basis
        else:
            conversion_id = f'to_{to_basis}'
            basis = to_basis

        try:
            repository = _conversion_matrices[conversion_id]
        except KeyError:
            raise ValueError(f'Unknown basis name {basis}')

        try:
            conversion = repository[nlevels]
        except KeyError:
            conversion = _make_conversion_matrix(basis, nlevels)

            _conversion_matrices[f'to_{basis}'][nlevels] = conversion
            _conversion_matrices[f'from_{basis}'][nlevels] = np.linalg.inv(conversion)

            conversion = repository[nlevels]

    converted = components
    for iq in range(-num_qudits, 0):
        converted = np.moveaxis(np.tensordot(converted, conversion, (iq, 0)), -1, iq)

    return converted


_conversion_matrices = {'from_qudit': dict(), 'to_qudit': dict()}


def _make_conversion_matrix(basis, nlevels):
    """Make the constant matrix that maps the given basis to gell-mann through left-multiplication.

    We compute the change-of-basis matrix C that relates basis b and Gell-Mann basis λ:
      X = α . λ = β . b
      λ = C . b
    Then
      β = α . C
    i.e. the converted components are obtained through right-multiplication by C.
    """

    conversion = np.zeros((nlevels ** 2, nlevels ** 2))

    if basis == 'qudit':
        # Matrix ordering and number of matrices in each sector
        # I (XYZ)_{01} (XYZ)_{12} ... (XY)_{02} (XY)_{13} ... (XY)_{03} (XY)_{14} ...
        # - ------------------------- ----------------------- -----------------------
        # 1                    3(n-1)                  2(n-1)                  2(n-2) ..

        # Identity
        conversion[0, 0] = np.sqrt(2. / nlevels)

        # XY
        for level in range(nlevels - 1):
            iq = 1 + 3 * level
            ig = (level + 2) ** 2 - 3
            for xy in range(2):
                conversion[ig + xy, iq + xy] = 1.

        # Other off-diagonals
        iq = 1 + 3 * (nlevels - 1)
        for gap in range(2, nlevels):
            for level in range(nlevels - gap):
                ig = (gap + level) ** 2 + 2 * level
                for xy in range(2):
                    conversion[ig + xy, iq + xy] = 1.

                iq += 2

        # Diagonals
        for level in range(nlevels - 1):
            ig = (level + 2) ** 2 - 1
            for iq in range(3, 3 * (level + 2), 3):
                conversion[ig, iq] = iq // 3

            conversion[ig] *= np.sqrt(2. / (level + 1) / (level + 2))

    return conversion


def diagonals(basis: Union[str, None, np.ndarray], dim: int) -> np.ndarray:
    if isinstance(basis, np.ndarray):
        offdiagonals = np.nonzero(paulis.symmetry(dim))
        indices = np.nonzero(np.all(basis.T[offdiagonals] == 0., axis=0))[0]

    elif basis is None or basis == 'gell-mann':
        indices = np.nonzero(paulis.symmetry(dim) == 0)[0]

    elif basis == 'qudit':
        indices = np.arange(dim) * 3

    else:
        raise ValueError(f'Unknown basis name {basis}')

    return indices


def matrix_labels(basis: str, dim: int) -> List[str]:
    if basis == 'qudit':
        symbol = ['I']
        for l in range(dim - 1):
            symbol += [f'{s}_{{{l}{l + 1}}}' for s in ['X', 'Y', 'Z']]
        for d in range(dim - 1, 1, -1):
            for b in range(d - 1):
                symbol += [f'{s}_{{{b}{b + d}}}' for s in ['X', 'Y']]

    else:
        raise ValueError(f'Unknown basis name {basis}')

    return symbol
