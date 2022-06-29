"""Hamiltonian visualization routines."""

from typing import Union, Tuple, List
import numpy as np
import qutip as qtp

try:
    get_ipython()
except NameError:
    has_ipython = False
    PrintReturnType = str
else:
    has_ipython = True
    from IPython.display import Latex
    PrintReturnType = Latex

from rqutils.qprint import QPrintBraKet

from ..drive import HamiltonianCoefficient

def print_hamiltonian(
    hamiltonian: List[Union[qtp.Qobj, Tuple[qtp.Qobj, HamiltonianCoefficient]]],
    phase_norm: Tuple[float, str]=(np.pi, 'π')
) -> PrintReturnType:
    """Print the Hamiltonian list built by HamiltonianBuilder.

    Args:
        hamiltonian: Input list to ``qutip.sesolve``.
        phase_norm: Normalization for displayed phases.

    Returns:
        A representation object for a LaTeX ``align`` expression or an expression string, with one Hamiltonian term per line.
    """
    exprs = []
    start = 0
    if isinstance(hamiltonian[0], qtp.Qobj):
        exprs.append(QPrintBraKet(hamiltonian[0].full(), dim=hamiltonian[0].dims[0], lhs_label=r'H_{\mathrm{static}} &'))
        start += 1

    for iterm, term in enumerate(hamiltonian[start:]):
        exprs.append(QPrintBraKet(term[0], dim=term[0].dims[0], lhs_label=f'H_{{{iterm}}} &', amp_norm=(1., fr'[\text{{{term[1]}}}]*'), phase_norm=phase_norm))

    if has_ipython:
        return Latex(r'\begin{align}' + r' \\ '.join(expr.latex(env=None) for expr in exprs) + r'\end{align}')
    else:
        return '\n'.join(str(expr) for expr in exprs)
