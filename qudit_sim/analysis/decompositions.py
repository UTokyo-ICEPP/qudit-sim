"""Analysis of Pauli decompositions of Hamiltonians and gates."""

from typing import Union, List, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    get_ipython()
except NameError:
    has_ipython = False
else:
    has_ipython = True
    from IPython.display import Latex

from rqutils.math import matrix_angle
from rqutils.qprint import QPrintPauli
import rqutils.paulis as paulis

from ..util import FrequencyScale, PulseSimResult

def print_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    symbol: Optional[str] = None,
    precision: int = 3,
    threshold: float = 1.e-3,
    lhs_label: Optional[str] = None,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto
) -> Union[Latex, str]:
    r"""Compose a LaTeX expression of the effective Hamiltonian from the Pauli components.

    Args:
        components: Array of Pauli components returned by find_heff.
        uncertainties: Array of component uncertainties.
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        precision: Number of digits below the decimal point to show.
        threshold: Ignore terms with absolute components below this value relative to the given scale
            (if >0) or to the maximum absolute component (if <0).
        scale: Normalize the components with the frequency scale. If None, components are taken
            to be dimensionless. If `FrequencyScale.auto`, scale is found from the maximum absolute
            value of the components. String `'pi'` is also allowed, in which case the components are
            normalized by :math:`\pi`.

    Returns:
        A representation object for a LaTeX expression or an expression string for the effective Hamiltonian.
    """
    max_abs = np.amax(np.abs(components))

    if scale is FrequencyScale.auto:
        scale = FrequencyScale.find_energy_scale(max_abs)

    if scale is None:
        scale_omega = 1.
        if lhs_label is None:
            lhs_label = r'i \mathrm{log} U'
    elif scale == 'pi':
        scale_omega = np.pi
        if lhs_label is None:
            lhs_label = r'\frac{i \mathrm{log} U}{\pi}'
    else:
        scale_omega = scale.pulsatance_value
        if lhs_label is None:
            lhs_label = r'\frac{H}{2\pi\,\mathrm{%s}}' % scale.frequency_unit

    components = components / scale_omega
    max_abs /= scale_omega

    if threshold > 0.:
        amp_cutoff = threshold / max_abs
    else:
        amp_cutoff = -threshold / scale_omega

    if uncertainties is not None:
        selected = np.nonzero(np.abs(components) > amp_cutoff * max_abs)
        unc = np.zeros_like(uncertainties)
        unc[selected] = uncertainties[selected] / scale_omega

        central = QPrintPauli(components, amp_format=f'.{precision}f',
                              amp_cutoff=amp_cutoff, symbol=symbol)

        uncert = QPrintPauli(unc, amp_format=f'.{precision}f',
                             amp_cutoff=0., symbol=symbol)

        if has_ipython:
            return Latex(fr'\begin{{split}} {lhs_label} & = {central.latex(env=None)} \\'
                         + fr' & \pm {uncert.latex(env=None)} \end{{split}}')
        else:
            return f'{lhs_label}  = {central}\n{" " * len(lhs_label)} +- {uncert}'

    else:
        pobj = QPrintPauli(components, amp_format=f'.{precision}f', amp_cutoff=amp_cutoff,
                           lhs_label=lhs_label, symbol=symbol)

        if has_ipython:
            return Latex(pobj.latex())
        else:
            return str(pobj)


def plot_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    threshold: float = 1.e-2,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto,
    ignore_identity: bool = True
) -> mpl.figure.Figure:
    """Plot the Hamiltonian components as a bar graph in the decreasing order in the absolute value.

    Args:
        components: Array of Pauli components returned by find_heff.
        uncertainties: Array of component uncertainties.
        threshold: Ignore terms with absolute components below this value relative to the given scale
            (if >0) or to the maximum absolute component (if <0).
        scale: Normalize the components with the frequency scale. If None, components are taken
            to be dimensionless. If `FrequencyScale.auto`, scale is found from the maximum absolute
            value of the components. String `'pi'` is also allowed, in which case the components are
            normalized by :math:`\pi`.
        ignore_identity: Ignore the identity term.

    Returns:
        A Figure object containing the bar graph.
    """
    max_abs = np.amax(np.abs(components))

    if scale is FrequencyScale.auto:
        scale = FrequencyScale.find_energy_scale(max_abs)

    if scale is None:
        scale_omega = 1.
        ylabel = r'$\nu$'
    elif scale == 'pi':
        scale_omega = np.pi
        ylabel = r'$\nu/\pi$'
    else:
        scale_omega = scale.pulsatance_value
        # If we normalize by 2*pi*frequency, the displayed values are in frequency
        ylabel = r'$\nu\,(2\pi\,\mathrm{' + scale.frequency_unit + '})$'

    # Dividing by omega -> now everything is in terms of frequency (not angular)
    # Note: Don't use '/='!
    components = components / scale_omega

    if ignore_identity:
        identity_index = (0,) * len(components.shape)
        components[identity_index] = 0.

    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold *= -max_abs / scale_omega

    flat_indices = np.argsort(-np.abs(components.reshape(-1)))
    nterms = np.count_nonzero(np.abs(components) > threshold)
    indices = np.unravel_index(flat_indices[:nterms], components.shape)

    if uncertainties is None:
        yerr = None
    else:
        uncertainties = uncertainties / scale_omega
        if ignore_identity:
            uncertainties[identity_index] = 0.

        yerr = uncertainties[indices]

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), components[indices], yerr=yerr)

    ax.axhline(0., color='black', linewidth=0.5)

    pauli_dim = np.around(np.sqrt(components.shape)).astype(int)
    labels = paulis.labels(pauli_dim, symbol='',
                           delimiter=('' if pauli_dim[0] == 2 else ','))

    xticks = np.char.add(np.char.add('$', labels), '$')

    ax.set_xticks(np.arange(nterms), labels=xticks[indices])
    ax.set_ylabel(ylabel)

    return fig


def gate_components(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: Optional[int] = None
) -> np.ndarray:
    r"""Compute the Pauli components of the generator of the unitary obtained from the simulation.

    Args:
        sim_result: Pulse simulation result or a list thereof.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`), or a
        list of such arrays if `sim_result` is an array.
    """
    if isinstance(sim_result, list):
        return list(_single_gate_components(res, comp_dim) for res in sim_result)
    else:
        return _single_gate_components(sim_result, comp_dim)


def _single_gate_components(sim_result, comp_dim):
    gate = sim_result.states[-1]
    components = paulis.components(-matrix_angle(gate), sim_result.dim).real

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        components = paulis.truncate(components, (comp_dim,) * len(sim_result.dim))

    return components
