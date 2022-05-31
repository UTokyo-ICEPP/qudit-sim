"""Hamiltonian and gate analysis routines."""

from typing import Union, Tuple, List, Optional, Dict, Hashable
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as qtp

try:
    from IPython.display import Latex
except ImportError:
    has_ipython = False
else:
    has_ipython = True

from rqutils.math import matrix_angle
from rqutils.qprint import QPrintBraKet, QPrintPauli
import rqutils.paulis as paulis

from .util import HamiltonianCoefficient, FrequencyScale, PulseSimResult
from .hamiltonian import HamiltonianBuilder
from .pulse_sim import pulse_sim
from .find_heff import find_heff

def print_hamiltonian(
    hamiltonian: List[Union[qtp.Qobj, Tuple[qtp.Qobj, HamiltonianCoefficient]]],
    phase_norm: Tuple[float, str]=(np.pi, 'π')
) -> Union[Latex, str]:
    """IPython printer of the Hamiltonian list built by HamiltonianBuilder.

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


def print_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    symbol: Optional[str] = None,
    precision: int = 3,
    threshold: float = 1.e-3,
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
        lhs_label = r'i \mathrm{log} U'
    elif scale == 'pi':
        scale_omega = np.pi
        lhs_label = r'\frac{i \mathrm{log} U}{\pi}'
    else:
        scale_omega = scale.pulsatance_value
        lhs_label = r'\frac{H_{\mathrm{eff}}}{\mathrm{%s}}' % scale.frequency_unit

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
        ylabel = r'$\nu\,(\mathrm{' + scale.frequency_unit + '})$'

    # Dividing by omega -> now everything is in terms of frequency (not angular)
    # Note: Don't use /=!
    components = components / scale_omega
    uncertainties = uncertainties / scale_omega

    if ignore_identity:
        components.reshape(-1)[0] = 0.
        uncertainties.reshape(-1)[0] = 0.

    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold *= -max_abs / scale_omega

    flat_indices = np.argsort(-np.abs(components.reshape(-1)))
    nterms = np.count_nonzero(np.abs(components) > threshold)
    indices = np.unravel_index(flat_indices[:nterms], components.shape)

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), components[indices], yerr=uncertainties[indices])

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
    components = paulis.components(-matrix_angle(gate)).real

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        components = paulis.truncate(components, (comp_dim,) * len(sim_result.dim))

    return components


def heff_analysis(
    hgen: HamiltonianBuilder,
    drive_def: List[Tuple[Hashable, float, complex]],
    comp_dim: int = 2,
    method: str = 'fidelity',
    method_params: Optional[Dict] = None,
    num_drive_cycles: Tuple[int, int, int] = (500, 1000, 20),
    save_result_to: Optional[str] = None,
    log_level: int = logging.WARNING
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Run a full effective Hamiltonian analysis.

    Given a HamiltonianBuilder object with the full specification offrequencies and couplings but
    no drive information, add constant drive terms, run the effective Hamiltonian extraction from
    simulations with several different durations, and compute the average and uncertainties of the
    :math:`H_{\mathrm{eff}}` components.

    The

    Args:
        hgen: Hamiltonian with no drive term.
        drive_def: A list of drive term specifications (qudit id, frequency, amplitude).

    Returns:
        An array of effective Hamiltonian components and another array representing their
        uncertainties.
    """
    max_frequency = 0.

    for qudit_id, frequency, amplitude in drive_def:
        hgen.add_drive(qudit_id, frequency=frequency, amplitude=amplitude)
        max_frequency = max(max_frequency, frequency)

    cycle = 2. * np.pi / max_frequency
    durations = np.linspace(num_drive_cycles[0] * cycle, num_drive_cycles[1] * cycle, num_drive_cycles[2])

    tlist = list({'points_per_cycle': 8, 'duration': duration} for duration in durations)

    sim_results = pulse_sim(hgen, tlist, rwa=False, save_result_to=save_result_to, log_level=log_level)

    components_list = find_heff(sim_results, comp_dim=comp_dim, method=method, method_params=method_params,
                                save_result_to=save_result_to, log_level=log_level)

    components_list = np.array(components_list)

    # duration-weighted average of components (longer-time simulation result is preferred)
    components = np.sum(components_list.reshape(durations.shape[0], -1) * durations[:, None], axis=0)
    components /= np.sum(durations)
    components = components.reshape(components_list.shape[1:])

    uncertainties = np.amax(components_list, axis=0) - np.amin(components_list, axis=0)

    return components, uncertainties
