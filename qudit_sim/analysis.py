"""Hamiltonian and gate analysis routines."""

from typing import Union, Tuple, List, Optional, Dict, Hashable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as qtp

from rqutils.math import matrix_angle
from rqutils.qprint import QPrintBraKet, QPrintPauli, LaTeXRepr
import rqutils.paulis as paulis

from .util import HamiltonianCoefficient, FrequencyScale, PulseSimResult
from .pulse_sim import pulse_sim
from .find_heff import find_heff

def print_hamiltonian(
    hamiltonian: List[Union[qtp.Qobj, Tuple[qtp.Qobj, HamiltonianCoefficient]]],
    phase_norm: Tuple[float, str]=(np.pi, 'π')
) -> LaTeXRepr:
    """IPython printer of the Hamiltonian list built by HamiltonianBuilder.

    Args:
        hamiltonian: Input list to ``qutip.sesolve``.
        phase_norm: Normalization for displayed phases.

    Returns:
        A representation object for a LaTeX ``align`` expression with one Hamiltonian term per line.
    """
    lines = []
    start = 0
    if isinstance(hamiltonian[0], qtp.Qobj):
        lines.append(QPrintBraKet(hamiltonian[0].full(), dim=hamiltonian[0].dims[0], lhs_label=r'H_{\mathrm{static}} &').latex(env=None))
        start += 1

    for iterm, term in enumerate(hamiltonian[start:]):
        lines.append(QPrintBraKet(term[0], dim=term[0].dims[0], lhs_label=f'H_{{{iterm}}} &', amp_norm=(1., fr'[\text{{{term[1]}}}]*'), phase_norm=phase_norm).latex(env=None))

    return LaTeXRepr(r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}')


def print_components(
    components: np.ndarray,
    symbol: Optional[str] = None,
    precision: int = 3,
    threshold: float = 1.e-3,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto
) -> LaTeXRepr:
    r"""Compose a LaTeX expression of the effective Hamiltonian from the Pauli components.

    Args:
        components: Array of Pauli components returned by find_heff
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        precision: Number of digits below the decimal point to show.
        threshold: Ignore terms with absolute components below this value relative to the given scale
            (if >0) or to the maximum absolute component (if <0).
        scale: Normalize the components with the frequency scale. If None, components are taken
            to be dimensionless. If `FrequencyScale.auto`, scale is found from the maximum absolute
            value of the components. String `'pi'` is also allowed, in which case the components are
            normalized by :math:`\pi`.

    Returns:
        A LaTeX expression string for the effective Hamiltonian.
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

    if threshold < 0.:
        threshold *= -max_abs / scale_omega

    pobj = QPrintPauli(components / scale_omega,
                       amp_format=f'.{precision}f',
                       epsilon=(threshold * scale_omega / max_abs),
                       lhs_label=lhs_label,
                       symbol=symbol)

    return LaTeXRepr(pobj.latex())


def plot_components(
    components: np.ndarray,
    threshold: float = 1.e-2,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto,
    ignore_identity: bool = True
) -> mpl.figure.Figure:
    """Plot the Hamiltonian components as a bar graph in the decreasing order in the absolute value.

    Args:
        components: Array of Pauli components returned by find_heff
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
    components = components.copy()

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
    components /= scale_omega

    if ignore_identity:
        components.reshape(-1)[0] = 0.

    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold *= -max_abs / scale_omega

    flat_indices = np.argsort(-np.abs(components.reshape(-1)))
    nterms = np.count_nonzero(np.abs(components) > threshold)
    indices = np.unravel_index(flat_indices[:nterms], components.shape)

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), components[indices])

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
    """Run a full effective Hamiltonian analysis.
    
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
