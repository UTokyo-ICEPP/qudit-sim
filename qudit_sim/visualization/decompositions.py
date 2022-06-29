"""Visualizations of Pauli decompositions of Hamiltonians and gates."""

from typing import Union, List, Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    get_ipython()
except NameError:
    has_ipython = False
    PrintReturnType = str
else:
    has_ipython = True
    from IPython.display import Latex
    PrintReturnType = Latex

from rqutils.math import matrix_angle
from rqutils.qprint import QPrintPauli
import rqutils.paulis as paulis

from ..sim_result import PulseSimResult
from ..scale import FrequencyScale

def print_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    symbol: Optional[str] = None,
    precision: int = 3,
    threshold: float = 1.e-3,
    lhs_label: Optional[str] = None,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto
) -> PrintReturnType:
    r"""Compose a LaTeX expression of the effective Hamiltonian from the Pauli components.

    Args:
        components: Array of Pauli components returned by find_heff.
        uncertainties: Array of component uncertainties.
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        precision: Number of digits below the decimal point to show.
        threshold: Ignore terms with absolute components below this value relative to the given scale
            (if >0) or to the maximum absolute component (if <0).
        lhs_label: Left-hand-side label.
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
    elif isinstance(scale, tuple):
        scale_omega = scale[0]
        if lhs_label is None:
            lhs_label = fr'\frac{{i \mathrm{{log}} U}}{{{scale[1]}}}'
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
        ylabel = r'$\theta$'
    elif scale == 'pi':
        scale_omega = np.pi
        ylabel = r'$\theta/\pi$'
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


def plot_evolution(
    sim_result: Optional[PulseSimResult] = None,
    time_evolution: Optional[np.ndarray] = None,
    tlist: Optional[np.ndarray] = None,
    dim: Optional[Tuple[int, ...]] = None,
    differential: bool = False,
    threshold: float = 0.01,
    select_components: Optional[List[Tuple[int, ...]]] = None,
    eigvals: bool = True,
    align_ylim: bool = True,
    tscale: Optional[FrequencyScale] = FrequencyScale.auto,
    fig: Optional[mpl.figure.Figure] = None,
    title: str = ''
) -> Tuple[List[Tuple[int, ...]], mpl.figure.Figure]:
    r"""Plot the Pauli components of the generator of a time evolution as a function of time.

    The time evolution, time points, and the operator dimension can either be passed as a simulation
    result object or individually.

    Args:
        sim_result: Simulation result object. If not None, ``time_evolution``, ``tlist``, and ``dim``
            are ignored.
        time_evolution: Time evolution unitaries.
        tlist: Time points.
        dim: Operator dimension.
        differential: If True, plot the differential of the time evolution, i.e.
            :math:`U_{H}(t_i) U_{H}(t_{i-1})^{\dagger}`.
        threshold: Only the Pauli components whose values exceed this value are plotted. Ignored if
            ``select_components`` is not None.
        select_components: List of indices of the components to plot.
        eigvals: If True, add a plot of the generator eigenvalue evolution.
        align_ylim: If True, the vertical axis limits are aligned over all plots.
        tscale: Time scale.
        fig: Figure to add the plots into.
        title: Title of the figure.

    Returns:
        The indices of the plotted components and the plot figure.
    """

    if sim_result is not None:
        time_evolution = sim_result.states
        tlist = sim_result.times
        dim = sim_result.dim

    if tscale is FrequencyScale.auto:
        tscale = FrequencyScale.find_time_scale(tlist[-1])

    if tscale is not None:
        tlist = tlist * tscale.frequency_value

    if differential:
        time_evolution = time_evolution[1:] @ time_evolution[:-1].transpose((0, 2, 1)).conjugate()
        tlist = tlist[1:]

    generator, ev = matrix_angle(time_evolution, with_diagonals=True)
    components = paulis.components(-1. * generator, dim=dim).real
    components = np.moveaxis(components, 0, -1)

    if select_components is None:
        # Make a list of tuples from a tuple of arrays
        select_components = list(zip(*np.nonzero(np.amax(np.abs(components), axis=-1) > threshold)))

    num_axes = len(select_components)
    if eigvals:
        num_axes += 1

    if num_axes == 0:
        if fig is None:
            fig = plt.figure()

        return select_components, fig

    nx = np.floor(np.sqrt(num_axes)).astype(int)
    nx = max(nx, 4)
    nx = min(nx, 9)
    ny = np.ceil(num_axes / nx).astype(int)

    if fig is None:
        fig, _ = plt.subplots(ny, nx, figsize=(nx * 4, ny * 4))
    else:
        fig.set_figheight(ny * 4.)
        fig.set_figwidth(nx * 4.)
        fig.subplots(ny, nx)

    if len(select_components) > 0:
        labels = paulis.labels(dim, norm=False)

        if align_ylim:
            indices_array = np.array(tuple(zip(select_components)))
            selected_compos = components[indices_array]
            ymax = np.amax(selected_compos)
            ymin = np.amin(selected_compos)
            vrange = ymax - ymin
            ymax += 0.2 * vrange
            ymin -= 0.2 * vrange

        for iax, index in enumerate(select_components):
            ax = fig.axes[iax]

            ax.set_title(f'${labels[index]}$')
            ax.plot(tlist, components[index])

            ax.axhline(0., color='black', linewidth=0.5)
            if align_ylim:
                ax.set_ylim(ymin, ymax)
            ax.set_ylabel('rad')

    if eigvals:
        ax = fig.axes[len(select_components)]

        ev = np.sort(-1. * ev, axis=1)

        ax.set_title('Generator eigenvalues')
        ax.plot(tlist, ev)

        for y in [-np.pi, 0., np.pi]:
            ax.axhline(y, color='black', linewidth=0.5, linestyle='dashed')
        ax.set_ylabel('rad')

    for ax in fig.axes:
        if not ax.get_lines():
            continue

        if tscale is None:
            ax.set_xlabel('t')
        else:
            ax.set_xlabel(f't ({tscale.time_unit})')

    if title:
        fig.suptitle(title, fontsize=20)

    fig.tight_layout(rect=[0., 0., 1., 0.98])

    return select_components, fig
