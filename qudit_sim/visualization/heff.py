"""Effective Hamiltonian visualization."""

from typing import Tuple, List, Sequence, Optional, Dict, Hashable, Union, Any
import logging
import numpy as np
import h5py
import scipy.optimize as sciopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

try:
    get_ipython()
except NameError:
    has_ipython = False
else:
    has_ipython = True
    from IPython.display import Latex

import rqutils.paulis as paulis
from rqutils.math import matrix_angle

from ..apps.heff_tools import unitary_subtraction, heff_fidelity
from ..scale import FrequencyScale
from ..basis import change_basis, diagonals, matrix_labels
from .decompositions import print_components, plot_evolution

def inspect_heff_fit(
    filename: str,
    threshold: float = 0.01,
    tscale: FrequencyScale = FrequencyScale.auto,
    align_ylim: bool = True,
    select_components: Optional[List[Tuple]] = None,
    metrics: bool = True,
    basis: Optional[Union[str, np.ndarray]] = None,
    symbol: Optional[str] = None
) -> List[mpl.figure.Figure]:
    """Plot the time evolution of Pauli components before and after fidelity maximization.

    Args:
        filename: Name of the HDF5 file containing the fit result.
        threshold: Threshold for a Pauli component to be plotted, in radians. Ignored if
            limit_components is not None.
        tscale: Scale for the time axis.
        align_ylim: Whether to align the y axis limits for all plots.
        select_components: List of Pauli components to plot.
        metrics: If True, a figure for fit information is included.
        basis: Represent the components in the given matrix basis.
        symbol: Symbol to use instead of the numeric indices for the matrices.

    Returns:
        A list of two (three if metrics=True) figures.
    """
    with h5py.File(filename, 'r') as source:
        dim = tuple(source['dim'][()])
        num_qudits = len(dim)
        num_sim_levels = dim[0]
        time_evolution = source['states'][()]
        tlist = source['times'][()]
        comp_dim = int(source['comp_dim'][()])
        fit_start = source['fit_start'][()]
        if num_sim_levels != comp_dim:
            components = source['components_original'][()]
        else:
            components = source['components'][()]

        offset_components = source['offset_components'][()]

        try:
            loss = source['loss'][()]
            grad = source['grad'][()]
        except KeyError:
            loss = None
            grad = None

    tlist_fit = tlist[fit_start:] - tlist[fit_start]

    if tscale is FrequencyScale.auto:
        tscale = FrequencyScale.find_time_scale(tlist[-1])

    figures = []

    ## First figure: original time evolution and Heff
    indices, fig_generator = plot_evolution(time_evolution=time_evolution,
                                            tlist=tlist,
                                            dim=dim,
                                            threshold=threshold,
                                            select_components=select_components,
                                            eigvals=True,
                                            align_ylim=align_ylim,
                                            tscale=tscale,
                                            title='Original time evolution vs $H_{eff}$',
                                            basis=basis,
                                            symbol=symbol)
    figures.append(fig_generator)

    # Add lines with slope=compo for terms above threshold
    xval = (tlist_fit + tlist[fit_start]) * tscale.frequency_value
    x0 = tlist[fit_start] * tscale.frequency_value

    components_orig = components
    offset_components_orig = offset_components

    if basis is not None:
        components = change_basis(components, basis)
        offset_components = change_basis(offset_components, basis)

    for iax, index in enumerate(indices):
        ax = fig_generator.axes[iax]
        yval = components[index] * tlist_fit + offset_components[index]

        ax.plot(xval, yval)

    # Indicate the start-of-fit time
    for ax in fig_generator.axes:
        if ax.get_lines():
            ax.axvline(x0, linestyle='dotted', color='black', linewidth=0.5)

    _highlight_comp_dim_components(fig_generator, indices, comp_dim, basis)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    handles = [
        mpl.lines.Line2D([0.], [0.], color=colors[0]),
        mpl.lines.Line2D([0.], [0.], color=colors[1])
    ]
    labels = [
        'Generator component',
        '$H_{eff}t$ component'
    ]
    fig_generator.legend(handles, labels, 'upper right')

    ## Second figure: subtracted unitaries
    target = unitary_subtraction(time_evolution[fit_start:], components_orig, offset_components_orig, tlist_fit)

    indices, fig_target = plot_evolution(time_evolution=target,
                                         tlist=(tlist_fit + tlist[fit_start]),
                                         dim=dim,
                                         threshold=threshold,
                                         select_components=select_components,
                                         align_ylim=align_ylim,
                                         tscale=tscale,
                                         title='Unitary-subtracted time evolution',
                                         basis=basis,
                                         symbol=symbol)
    figures.append(fig_target)

    _highlight_comp_dim_components(fig_target, indices, comp_dim, basis)

    if metrics:
        ## Third figure: fit metrics plots
        fig_metrics, axes = plt.subplots(1, 3, figsize=(16, 4))
        figures.append(fig_metrics)
        fig_metrics.suptitle('Fit metrics', fontsize=16)

        # fidelity
        final_fidelity = heff_fidelity(time_evolution[fit_start:], components_orig, offset_components_orig, tlist_fit)
        ax = axes[0]
        ax.set_title('Final fidelity')
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel('fidelity')
        ax.plot((tlist_fit + tlist[fit_start]) * tscale.frequency_value, final_fidelity)
        ax.axhline(1., color='black', linewidth=0.5)

        ## Intermediate data is not available if minuit is used
        if loss is not None:
            ax = axes[1]
            ax.set_title('Loss evolution')
            ax.set_xlabel('steps')
            ax.set_ylabel('loss')
            ax.plot(loss)
            ax.axhline(0., color='black', linewidth=0.5)

            ax = axes[2]
            ax.set_title('Gradient evolution')
            ax.set_xlabel('steps')
            ax.set_ylabel('max(abs(grad))')
            ax.plot(np.amax(np.abs(grad.reshape(grad.shape[0], -1)), axis=1))
            ax.axhline(0., color='black', linewidth=0.5)

        fig_metrics.tight_layout()

    return figures


def _highlight_comp_dim_components(fig, indices, comp_dim, basis):
    for iax, index in enumerate(indices):
        ax = fig.axes[iax]

        if basis is None or basis == 'gell-mann':
            if np.all(np.array(index) < comp_dim ** 2):
                for spine in ax.spines.values():
                    spine.set(linewidth=2.)

        else:
            # Not implemented yet
            pass


def plot_amplitude_scan(
    amplitudes: np.ndarray,
    components: Sequence,
    threshold: Optional[float] = None,
    amp_scale: FrequencyScale = FrequencyScale.auto,
    compo_scale: FrequencyScale = FrequencyScale.auto,
    max_poly_order: int = 4,
    select_components: Optional[List[Tuple]] = None,
    basis: Optional[Union[str, np.ndarray]] = None,
    symbol: Optional[str] = None
) -> Tuple[mpl.figure.Figure, np.ndarray, FrequencyScale, FrequencyScale]:
    """Plot the result of the amplitude scan.

    See the last example in examples/validation/heff.ipynb for how to prepare the inputs to this function.

    This function performs polynomial fits to the amplitude dependences of the Pauli components and returns
    the best-fit parameters as well as adds the fit curves to the plots. Amplitude variable in the polynomials
    are normalized to O(1) to avoid numerical errors. The third and fourth return values of this function
    represent the amplitude and component scales used for the normalization.

    Args:
        amplitudes: Array of drive amplitudes (assumed real)
        components: Returned list of effective Hamiltonian components from find_heff.
        threshold: Only plot components whose maximum is above this threshold. Defaults to
            `0.01 * compo_scale.pulsatance_value`.
        amp_scale: Scale of the drive amplitude.
        compo_scale: Scale of the components.
        max_poly_order: Maximum polynomial order for fitting the amplitude dependencies of the components.
        select_components: List of Pauli components to plot.
        basis: Represent the components in the given matrix basis.
        symbol: Symbol to use instead of the numeric indices for the matrices.

    Returns:
        Plot Figure, polynomial coefficients from the fits, list of selected components, and the amplitude
        and components scale used to normalize the amplitude variable in the polynomials.
    """
    components = np.asarray(components)

    num_qudits = len(components.shape) - 1 # first dim: amplitudes
    comp_dim = int(np.around(np.sqrt(components.shape[1])))
    num_paulis = comp_dim ** 2

    if basis is not None:
        components = change_basis(components, basis, num_qudits=num_qudits)
        if symbol is None:
            symbol = matrix_labels(basis, comp_dim)

    if amp_scale is FrequencyScale.auto:
        amp_scale = FrequencyScale.find_energy_scale(np.amax(np.abs(amplitudes)))

    amp_scale_omega = amp_scale.pulsatance_value
    amplitudes = amplitudes / amp_scale_omega

    if compo_scale is FrequencyScale.auto:
        compo_scale = FrequencyScale.find_energy_scale(np.amax(np.abs(components)))

    compo_scale_omega = compo_scale.pulsatance_value
    components = components / compo_scale_omega

    # Amplitudes and components are all divided by omega and are in terms of frequency

    compo_maxima = np.amax(np.abs(components), axis=0)

    if select_components is None:
        if threshold is None:
            threshold = compo_scale_omega * 1.e-2

        threshold /= compo_scale_omega

        select_components = list(zip(*np.nonzero(compo_maxima > threshold)))

    min_max = min(compo_maxima[index] for index in select_components)

    coefficients = list(np.zeros(max_poly_order + 1) for _ in range(len(select_components)))

    pauli_dim = (comp_dim,) * num_qudits

    if symbol is None:
        symbol = ''
        delimiter = r'\,' if comp_dim == 2 else ','
    else:
        delimiter = r'\,'

    symbol = [symbol] * num_qudits

    basis_labels = paulis.labels(pauli_dim, symbol=symbol, delimiter=delimiter, norm=False)

    diag_indices = diagonals(basis, comp_dim)

    amplitudes_fine = np.linspace(amplitudes[0], amplitudes[-1], 100)
    num_amps = amplitudes.shape[0]

    filled_markers = MarkerStyle.filled_markers
    num_markers = len(filled_markers)

    ymax = 0.
    ymin = 0.

    fig, ax = plt.subplots(1, 1)

    for icompo, index in enumerate(select_components):
        label = basis_labels[index]
        is_diagonal = all(idx in diag_indices for idx in index)

        plot_label = f'${label}$'

        if compo_maxima[index] > 50. * min_max:
            plot_scale = 0.1
            plot_label += r' ($\times 0.1$)'
        else:
            plot_scale = 1.

        # First axis: amplitudes
        compos = components[(slice(None),) + index]

        ymax = max(ymax, np.amax(compos) * plot_scale)
        ymin = min(ymin, np.amin(compos) * plot_scale)

        pathcol = ax.scatter(amplitudes, compos * plot_scale, marker=filled_markers[icompo % num_markers], label=plot_label)

        # Perform a polynomial fit
        if is_diagonal:
            curve = _poly_even
            p0 = np.zeros(max_poly_order // 2 + 1)
        else:
            curve = _poly_odd
            p0 = np.zeros((max_poly_order + 1) // 2)

        try:
            popt, _ = sciopt.curve_fit(curve, amplitudes, compos, p0=p0)
        except RuntimeError:
            logging.warning(f'Components for {label} could not be fit with an order {max_poly_order} polynomial.')
            continue
        except OptimizeWarning:
            logging.warning(f'Covariance of the fit parameters for {label} could not be determined.')

        if is_diagonal:
            coefficients[icompo][::2] = popt
        else:
            coefficients[icompo][1::2] = popt

        ax.plot(amplitudes_fine, curve(amplitudes_fine, *popt) * plot_scale, color=pathcol.get_edgecolor())

    ax.set_ylim(ymin * 1.2, ymax * 1.2)
    ax.grid(True)
    # Since amp and nu are normalized by (2*pi*frequency), displayed values are frequencies
    ax.set_xlabel(fr'Drive amplitude ($2\pi\,\mathrm{{{amp_scale.frequency_unit}}}$)')
    ax.set_ylabel(fr'$\nu$ ($2\pi\,\mathrm{{{compo_scale.frequency_unit}}}$)')
    ax.legend()

    fig.tight_layout()

    return fig, coefficients, select_components, amp_scale, compo_scale


def _poly_even(x, *args):
    value = args[0]
    for iarg, arg in enumerate(args[1:]):
        value += arg * np.power(x, 2 * (iarg + 1))
    return value

def _poly_odd(x, *args):
    value = 0.
    for iarg, arg in enumerate(args):
        value += arg * np.power(x, 2 * iarg + 1)
    return value


if has_ipython:
    print_type = Latex
else:
    print_type = str

def print_amplitude_scan(
    coefficients: List[np.ndarray],
    select_components: List[Tuple[int, ...]],
    amp_scale: FrequencyScale,
    compo_scale: FrequencyScale,
    symbols: Optional[List[str]] = None
) -> print_type:
    """Print a LaTeX expression of the amplitude scan fit results.

    Args:
        coefficients: Array of polynomial coefficients given by plot_amplitude_scan.
        select_components: List of indices of the Pauli components to print.
        amp_scale: Amplitude normalization scale.
        compo_scale: Pauli components normalization scale.
        symbols: Symbols to use instead of the numeric indices for the matrices.

    Returns:
        A LaTeX representation of the polynomials.
    """
    poly_order = coefficients[0].shape[0]

    lines = []

    for coeff, index in zip(coefficients, select_components):
        if symbol is None:
            basis_label = ','.join(f'{i}' for i in index)
        else:
            basis_label = ''.join(symbols[i] for i in index)

        if has_ipython:
            line = fr'\frac{{\nu_{{{basis_label}}}}}{{2\pi\,\mathrm{{{compo_scale.frequency_unit}}}}} &='
        else:
            line = f'nu[{basis_label}]/(2π {compo_scale.frequency_unit}) ='

        for order, p in enumerate(coeff):
            if p == 0.:
                continue

            pstr = f'{abs(p):.2e}'

            if has_ipython:
                epos = pstr.index('e')
                power = int(pstr[epos + 1:])
                if power == -1:
                    pexp = f'{abs(p):.3f}'
                elif power == 0:
                    pexp = f'{abs(p):.2f}'
                elif power == 1:
                    pexp = f'{abs(p):.1f}'
                else:
                    pexp = fr'\left({pstr[:epos]} \times 10^{{{power}}}\right)'
            else:
                pexp = pstr

            if p < 0.:
                pexp = f'-{pexp}'

            if order == 0:
                line += pexp
            elif order == 1:
                line += f'{pexp} A'
            else:
                if p > 0.:
                    pexp = f'+ {pexp}'
                line += f'{pexp} A^{{{order}}}'

        lines.append(line)

    if has_ipython:
        linebreak = r' \\ '
        expr = Latex(fr'\begin{{align}}{linebreak.join(lines)}\end{{align}} A: amplitude in $2\pi\,\mathrm{{{amp_scale.frequency_unit}}}$')
    else:
        expr = '\n'.join(lines) + f'\nA: amplitude in 2π {amp_scale.frequency_unit}\n'

    return expr
