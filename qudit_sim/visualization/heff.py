"""Effective Hamiltonian visualization."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt
from matplotlib.markers import MarkerStyle
try:
    get_ipython()
except NameError:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = True
    from IPython.display import Latex

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from .decompositions import print_components, plot_evolution
from ..apps.heff import unitary_subtraction, heff_fidelity
from ..basis import change_basis, diagonals, matrix_labels
from ..scale import FrequencyScale
from ..sim_result import load_sim_result
from ..unitary import truncate_matrix, closest_unitary


def inspect_heff_fit(
    filename: str,
    threshold: float = 0.01,
    tscale: FrequencyScale = FrequencyScale.auto,
    align_ylim: bool = True,
    select_components: Optional[List[Tuple]] = None,
    basis: Optional[Union[str, np.ndarray]] = None,
    symbols: Optional[Union[str, List[str]]] = None,
    figures: Optional[List[str]] = None
) -> List[mpl.figure.Figure]:
    """Plot the time evolution of Pauli components before and after fidelity maximization.

    Args:
        filename: Name of the HDF5 file containing the fit result.
        threshold: Threshold for a Pauli component to be plotted, in radians. Ignored if
            limit_components is not None.
        tscale: Scale for the time axis.
        align_ylim: Whether to align the y axis limits for all plots.
        select_components: List of Pauli components to plot.
        basis: Represent the components in the given matrix basis.
        symbols: Symbols to use instead of the numeric indices for the matrices.

    Returns:
        A list of two (three if metrics=True) figures.
    """
    sim_result = load_sim_result(filename)
    with h5py.File(filename, 'r') as source:
        comp_dim = tuple(source['comp_dim'][()])
        fit_start, fit_end = source['fit_range'][()]
        components = source['components'][()]
        offset_components = source['offset_components'][()]
        fixed_indices = source['fixed'][()]

        try:
            loss = source['loss'][()]
            heff_grads = source['heff_grads'][()]
            offset_grads = source['offset_grads'][()]
            heff_compos = source['heff_compos'][()]
            offset_compos = source['offset_compos'][()]
        except KeyError:
            loss = None
            heff_grads = None
            offset_grads = None
            heff_compos = None
            offset_compos = None

    tlist = sim_result.times
    time_evolution = sim_result.states
    frame = sim_result.frame

    tlist_fit = tlist[fit_start:fit_end + 1] - tlist[fit_start]

    if tscale is FrequencyScale.auto:
        tscale = FrequencyScale.find_time_scale(tlist[-1])

    if comp_dim != frame.dim:
        time_evolution_trunc = truncate_matrix(time_evolution, frame.dim, comp_dim)
        time_evolution, trunc_fidelity = closest_unitary(time_evolution_trunc, with_fidelity=True)
    else:
        time_evolution_trunc = time_evolution
        trunc_fidelity = np.ones(tlist.shape[0])

    if basis is not None:
        components_tr = change_basis(components, to_basis=basis)
        offset_components_tr = change_basis(offset_components, to_basis=basis)
    else:
        components_tr = components
        offset_components_tr = offset_components

    if figures is None:
        figures = ['evolution', 'subtracted', 'metrics']

    figure_list = []

    if 'evolution' in figures:
        ## Original time evolution and Heff
        indices, fig_generator = plot_evolution(time_evolution=time_evolution,
                                                tlist=tlist,
                                                dim=comp_dim,
                                                threshold=threshold,
                                                select_components=select_components,
                                                eigvals=True,
                                                align_ylim=align_ylim,
                                                tscale=tscale,
                                                title='Original time evolution vs $H_{eff}$',
                                                basis=basis,
                                                symbols=symbols)
        figure_list.append(fig_generator)

        # Add lines with slope=compo for terms above threshold
        xval = (tlist_fit + tlist[fit_start]) * tscale.frequency_value
        x0 = tlist[fit_start] * tscale.frequency_value
        x1 = tlist[fit_end] * tscale.frequency_value

        for iax, index in enumerate(indices):
            ax = fig_generator.axes[iax]
            yval = components_tr[index] * tlist_fit + offset_components_tr[index]

            ax.plot(xval, yval)

        # Indicate the start-of-fit time
        for ax in fig_generator.axes:
            if ax.get_lines():
                ax.axvline(x0, linestyle='dotted', color='black', linewidth=0.5)
                ax.axvline(x1, linestyle='dotted', color='black', linewidth=0.5)

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

    if 'subtracted' in figures:
        ## Subtracted unitaries
        target = unitary_subtraction(time_evolution[fit_start:fit_end + 1], components,
                                     offset_components, tlist_fit)

        _, fig_target = plot_evolution(time_evolution=target,
                                       tlist=(tlist_fit + tlist[fit_start]),
                                       dim=comp_dim,
                                       threshold=threshold,
                                       select_components=select_components,
                                       align_ylim=align_ylim,
                                       tscale=tscale,
                                       title='Unitary-subtracted time evolution',
                                       basis=basis,
                                       symbols=symbols)
        figure_list.append(fig_target)

    if 'metrics' in figures:
        ## Fit metrics plots
        fig_metrics, axes = plt.subplots(1, 3, figsize=(16, 4))
        figure_list.append(fig_metrics)
        fig_metrics.suptitle('Fit metrics', fontsize=16)

        # fidelity
        full_fidelity = heff_fidelity(time_evolution_trunc[fit_start:fit_end + 1], components,
                                      offset_components, tlist_fit)
        xval = (tlist_fit + tlist[fit_start]) * tscale.frequency_value

        ax = axes[0]
        ax.set_title('Fidelity')
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel('fidelity')
        ax.plot(xval, trunc_fidelity[fit_start:fit_end + 1], label='Upper bound')
        ax.plot(xval, full_fidelity, label='Fidelity')
        ax.axhline(1., color='black', linewidth=0.5)
        ax.legend()

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
            ax.plot(np.amax(np.abs(heff_grads.reshape(heff_grads.shape[0], -1)), axis=1),
                    label='$H_{eff}$')
            ax.plot(np.amax(np.abs(offset_grads.reshape(offset_grads.shape[0], -1)), axis=1),
                    label='Offset')
            ax.axhline(0., color='black', linewidth=0.5)
            ax.legend()

        fig_metrics.tight_layout()

    if 'heff_values' in figures and heff_compos is not None:
        ## Fit evolution of Heff components
        indices = list(zip(*np.nonzero(np.logical_not(fixed_indices))))
        heff_compos_map = {idx: heff_compos[:, ic] for ic, idx in enumerate(indices)}
        if select_components is not None:
            indices = list(sorted(set(select_components) & set(indices)))

        num_axes = len(indices)
        nx = np.floor(np.sqrt(num_axes)).astype(int)
        nx = max(nx, 4)
        nx = min(nx, 9)
        ny = np.ceil(num_axes / nx).astype(int)
        fig_heff, axes = plt.subplots(ny, nx, figsize=(nx * 4, ny * 4))
        figure_list.append(fig_heff)
        fig_heff.suptitle('Heff components', fontsize=16)

        labels = paulis.labels(comp_dim, symbol=symbols, norm=False)

        for iax, idx in enumerate(indices):
            ax = fig_heff.axes[iax]
            ax.set_title(f'${labels[idx]}$')
            ax.plot(np.arange(len(heff_compos_map[idx])), heff_compos_map[idx])

        fig_heff.tight_layout()

    return figure_list


def plot_amplitude_scan(
    amplitudes: np.ndarray,
    components: Sequence,
    threshold: Optional[float] = None,
    amp_scale: FrequencyScale = FrequencyScale.auto,
    compo_scale: FrequencyScale = FrequencyScale.auto,
    max_poly_order: Union[int, None] = 4,
    use_all_powers: bool = False,
    auto_scale: bool = True,
    select_components: Optional[List[Tuple]] = None,
    basis: Optional[Union[str, np.ndarray]] = None,
    symbols: Optional[Union[str, List[str]]] = None
) -> Tuple[mpl.figure.Figure, np.ndarray, FrequencyScale, FrequencyScale]:
    """Plot the result of the amplitude scan.

    See the last example in examples/validation/heff.ipynb for how to prepare the inputs to this
    function.

    This function performs polynomial fits to the amplitude dependences of the Pauli components and
    returns the best-fit parameters as well as adds the fit curves to the plots. Amplitude variable
    in the polynomials are normalized to O(1) to avoid numerical errors. The third and fourth return
    values of this function represent the amplitude and component scales used for the normalization.

    Args:
        amplitudes: Array of drive amplitudes (assumed real)
        components: Returned list of effective Hamiltonian components from find_heff.
        threshold: Only plot components whose maximum is above this threshold. Defaults to
            `0.01 * compo_scale.pulsatance_value`.
        amp_scale: Scale of the drive amplitude.
        compo_scale: Scale of the components.
        max_poly_order: Maximum polynomial order for fitting the amplitude dependencies of the
            components. Set to None to skip fitting.
        use_all_powers: If True, do not restrict polynomial fit to odd / even powers
            depending on the diagonality of the Hamiltonian component.
        select_components: List of Pauli components to plot.
        basis: Represent the components in the given matrix basis.
        symbols: Symbols to use instead of the numeric indices for the matrices.

    Returns:
        Plot Figure, polynomial coefficients from the fits, list of selected components, and the amplitude
        and components scale used to normalize the amplitude variable in the polynomials.
    """
    components = np.asarray(components)

    num_qudits = len(components.shape) - 1 # first dim: amplitudes
    comp_dim = tuple(np.around(np.sqrt(components.shape[1:])).astype(int))

    if basis is not None:
        components = change_basis(components, to_basis=basis, num_qudits=num_qudits)
        if symbols is None:
            symbols = list(matrix_labels(basis, dim) for dim in comp_dim)

    if amp_scale is None:
        amp_scale_omega = 1.
    else:
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

    if max_poly_order is None:
        coefficients = None
    else:
        coefficients = list(np.zeros(max_poly_order + 1) for _ in range(len(select_components)))

    if symbols is None:
        symbols = [''] * len(comp_dim)
        delimiter = r'\,' if all(dim == 2 for dim in comp_dim) else ','
    else:
        delimiter = r'\,'

    basis_labels = paulis.labels(comp_dim, symbol=symbols, delimiter=delimiter, norm=False)

    diag_indices = list(diagonals(basis, dim) for dim in comp_dim)

    amplitudes_fine = np.linspace(amplitudes[0], amplitudes[-1], 100)
    num_amps = amplitudes.shape[0]

    filled_markers = MarkerStyle.filled_markers
    num_markers = len(filled_markers)

    ymax = 0.
    ymin = 0.

    fig, ax = plt.subplots(1, 1)

    for icompo, index in enumerate(select_components):
        label = basis_labels[index]
        is_diagonal = all(idx in diag for idx, diag in zip(index, diag_indices))

        plot_label = f'${label}$'

        if auto_scale and compo_maxima[index] > 50. * min_max:
            plot_scale = 0.1
            plot_label += r' ($\times 0.1$)'
        else:
            plot_scale = 1.

        # First axis: amplitudes
        compos = components[(slice(None),) + index]

        ymax = max(ymax, np.amax(compos) * plot_scale)
        ymin = min(ymin, np.amin(compos) * plot_scale)

        pathcol = ax.scatter(amplitudes, compos * plot_scale,
                             marker=filled_markers[icompo % num_markers], label=plot_label)

        if max_poly_order is None:
            continue

        # Perform a polynomial fit
        if use_all_powers:
            curve = _poly_all
            p0 = np.zeros(max_poly_order + 1)
        elif is_diagonal:
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
        except sciopt.OptimizeWarning:
            logging.warning(f'Covariance of the fit parameters for {label} could not be determined.')

        if use_all_powers:
            coefficients[icompo] = popt
        elif is_diagonal:
            coefficients[icompo][::2] = popt
        else:
            coefficients[icompo][1::2] = popt

        ax.plot(amplitudes_fine, curve(amplitudes_fine, *popt) * plot_scale, color=pathcol.get_edgecolor())

    ax.set_ylim(ymin * 1.2, ymax * 1.2)
    ax.grid(True)
    # Since amp and nu are normalized by (2*pi*frequency), displayed values are frequencies
    xlabel = 'Drive amplitude'
    if amp_scale is not None:
        xlabel += fr' ($2\pi\,\mathrm{{{amp_scale.frequency_unit}}}$)'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(fr'$\nu$ ($2\pi\,\mathrm{{{compo_scale.frequency_unit}}}$)')
    ax.legend()

    fig.tight_layout()

    return fig, coefficients, select_components, amp_scale, compo_scale


def _poly_even(x, *args):
    coeffs = np.zeros(2 * len(args) - 1)
    coeffs[0::2] = args
    return np.polynomial.polynomial.polyval(x, coeffs)

def _poly_odd(x, *args):
    coeffs = np.zeros(2 * len(args))
    coeffs[1::2] = args
    return np.polynomial.polynomial.polyval(x, coeffs)

def _poly_all(x, *args):
    return np.polynomial.polynomial.polyval(x, args)

if HAS_IPYTHON:
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
        if symbols is None:
            basis_label = ','.join(f'{i}' for i in index)
        else:
            basis_label = ''.join(symbols[i] for i in index)

        if HAS_IPYTHON:
            line = fr'\frac{{\nu_{{{basis_label}}}}}{{2\pi\,\mathrm{{{compo_scale.frequency_unit}}}}} &='
        else:
            line = f'nu[{basis_label}]/(2π {compo_scale.frequency_unit}) ='

        for order, p in enumerate(coeff):
            if p == 0.:
                continue

            pstr = f'{abs(p):.2e}'

            if HAS_IPYTHON:
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

    if HAS_IPYTHON:
        linebreak = r' \\ '
        expr = Latex(fr'\begin{{align}}{linebreak.join(lines)}\end{{align}} A: amplitude in $2\pi\,\mathrm{{{amp_scale.frequency_unit}}}$')
    else:
        expr = '\n'.join(lines) + f'\nA: amplitude in 2π {amp_scale.frequency_unit}\n'

    return expr
