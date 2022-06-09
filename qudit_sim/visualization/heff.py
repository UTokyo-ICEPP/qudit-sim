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
from ..util import FrequencyScale
from .decompositions import print_components, plot_time_evolution

def inspect_heff_fit(
    filename: str,
    threshold: float = 0.01,
    tscale: FrequencyScale = FrequencyScale.auto,
    align_ylim: bool = False,
    select_components: Optional[List[Tuple]] = None,
    digest: bool = True
) -> List[mpl.figure.Figure]:
    """Plot the time evolution of Pauli components before and after fidelity maximization.

    Args:
        filename: Name of the HDF5 file containing the fit result.
        threshold: Threshold for a Pauli component to be plotted, in radians. Ignored if
            limit_components is not None.
        tscale: Scale for the time axis.
        align_ylim: Whether to align the y axis limits for all plots.
        limit_components: List of Pauli components to plot.
        digest: If True, a figure for fit information is included.

    Returns:
        A list of two (three if digest=True) figures.
    """
    with h5py.File(filename, 'r') as source:
        num_qudits = int(source['num_qudits'][()])
        num_sim_levels = int(source['num_sim_levels'][()])
        comp_dim = int(source['comp_dim'][()])
        time_evolution = source['time_evolution'][()]
        tlist = source['tlist'][()]
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

    dim = (num_sim_levels,) * num_qudits
    tlist_fit = tlist[fit_start:] - tlist[fit_start]

    figures = []

    ## First figure: original time evolution and Heff
    indices, fig_ilogu = plot_time_evolution(time_evolution=time_evolution,
                                             tlist=tlist,
                                             dim=dim,
                                             threshold=threshold,
                                             select_components=select_components,
                                             align_ylim=align_ylim,
                                             tscale=tscale)
    figures.append(fig_ilogu)

    fig_ilogu.suptitle(r'Original time evolution vs $H_{eff}$', fontsize=24)

    # Add lines with slope=compo for terms above threshold
    if tscale is FrequencyScale.auto:
        tscale = FrequencyScale.find_time_scale(tlist[-1])

    for iax, index in enumerate(indices):
        ax = fig_ilogu.axes[iax]
        xval = (tlist_fit + tlist[fit_start]) * tscale.frequency_value
        yval = components[index] * tlist_fit + offset_components[index]
        x0 = tlist[fit_start] * tscale.frequency_value

        ax.plot(xval, yval)
        ax.axvline(x0, linestyle='--', color='black', linewidth=0.5)

        if np.all(np.array(index) < comp_dim ** 2):
            for spine in ax.spines.values():
                spine.set(linewidth=2.)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    handles = [
        mpl.lines.Line2D([0.], [0.], color=colors[0]),
        mpl.lines.Line2D([0.], [0.], color=colors[1])
    ]
    labels = [
        'ilogU(t) component',
        '$H_{eff}t$ component $\pm$ offset'
    ]
    fig_ilogu.legend(handles, labels, 'upper right')

    ## Second figure: subtracted unitaries
    target = unitary_subtraction(time_evolution[fit_start:], components, offset_components, tlist_fit)

    indices, fig_target = plot_time_evolution(time_evolution=target,
                                              tlist=(tlist_fit + tlist[fit_start]),
                                              dim=dim,
                                              threshold=threshold,
                                              select_components=select_components,
                                              align_ylim=align_ylim,
                                              tscale=tscale)
    figures.append(fig_target)

    fig_target.suptitle('Unitary-subtracted time evolution', fontsize=24)

    # Highlight the plots within the computational dimensions
    for iax, index in enumerate(indices):
        ax = fig_ilogu.axes[iax]

        if np.all(np.array(index) < comp_dim ** 2):
            for spine in ax.spines.values():
                spine.set(linewidth=2.)

    if digest:
        ## Third figure: fit digest plots
        fig_digest, axes = plt.subplots(1, 4, figsize=(16, 4))
        figures.append(fig_digest)
        fig_digest.suptitle('Fit metrics', fontsize=24)

        # fidelity
        final_fidelity = heff_fidelity(time_evolution[fit_start:], components, offset_components, tlist_fit)
        ax = axes[0]
        ax.set_title('Final fidelity')
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel('fidelity')
        ax.plot(tlist_fit + tlist[fit_start], final_fidelity)
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
            ax.plot(np.amax(np.abs(grad), axis=1))
            ax.axhline(0., color='black', linewidth=0.5)

    for fig in figures:
        fig.tight_layout()

    return figures


def plot_amplitude_scan(
    amplitudes: np.ndarray,
    components: Sequence,
    threshold: Optional[float] = None,
    amp_scale: FrequencyScale = FrequencyScale.auto,
    compo_scale: FrequencyScale = FrequencyScale.auto,
    max_poly_order: int = 4
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

    Returns:
        Plot Figure, polynomial coefficients from the fits, and the amplitude and components scale used to
        normalize the amplitude variable in the polynomials.
    """
    components = np.asarray(components)

    num_qudits = len(components.shape) - 1 # first dim: amplitudes
    comp_dim = int(np.around(np.sqrt(components.shape[1])))
    num_paulis = comp_dim ** 2

    if amp_scale is FrequencyScale.auto:
        amp_scale = FrequencyScale.find_energy_scale(np.amax(np.abs(amplitudes)))

    amp_scale_omega = amp_scale.pulsatance_value
    amps_norm = amplitudes / amp_scale_omega

    if compo_scale is FrequencyScale.auto:
        compo_scale = FrequencyScale.find_energy_scale(np.amax(np.abs(components)))

    compo_scale_omega = compo_scale.pulsatance_value
    compos_norm = components / compo_scale_omega

    if threshold is None:
        threshold = compo_scale_omega * 1.e-2

    threshold /= compo_scale_omega

    # Amplitude, compo, and threshold are all divided by omega and are in terms of frequency

    amax = np.amax(np.abs(compos_norm), axis=0)
    num_above_threshold = np.count_nonzero(amax > threshold)
    min_max = np.amin(np.where(amax > threshold, amax, np.amax(amax)))

    # Mask to apply to the components array

    plot_mask = np.where(amax > threshold, 1, 0)
    plot_mask = np.where(amax > 50. * min_max, 2, plot_mask)

    # Array for polynomial fit results

    coefficients = np.zeros(components.shape[1:] + (max_poly_order + 1,), dtype=float)

    if num_qudits > 1 and num_above_threshold > 6:
        # Which Pauli components of the first qudit have plots to draw?
        has_passing_compos = np.any(amax.reshape(num_paulis, -1) > threshold, axis=1)
        num_plots = np.sum(has_passing_compos)

        nv = np.ceil(np.sqrt(num_plots)).astype(int)
        nh = np.ceil(num_plots / nv).astype(int)
        fig, axes = plt.subplots(nv, nh, figsize=(16, 12))

        prefixes = paulis.labels(comp_dim, symbol='', norm=False)

        iax = 0

        for ip in range(num_paulis):
            if not has_passing_compos[ip]:
                continue

            ax = axes.reshape(-1)[iax]
            _plot_amplitude_scan_on(ax, amps_norm, compos_norm[:, ip], plot_mask[ip],
                                    max_poly_order, coefficients[ip], prefix=prefixes[ip])

            iax += 1

    else:
        num_plots = 1
        fig, axes = plt.subplots(1, 1, squeeze=False)
        _plot_amplitude_scan_on(axes[0, 0], amps_norm, compos_norm, plot_mask, max_poly_order, coefficients)

    cmax = np.amax(compos_norm, axis=0)
    cmin = np.amin(compos_norm, axis=0)
    ymax = np.amax(np.where(plot_mask == 2, cmax * 0.1, cmax))
    ymin = np.amin(np.where(plot_mask == 2, cmin * 0.1, cmin))

    for ax in axes.reshape(-1)[:num_plots]:
        ax.set_ylim(ymin * 1.2, ymax * 1.2)
        ax.grid(True)
        # Since amp and nu are normalized by (2*pi*frequency), displayed values are frequencies
        ax.set_xlabel(fr'Drive amplitude ($2\pi\,\mathrm{{{amp_scale.frequency_unit}}}$)')
        ax.set_ylabel(fr'$\nu$ ($2\pi\,\mathrm{{{compo_scale.frequency_unit}}}$)')
        ax.legend()

    fig.tight_layout()

    return fig, coefficients, amp_scale, compo_scale


def _plot_amplitude_scan_on(ax, amps_norm, compos_norm, plot_mask, max_poly_order, coefficients, prefix=''):
    num_qudits = len(compos_norm.shape) - 1 # first dimension: amplitudes
    comp_dim = int(np.around(np.sqrt(compos_norm.shape[1])))

    basis_labels = paulis.labels((comp_dim,) * num_qudits, symbol='',
                                 delimiter=('' if comp_dim == 2 else ','), norm=False)

    amps_norm_fine = np.linspace(amps_norm[0], amps_norm[-1], 100)
    num_amps = amps_norm.shape[0]

    filled_markers = MarkerStyle.filled_markers
    num_markers = len(filled_markers)

    imarker = 0

    for index in np.ndindex(compos_norm.shape[1:]):
        if plot_mask[index] == 0:
            continue

        if comp_dim == 2:
            label = prefix + basis_labels[index]
        else:
            if prefix:
                label = f'{prefix},{basis_labels[index]}'
            else:
                label = basis_labels[index]

        plot_label = f'${label}$'

        if plot_mask[index] == 2:
            plot_scale = 0.1
            plot_label += r' ($\times 0.1$)'
        else:
            plot_scale = 1.

        compos = compos_norm[(slice(None),) + index]

        pathcol = ax.scatter(amps_norm, compos * plot_scale, marker=filled_markers[imarker % num_markers], label=plot_label)

        imarker += 1

        # Perform a polynomial fit

        even = np.sum(compos[:num_amps // 2] * compos[-num_amps // 2:]) > 0.

        if even:
            curve = _poly_even
            p0 = np.zeros(max_poly_order // 2 + 1)
        else:
            curve = _poly_odd
            p0 = np.zeros((max_poly_order + 1) // 2)

        try:
            popt, _ = sciopt.curve_fit(curve, amps_norm, compos, p0=p0)
        except RuntimeError:
            logging.warning(f'Components for {label} could not be fit with an order {max_poly_order} polynomial.')
            continue
        except OptimizeWarning:
            logging.warning(f'Covariance of the fit parameters for {label} could not be determined.')

        if even:
            coefficients[index][::2] = popt
        else:
            coefficients[index][1::2] = popt

        ax.plot(amps_norm_fine, curve(amps_norm_fine, *popt) * plot_scale, color=pathcol.get_edgecolor())

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
    coefficients: np.ndarray,
    amp_scale: FrequencyScale,
    compo_scale: FrequencyScale
) -> print_type:
    """Print a LaTeX expression of the amplitude scan fit results.

    Args:
        coefficients: array of polynomial coefficients given by plot_amplitude_scan.
        amp_scale: Amplitude normalization scale.
        compo_scale: Pauli components normalization scale.

    Returns:
        A LaTeX representation of the polynomials.
    """
    num_qudits = len(coefficients.shape) - 1 # last dimension is for polynomial coefficients
    comp_dim = int(np.around(np.sqrt(coefficients.shape[0])))

    basis_labels = paulis.labels((comp_dim,) * num_qudits, symbol='',
                                 delimiter=('' if comp_dim == 2 else ','), norm=False)

    poly_order = coefficients.shape[-1]

    lines = []

    for index in np.ndindex(coefficients.shape[:-1]):
        if np.allclose(coefficients[index], np.zeros(poly_order)):
            continue

        if has_ipython:
            line = fr'\frac{{\nu_{{{basis_labels[index]}}}}}{{2\pi\,\mathrm{{{compo_scale.frequency_unit}}}}} &='
        else:
            line = f'nu[{basis_labels[index]}]/(2π {compo_scale.frequency_unit}) ='

        for order, p in enumerate(coefficients[index]):
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
