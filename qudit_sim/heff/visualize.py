"""Visualization routines for effective Hamiltonian studies."""

from typing import Optional, Tuple, List, Sequence
import logging
import numpy as np
import h5py
import scipy.optimize as sciopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import rqutils.paulis as paulis
from rqutils.math import matrix_exp, matrix_angle
from rqutils.qprint import QPrintPauli, LaTeXRepr

from ..util import FrequencyScale
from .common import make_heff_t

twopi = 2. * np.pi

def print_components(
    components: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = None,
    scale: Optional[FrequencyScale] = None
) -> LaTeXRepr:
    """Generate a LaTeX expression of the effective Hamiltonian from the Pauli components.

    The dymamic range of the numerical values of the components is set by the maximum absolute
    value. For example, if the maximum absolute value is between 1.e+6 and 1.e+9, the components
    are expressed in MHz, with the minimum of 0.001 MHz. Pauli terms whose components have
    absolute values below the threshold are ignored.

    Args:
        components: Array of Pauli components returned by find_heff
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        threshold: Ignore terms with absolute components below this value.
        scale: Manually set the scale.

    Returns:
        A LaTeX expression string for the effective Hamiltonian.
    """
    max_abs = np.amax(np.abs(components))

    if scale is None:
        scale = _find_scale(max_abs)

    scale_omega = scale.pulsatance_value

    if threshold is None:
        threshold = scale_omega * 1.e-2

    lhs_label = r'\frac{H_{\mathrm{eff}}}{\mathrm{%s}}' % scale.name

    pobj = QPrintPauli(components / scale_omega,
                       epsilon=(threshold / max_abs),
                       lhs_label=lhs_label,
                       symbol=symbol)

    return LaTeXRepr(pobj.latex())


def plot_components(
    components: np.ndarray,
    threshold: Optional[float] = None,
    scale: Optional[FrequencyScale] = None,
    ignore_identity: bool = True
) -> mpl.figure.Figure:
    """Plot the Hamiltonian components as a bar graph in the decreasing order in the absolute value.

    The dymamic range of the numerical values of the components is set by the maximum absolute
    value. For example, if the maximum absolute value is between 1.e+6 and 1.e+9, the components
    are expressed in MHz, with the minimum of 0.001 MHz. Pauli terms whose components have
    absolute values below the threshold are ignored.

    Args:
        components: Array of Pauli components returned by find_heff
        threshold: Ignore terms with absolute components below this value.
        scale: Manually set the scale.
        ignore_identity: Ignore the identity term.

    Returns:
        A Figure object containing the bar graph.
    """
    components = components.copy()

    max_abs = np.amax(np.abs(components))

    if scale is None:
        scale = _find_scale(max_abs)

    scale_omega = scale.pulsatance_value

    if threshold is None:
        threshold = scale_omega * 1.e-2

    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold *= -max_abs

    # Dividing by omega -> now everything is in terms of frequency (not angular)
    components /= scale_omega
    threshold /= scale_omega

    if ignore_identity:
        components.reshape(-1)[0] = 0.

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
    ax.set_ylabel(r'$\nu/(\mathrm{' + scale.frequency_unit + '})$')

    return fig


def _find_scale(val):
    for scale in reversed(FrequencyScale):
        omega = scale.pulsatance_value
        if val > 0.1 * omega:
            return scale

    raise RuntimeError(f'Could not find a proper scale for value {val}')


def inspect_leastsq_minimization(
    filename: str,
    threshold: float = 0.01,
    tscale: FrequencyScale = FrequencyScale.MHz,
    align_ylim: bool = True,
    limit_components: Optional[List[Tuple]] = None
) -> List[mpl.figure.Figure]:
    r"""Inspect the results of least-squares effective Hamiltonian extraction.

    Args:
        filename: The HDF5 file name produced by `find_heff`.
        threshold: Ignore components of :math:`i \mathrm{log} U(t)` with absolute value (in radians) below this value.
        tscale: Time scale for the horizontal axes.
        align_ylim: Align the ranges of vertical axes.
        limit_components: Plot only the components included in this list.

    Returns:
        A list of figures (one per fit iteration).
    """
    with h5py.File(filename, 'r') as source:
        num_qudits = source['num_qudits'][()]
        num_sim_levels = source['num_sim_levels'][()]
        comp_dim = source['comp_dim'][()]
        tlist = source['tlist'][()]
        ilogvs = source['ilogvs'][()]
        max_com = source['max_com'][()]
        min_compo_ratio = source['min_compo_ratio'][()]
        num_update_per_iteration = source['num_update_per_iteration'][()]
        ilogu_compos = source['ilogu_compos'][()]
        last_valid_it = source['last_valid_it'][()]
        heff_compos = source['heff_compos'][()]
        compos = source['compos'][()]
        success = source['fit_success'][()]
        com = source['com'][()]

    num_paulis = num_sim_levels ** 2
    basis_size = num_paulis ** num_qudits - 1

    tlist *= tscale.frequency_value
    compos /= tscale.frequency_value
    heff_compos /= tscale.frequency_value

    num_loops = ilogu_compos.shape[0]

    max_heff_compo = np.amax(np.abs(heff_compos), axis=1, keepdims=True)
    max_heff_compo = np.repeat(max_heff_compo, basis_size, axis=1)
    compo_ratio = np.zeros_like(compos)
    np.divide(np.abs(compos), max_heff_compo, out=compo_ratio, where=(max_heff_compo > 0.))

    is_update_candidate = success & (com < max_com)
    compo_nonnegligible = (compo_ratio > min_compo_ratio)
    is_update_candidate[1:] &= compo_nonnegligible[1:] | (last_valid_it[1:, None] != tlist.shape[0])
    num_candidates = np.count_nonzero(is_update_candidate[iloop], axis=1)
    num_candidates = np.minimum(num_update_per_iteration, num_candidates)
    masked_compos = np.where(is_update_candidate, np.abs(compos), 0.)
    sorted_compos = -np.sort(-masked_compos)
    compo_mins = np.where(num_candidates < num_basis,
                          sorted_compos[np.arange(num_loops), num_candidates],
                          0.)
    selected = (masked_compos > compo_mins[:, None])

    log_com = np.zeros_like(com)
    np.log10(com, out=log_com, where=(com > 0.))
    log_cr = np.ones_like(compo_ratio) * np.amin(compo_ratio, where=(compo_ratio > 0.), initial=np.amax(compo_ratio))
    np.log10(compo_ratio, out=log_cr, where=(compo_ratio > 0.))

    if limit_components is not None:
        tuple_of_indices = tuple(np.array(index[i] for index in limit_components) for i in range(len(limit_components[0])))
        limit_components = np.ravel_multi_index(tuple_of_indices, (num_paulis,) * num_qudits)
        limit_components = np.sort(limit_components)

    figures = []

    for iloop in range(num_loops):
        if limit_components is None:
            # Make a list of tuples from a tuple of arrays
            indices_loop = np.nonzero(np.amax(np.abs(ilogu_compos[iloop]), axis=0) > threshold)[0] + 1
            indices_loop = np.concatenate([np.array([0]), indices_loop])
        else:
            indices_loop = limit_components

        fig, axes = _make_figure(len(indices_iter) + 2)
        figures.append(fig)
        fig.suptitle(f'Iteration {iloop} (last_valid_it {last_valid_it[iloop]})', fontsize=24)

        ## First row: global digest plots
        # log(cr) versus log(com)
        ax = axes[0]

        ax.set_xlabel(r'$log_{10}(com)$')
        if iloop == 0:
            ax.hist(log_com[iloop])
            ax.axvline(np.log10(max_com), color='red')
        else:
            ax.set_ylabel(r'$log_{10}(cr)$')
            ax.scatter(log_com[iloop], log_cr[iloop])
            ax.axvline(np.log10(max_com), color='red')
            ax.axhline(np.log10(min_compo_ratio), color='red')

        # ilogvs
        ax = axes[1]
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.plot(tlist, np.sort(ilogvs[iloop], axis=1))

        ## individual pauli components
        _plot_ilogu_compos(axes[2:], compos, num_qudits, num_paulis, indices_loop, tlist, tscale, comp_dim)

        for iax, basis_index_flat in enumerate(indices_loop):
            ax = axes[iax]

            ymin, ymax = ax.ylim()

            ibase = basis_index_flat - 1

            # Show cumul-over-max and compo ratio values
            ax.text(tlist[10], ymin + (ymax - ymin) * 0.1, f'com {com[iloop, ibase]:.3f} cr {compo_ratio[iloop, ibase]:.3f}')

            # Draw the fit line for successful fits
            if success[iloop, ibase]:
                ax.plot(tlist, compos[ibase] * tlist)

            if selected[iloop, ibase]:
                ax.tick_params(color='magenta', labelcolor='magenta')
                for spine in ax.spines.values():
                    spine.set(edgecolor='magenta')

    return figures


def inspect_fidelity_maximization(
    filename: str,
    threshold: float = 0.01,
    tscale: FrequencyScale = FrequencyScale.MHz,
    align_ylim: bool = False,
    limit_components: Optional[List[Tuple]] = None,
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
        if num_sim_levels != comp_dim:
            heff_compos = source['heff_compos_original'][()]
        else:
            heff_compos = source['heff_compos'][()]
        final_fidelity = source['final_fidelity'][()]
        try:
            loss = source['loss'][()]
            grad = source['grad'][()]
        except KeyError:
            loss = None
            grad = None

    num_paulis = num_sim_levels ** 2

    tlist *= tscale.frequency_value
    heff_compos /= tscale.frequency_value

    ilogus = -matrix_angle(time_evolution)
    ilogu_compos = paulis.components(ilogus, (num_sim_levels,) * num_qudits).real
    ilogu_compos = ilogu_compos.reshape(tlist.shape[0], -1)[:, 1:]

    heff = paulis.compose(heff_compos, (num_sim_levels,) * num_qudits)
    heff_t = make_heff_t(heff, tlist)
    ueff_dagger = matrix_exp(1.j * heff_t, hermitian=-1)

    target = np.matmul(time_evolution, ueff_dagger)
    ilogtargets, ilogvs = matrix_angle(target, with_diagonals=True)
    ilogtargets *= -1.
    ilogtarget_compos = paulis.components(ilogtargets, (num_sim_levels,) * num_qudits).real
    ilogtarget_compos = ilogtarget_compos.reshape(tlist.shape[0], -1)[:, 1:]

    if limit_components is None:
        # Make a list of tuples from a tuple of arrays
        indices_ilogu = np.nonzero(np.amax(np.abs(ilogu_compos), axis=0) > threshold)[0] + 1
        indices_ilogu = np.concatenate([np.array([0]), indices_ilogu])
        indices_ilogtarget = np.nonzero(np.amax(np.abs(ilogtarget_compos), axis=0) > threshold)[0] + 1
        indices_ilogtarget = np.concatenate([np.array([0]), indices_ilogtarget])
    else:
        tuple_of_indices = tuple([index[i] for index in limit_components] for i in range(len(limit_components[0])))
        limit_components = np.ravel_multi_index(tuple_of_indices, (num_paulis,) * num_qudits)
        limit_components = np.sort(limit_components)

        indices_ilogu = limit_components
        indices_ilogtarget = limit_components

    figures = []

    ## First figure: original time evolution and Heff
    fig, axes = _make_figure(len(indices_ilogu))
    figures.append(fig)
    fig.suptitle(r'Original time evolution vs $H_{eff}$', fontsize=24)

    _plot_ilogu_compos(axes, ilogu_compos, num_qudits, num_paulis, indices_ilogu, tlist, tscale,
                       comp_dim, align_ylim=align_ylim)

    compo_line = axes[0].get_lines()[0]

    ## Add lines with slope=compo for terms above threshold
    for iax, basis_index_flat in enumerate(indices_ilogu):
        ax = axes[iax]
        basis_index = np.unravel_index(basis_index_flat, (num_paulis,) * num_qudits)
        fit_line, = ax.plot(tlist, heff_compos[basis_index] * tlist)

    fig.legend((compo_line, fit_line),
               ('ilogU(t) component', '$H_{eff}t$ component'), 'upper right')

    ## Second figure: subtracted unitaries
    fig, axes = _make_figure(len(indices_ilogtarget))
    figures.append(fig)
    fig.suptitle('Unitary-subtracted time evolution', fontsize=24)

    ## Plot individual pauli components
    _plot_ilogu_compos(axes, ilogtarget_compos, num_qudits, num_paulis, indices_ilogtarget, tlist, tscale,
                       comp_dim, align_ylim=align_ylim)

    if digest:
        ## Third figure: fit digest plots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        figures.append(fig)
        fig.suptitle('Fit metrics', fontsize=24)

        # fidelity
        ax = axes[0]
        ax.set_title('Final fidelity')
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel('fidelity')
        ax.plot(tlist, final_fidelity)
        ax.axhline(1., color='black', linewidth=0.5)

        # eigenphases of target matrices
        ax = axes[1]
        ax.set_title('Final target matrix eigenphases')
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel(r'$U(t)U_{eff}^{\dagger}(t)$ eigenphases')
        ax.plot(tlist, ilogvs)

        ## Intermediate data is not available if minuit is used
        if loss is not None:
            ax = axes[2]
            ax.set_title('Loss evolution')
            ax.set_xlabel('steps')
            ax.set_ylabel('loss')
            ax.plot(loss)
            ax.axhline(0., color='black', linewidth=0.5)

            ax = axes[3]
            ax.set_title('Gradient evolution')
            ax.set_xlabel('steps')
            ax.set_ylabel('max(abs(grad))')
            ax.plot(np.amax(np.abs(grad), axis=1))
            ax.axhline(0., color='black', linewidth=0.5)

    for fig in figures:
        fig.tight_layout()

    return figures


def _make_figure(num_axes, nxmin=4, nxmax=12):
    nx = np.floor(np.sqrt(num_axes)).astype(int)
    nx = max(nx, nxmin)
    nx = min(nx, nxmax)
    ny = np.ceil(num_axes / nx).astype(int)

    fig, axes = plt.subplots(ny, nx, figsize=(nx * 4, ny * 4))
    return fig, axes.reshape(-1)


def _plot_ilogu_compos(axes, compos_data, num_qudits, num_paulis, limit_components_flat, tlist, tscale, comp_dim, align_ylim=False):
    # compos_data: shape (T, num_basis - 1)
    # limit_components_flat
    num_sim_levels = np.around(np.sqrt(num_paulis)).astype(int)
    num_comp_paulis = comp_dim ** 2

    limit_components = np.unravel_index(limit_components_flat, (num_paulis,) * num_qudits)
    labels = paulis.labels((num_sim_levels,) * num_qudits, norm=False)[limit_components]
    compos_data_selected = compos_data[:, limit_components_flat - 1]

    iax = 0

    if limit_components_flat[0] == 0:
        l0p = paulis.l0_projector(comp_dim, num_sim_levels)
        l0_projection = l0p
        for _ in range(num_qudits - 1):
            l0_projection = np.kron(l0_projection, l0p)

        l0_compos = np.matmul(compos_data, l0_projection[1:])

        # lambda 0 projection onto computational subspace
        ax = axes[0]
        title = f'${labels[0]}$'
        if comp_dim != num_sim_levels:
            title += ' (projected)'
        ax.set_title(title)
        ax.plot(tlist, l0_compos)

        iax += 1

    while iax < limit_components_flat.shape[0]:
        ax = axes[iax]

        ax.set_title(f'${labels[iax]}$')
        ax.plot(tlist, compos_data_selected[:, iax])

        if (np.array([indices[iax] for indices in limit_components]) < num_comp_paulis).all():
            for spine in ax.spines.values():
                spine.set(linewidth=2.)

        iax += 1

    ymax = np.amax(compos_data_selected)
    ymin = np.amin(compos_data_selected)
    vrange = ymax - ymin
    ymax += 0.2 * vrange
    ymin -= 0.2 * vrange

    for ax in axes.reshape(-1):
        ax.axhline(0., color='black', linewidth=0.5)
        if align_ylim:
            ax.set_ylim(ymin, ymax)
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.set_ylabel('rad')


def plot_amplitude_scan(
    amplitudes: np.ndarray,
    components: Sequence,
    threshold: Optional[float] = None,
    amp_scale: Optional[FrequencyScale] = None,
    compo_scale: Optional[FrequencyScale] = None,
    max_poly_order: int = 4
) -> Tuple[mpl.figure.Figure, np.ndarray, FrequencyScale, FrequencyScale]:
    """Plot the result of the amplitude scan.

    See the last example in examples/validation/heff.ipynb for how to prepare the inputs to this function.

    Args:
        amplitudes: Array of drive amplitudes (assumed real)
        components: Returned list of effective Hamiltonian components from find_heff.
        threshold: Only plot components whose maximum is above this threshold. Defaults to
            `0.01 * compo_scale.pulsatance_value`.
        amp_scale: Scale of the drive amplitude. If set to None, the value is determined from the values in `amplitudes`.
        compo_scale: Scale of the components. If set to None, the value is determined from the global maximum of `abs(components)`.
        max_poly_order: Maximum polynomial order for fitting the amplitude dependencies of the components.

    Returns:
        A Figure, fit results (polynomial coefficients) as an array, amplitude scale, and components scale.
    """
    components = np.asarray(components)

    num_qudits = len(components.shape) - 1 # first dimension: amplitudes
    comp_dim = int(np.around(np.sqrt(components.shape[1])))
    num_paulis = comp_dim ** 2

    if amp_scale is None:
        amp_scale = _find_scale(np.amax(np.abs(amplitudes)))

    amp_scale_omega = amp_scale.pulsatance_value
    amps_norm = amplitudes / amp_scale_omega

    if compo_scale is None:
        compo_scale = _find_scale(np.amax(np.abs(components)))

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
        ax.set_xlabel(f'Drive amplitude (${amp_scale.frequency_unit}$)')
        ax.set_ylabel(fr'$\nu/{compo_scale.frequency_unit}$')
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


def print_amplitude_scan(
    coefficients: np.ndarray,
    amp_scale: FrequencyScale,
    compo_scale: FrequencyScale
) -> LaTeXRepr:
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

        line = fr'\frac{{\nu_{{{basis_labels[index]}}}}}{{\mathrm{{{compo_scale.frequency_unit}}}}} &='

        for order, p in enumerate(coefficients[index]):
            if p == 0.:
                continue

            pstr = f'{abs(p):.2e}'
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

            if p < 0.:
                pexp = f'-{pexp}'

            if order == 0:
                line += pexp
            elif order == 1:
                line += f'{pexp} A'
            else:
                if p > 0.:
                    pexp = f'+ {pexp}'
                line += f'{pexp} A^{order}'

        lines.append(line)

    linebreak = r' \\ '
    expr = LaTeXRepr(fr'\begin{{align}}{linebreak.join(lines)}\end{{align}} A: amplitude in $\mathrm{{{amp_scale.frequency_unit}}}$')

    return expr
