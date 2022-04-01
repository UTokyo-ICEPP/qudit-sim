from typing import Optional, Tuple, List
import numpy as np
import h5py
import scipy.optimize as sciopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from ..paulis import (get_num_paulis, get_pauli_dim,
                              make_generalized_paulis, make_prod_basis,
                              pauli_labels, prod_basis_labels, extract_coefficients,
                              unravel_basis_index, get_l0_projection)
from ..utils import matrix_ufunc, FrequencyScale
from .common import make_ueff

twopi = 2. * np.pi

def heff_expr(
    coefficients: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = None,
    scale: Optional[FrequencyScale] = None
) -> str:
    """Generate a LaTeX expression of the effective Hamiltonian from the Pauli coefficients.
    
    The dymamic range of the numerical values of the coefficients is set by the maximum absolute
    value. For example, if the maximum absolute value is between 1.e+6 and 1.e+9, the coefficients
    are expressed in MHz, with the minimum of 0.001 MHz. Pauli terms whose coefficients have
    absolute values below the threshold are ommitted.
    
    Args:
        heff: Array of Pauli coefficients returned by find_heff
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        threshold: Ignore terms with absolute coefficients below this value.
        scale: Manually set the scale.
        
    Returns:
        A LaTeX expression string for the effective Hamiltonian.
    """
    coefficients = coefficients.copy()
    
    num_qubits = len(coefficients.shape)
    labels = prod_basis_labels(coefficients.shape[0], num_qubits, symbol=symbol)

    if scale is None:
        scale = _find_scale(np.amax(np.abs(coefficients)))
        
    scale_omega = scale.pulsatance_value
    
    if threshold is None:
        threshold = scale_omega * 1.e-2
        
    coefficients /= scale_omega
    threshold /= scale_omega
            
    expr = ''
    
    for index in np.ndindex(coefficients.shape):
        coeff = coefficients[index]
        if abs(coeff) < threshold:
            continue
            
        if coeff < 0.:
            expr += ' - '
        elif expr:
            expr += ' + '
            
        if len(coefficients.shape) == 1:
            expr += f'{abs(coeff):.3f}{labels[index]}'
        else:
            if len(coefficients.shape) == 2:
                denom = '2'
            else:
                denom = '2^{%d}' % (len(coefficients.shape) - 1)
                
            expr += f'{abs(coeff):.3f}' + (r'\frac{%s}{%s}' % (labels[index], denom))
        
    return r'\frac{H_{\mathrm{eff}}}{\mathrm{' + scale.frequency_unit + '}} = ' + expr


def coeffs_bar(
    coefficients: np.ndarray,
    threshold: Optional[float] = None,
    ignore_identity: bool = True
) -> mpl.figure.Figure:
    
    coefficients = coefficients.copy()
    
    num_qubits = len(coefficients.shape)
    num_paulis = coefficients.shape[0]
    
    labels = prod_basis_labels(num_paulis, num_qubits, symbol='',
                               delimiter=('' if num_paulis == 4 else ','))
    
    maxval = np.amax(np.abs(coefficients))
       
    scale = _find_scale(maxval)
    scale_omega = scale.pulsatance_value
    if threshold is None:
        threshold = scale_omega * 1.e-2
        
    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold *= -maxval

    # Dividing by omega -> now everything is in terms of frequency (not angular)
    coefficients /= scale_omega
    threshold /= scale_omega
            
    if ignore_identity:
        coefficients = coefficients.copy()
        coefficients.reshape(-1)[0] = 0.
        
    flat_indices = np.argsort(-np.abs(coefficients.reshape(-1)))
    nterms = np.count_nonzero(np.abs(coefficients) > threshold)
    indices = np.unravel_index(flat_indices[:nterms], coefficients.shape)
    
    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), coefficients[indices])
    
    ax.axhline(0., color='black', linewidth=0.5)
    
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
    basis_indices: Optional[List[Tuple]] = None
) -> list:
    
    with h5py.File(filename, 'r') as source:
        num_qubits = source['num_qubits'][()]
        num_sim_levels = source['num_sim_levels'][()]
        comp_dim = source['comp_dim'][()]
        tlist = source['tlist'][()]
        ilogvs = source['ilogvs'][()]
        max_com = source['max_com'][()]
        min_coeff_ratio = source['min_coeff_ratio'][()]
        num_update_per_iteration = source['num_update_per_iteration'][()]
        ilogu_coeffs = source['ilogu_coeffs'][()]
        last_valid_it = source['last_valid_it'][()]
        heff_coeffs = source['heff_coeffs'][()]
        coeffs = source['coeffs'][()]
        success = source['fit_success'][()]
        com = source['com'][()]
    
    num_paulis = get_num_paulis(num_sim_levels)
    basis_size = num_paulis ** num_qubits - 1
    
    tlist *= tscale.frequency_value
    coeffs /= tscale.frequency_value
    heff_coeffs /= tscale.frequency_value
    
    num_loops = ilogu_coeffs.shape[0]
    
    max_heff_coeff = np.amax(np.abs(heff_coeffs), axis=1, keepdims=True)
    max_heff_coeff = np.repeat(max_heff_coeff, basis_size, axis=1)
    coeff_ratio = np.zeros_like(coeffs)
    np.divide(np.abs(coeffs), max_heff_coeff, out=coeff_ratio, where=(max_heff_coeff > 0.))
    
    is_update_candidate = success & (com < max_com)
    coeff_nonnegligible = (coeff_ratio > min_coeff_ratio)
    is_update_candidate[1:] &= coeff_nonnegligible[1:] | (last_valid_it[1:, None] != tlist.shape[0])
    num_candidates = np.count_nonzero(is_update_candidate[iloop], axis=1)
    num_candidates = np.minimum(num_update_per_iteration, num_candidates)
    masked_coeffs = np.where(is_update_candidate, np.abs(coeffs), 0.)
    sorted_coeffs = -np.sort(-masked_coeffs)
    coeff_mins = np.where(num_candidates < num_basis,
                          sorted_coeffs[np.arange(num_loops), num_candidates],
                          0.)
    selected = (masked_coeffs > coeff_mins[:, None])

    log_com = np.zeros_like(com)
    np.log10(com, out=log_com, where=(com > 0.))
    log_cr = np.ones_like(coeff_ratio) * np.amin(coeff_ratio, where=(coeff_ratio > 0.), initial=np.amax(coeff_ratio))
    np.log10(coeff_ratio, out=log_cr, where=(coeff_ratio > 0.))
    
    if basis_indices is not None:
        tuple_of_indices = tuple(np.array(index[i] for index in basis_indices) for i in range(len(basis_indices[0])))
        basis_indices = np.ravel_multi_index(tuple_of_indices, (num_paulis,) * num_qubits)
        basis_indices = np.sort(basis_indices)
        
    figures = []
    
    for iloop in range(num_loops):
        if basis_indices is None:
            # Make a list of tuples from a tuple of arrays
            indices_loop = np.nonzero(np.amax(np.abs(ilogu_coeffs[iloop]), axis=0) > threshold)[0] + 1
            indices_loop = np.concatenate([np.array([0]), indices_loop])
        else:
            indices_loop = basis_indices
        
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
            ax.axhline(np.log10(min_coeff_ratio), color='red')

        # ilogvs
        ax = axes[1]
        ax.set_xlabel(f't ({tscale.time_unit})')
        ax.plot(tlist, np.sort(ilogvs[iloop], axis=1))
        
        ## individual pauli coefficients
        _plot_ilogu_coeffs(axes[2:], coeffs, num_qubits, num_paulis, indices_loop, tlist, tscale, comp_dim)
        
        for iax, basis_index_flat in enumerate(indices_loop):
            ax = axes[iax]

            ymin, ymax = ax.ylim()
            
            ibase = basis_index_flat - 1
            
            # Show cumul-over-max and coeff ratio values
            ax.text(tlist[10], ymin + (ymax - ymin) * 0.1, f'com {com[iloop, ibase]:.3f} cr {coeff_ratio[iloop, ibase]:.3f}')
        
            # Draw the fit line for successful fits
            if success[iloop, ibase]:
                ax.plot(tlist, coeffs[ibase] * tlist)
                
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
    basis_indices: Optional[List[Tuple]] = None,
    digest: bool = True
) -> List[mpl.figure.Figure]:
    """Plot the time evolution of Pauli coefficients before and after fidelity maximization.
    
    Args:
        filename: Name of the HDF5 file containing the fit result.
        threshold: Threshold for a Pauli component to be plotted, in radians. Ignored if
            basis_indices is not None.
        tscale: Scale for the time axis.
        align_ylim: Whether to align the y axis limits for all plots.
        basis_indices: List of Pauli components to plot.
        digest: If True, a figure for fit information is included.
        
    Returns:
        A list of two (three if digest=True) figures.
    """
    
    with h5py.File(filename, 'r') as source:
        num_qubits = int(source['num_qubits'][()])
        num_sim_levels = int(source['num_sim_levels'][()])
        comp_dim = int(source['comp_dim'][()])
        time_evolution = source['time_evolution'][()]
        tlist = source['tlist'][()]
        if num_sim_levels != comp_dim:
            heff_coeffs = source['heff_coeffs_original'][()]
        else:
            heff_coeffs = source['heff_coeffs'][()]
        final_fidelity = source['final_fidelity'][()]
        try:
            loss = source['loss'][()]
            grad = source['grad'][()]
        except KeyError:
            loss = None
            grad = None
            
    num_paulis = get_num_paulis(num_sim_levels)
    
    tlist *= tscale.frequency_value
    heff_coeffs /= tscale.frequency_value
        
    ilogus = matrix_ufunc(lambda u: -np.angle(u), time_evolution)
    ilogu_coeffs = extract_coefficients(ilogus, num_qubits)
    ilogu_coeffs = ilogu_coeffs.reshape(tlist.shape[0], -1)[:, 1:]
    
    ueff_dagger = make_ueff(heff_coeffs, num_sim_levels, tlist, num_qubits, phase_factor=1.)
    target = np.matmul(time_evolution, ueff_dagger)
    ilogtargets, ilogvs = matrix_ufunc(lambda u: -np.angle(u), target, with_diagonals=True)
    ilogtarget_coeffs = extract_coefficients(ilogtargets, num_qubits)
    ilogtarget_coeffs = ilogtarget_coeffs.reshape(tlist.shape[0], -1)[:, 1:]
    
    if basis_indices is None:
        # Make a list of tuples from a tuple of arrays
        indices_ilogu = np.nonzero(np.amax(np.abs(ilogu_coeffs), axis=0) > threshold)[0] + 1
        indices_ilogu = np.concatenate([np.array([0]), indices_ilogu])
        indices_ilogtarget = np.nonzero(np.amax(np.abs(ilogtarget_coeffs), axis=0) > threshold)[0] + 1
        indices_ilogtarget = np.concatenate([np.array([0]), indices_ilogtarget])
    else:
        tuple_of_indices = tuple([index[i] for index in basis_indices] for i in range(len(basis_indices[0])))
        basis_indices = np.ravel_multi_index(tuple_of_indices, (num_paulis,) * num_qubits)
        basis_indices = np.sort(basis_indices)
        
        indices_ilogu = basis_indices
        indices_ilogtarget = basis_indices

    figures = []
    
    ## First figure: original time evolution and Heff
    fig, axes = _make_figure(len(indices_ilogu))
    figures.append(fig)
    fig.suptitle(r'Original time evolution vs $H_{eff}$', fontsize=24)
        
    _plot_ilogu_coeffs(axes, ilogu_coeffs, num_qubits, num_paulis, indices_ilogu, tlist, tscale,
                       comp_dim, align_ylim=align_ylim)
    
    coeff_line = axes[0].get_lines()[0]
    
    ## Add lines with slope=coeff for terms above threshold
    for iax, basis_index_flat in enumerate(indices_ilogu):
        ax = axes[iax]
        basis_index = np.unravel_index(basis_index_flat, (num_paulis,) * num_qubits)
        fit_line, = ax.plot(tlist, heff_coeffs[basis_index] * tlist)
        
    fig.legend((coeff_line, fit_line),
               ('ilogU(t) coefficient', '$H_{eff}t$ coefficient'), 'upper right')

    ## Second figure: subtracted unitaries
    fig, axes = _make_figure(len(indices_ilogtarget))
    figures.append(fig)
    fig.suptitle('Unitary-subtracted time evolution', fontsize=24)

    ## Plot individual pauli coefficients
    _plot_ilogu_coeffs(axes, ilogtarget_coeffs, num_qubits, num_paulis, indices_ilogtarget, tlist, tscale,
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


def _plot_ilogu_coeffs(axes, coeffs_data, num_qubits, num_paulis, basis_indices_flat, tlist, tscale, comp_dim, align_ylim=False):
    # coeffs_data: shape (T, num_basis - 1)
    # basis_indices_flat
    num_sim_levels = get_pauli_dim(num_paulis)
    num_comp_paulis = get_num_paulis(comp_dim)

    basis_indices = np.unravel_index(basis_indices_flat, (num_paulis,) * num_qubits)
    labels = prod_basis_labels(num_paulis, num_qubits)[basis_indices]
    coeffs_data_selected = coeffs_data[:, basis_indices_flat - 1]

    iax = 0

    if basis_indices_flat[0] == 0:
        l0p = get_l0_projection(comp_dim, num_sim_levels)
        l0_projection = l0p
        for _ in range(num_qubits - 1):
            l0_projection = np.kron(l0_projection, l0p)

        l0_coeffs = np.matmul(coeffs_data, l0_projection[1:])

        # lambda 0 projection onto computational subspace
        ax = axes[0]
        ax.set_title(f'${labels[0]}$ (projected)')
        ax.plot(tlist, l0_coeffs)
        
        iax += 1
        
    while iax < basis_indices_flat.shape[0]:
        ax = axes[iax]

        ax.set_title(f'${labels[iax]}$')
        ax.plot(tlist, coeffs_data_selected[:, iax])
        
        if (np.array([indices[iax] for indices in basis_indices]) < num_comp_paulis).all():
            for spine in ax.spines.values():
                spine.set(linewidth=2.)
                
        iax += 1
    
    ymax = np.amax(coeffs_data_selected)
    ymin = np.amin(coeffs_data_selected)
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
    coefficients: np.ndarray,
    threshold: Optional[float] = None,
    amp_scale: Optional[FrequencyScale] = None,
    coeff_scale: Optional[FrequencyScale] = None,
    max_poly_order: int = 4
) -> Tuple[mpl.figure.Figure, List[str], FrequencyScale]:
    
    num_qubits = len(coefficients.shape) - 1 # first dimension: amplitudes
    comp_dim = get_pauli_dim(coefficients.shape[1])
    num_paulis = get_num_paulis(comp_dim)

    if amp_scale is None:
        amp_scale = _find_scale(np.amax(np.abs(amplitudes)))
        
    amp_scale_omega = amp_scale.pulsatance_value
    amps_norm = amplitudes / amp_scale_omega

    if coeff_scale is None:
        coeff_scale = _find_scale(np.amax(np.abs(coefficients)))
        
    coeff_scale_omega = coeff_scale.pulsatance_value
    coeffs_norm = coefficients / coeff_scale_omega
    
    if threshold is None:
        threshold = coeff_scale_omega * 1.e-2
        
    threshold /= coeff_scale_omega
    
    # Amplitude, coeff, and threshold are all divided by omega and are in terms of frequency
    
    amax = np.amax(np.abs(coeffs_norm), axis=0)
    num_above_threshold = np.count_nonzero(amax > threshold)

    if num_qubits > 1 and num_above_threshold > 6:
        # Which Pauli components of the first qubit have plots to draw?
        has_passing_coeffs = np.any(amax.reshape(num_paulis, -1) > threshold, axis=1)
        num_plots = np.sum(has_passing_coeffs)
        
        nv = np.ceil(np.sqrt(num_plots)).astype(int)
        nh = np.ceil(num_plots / nv).astype(int)
        fig, axes = plt.subplots(nv, nh, figsize=(16, 12))
        
        prefixes = pauli_labels(num_paulis, symbol='')
        
        expr_lines = []
        iax = 0
        ymin, ymax = 0., 0.

        for ip in range(num_paulis):
            if not has_passing_coeffs[ip]:
                continue

            ax = axes.reshape(-1)[iax]
            expr_lines += _plot_amplitude_scan_on(ax, amps_norm, coeffs_norm[:, ip],
                                                  threshold, max_poly_order, prefix=prefixes[ip])
            
            i, a = ax.get_ylim()
            ymin = min(i, ymin)
            ymax = max(a, ymax)
            
            iax += 1
            
        for ax in axes.reshape(-1)[:num_plots]:
            ax.set_ylim(ymin, ymax)
            
    else:
        num_plots = 1
        fig, axes = plt.subplots(1, 1, squeeze=False)
        expr_lines = _plot_amplitude_scan_on(axes[0, 0], amps_norm, coeffs_norm, threshold,
                                             max_poly_order)
        
    nu = r'\nu'
    nusub = lambda s: r'\nu_{' + s + '}'
    frac = lambda n, d: r'\frac{' + n + '}{' + d + '}'
    mathrm = lambda r: r'\mathrm{' + r + '}'
    
    for ax in axes.reshape(-1)[:num_plots]:
        ax.grid(True)
        ax.set_xlabel(f'Drive amplitude (${amp_scale.frequency_unit}$)')
        ax.set_ylabel(f'${nu}/{coeff_scale.frequency_unit}$')
        
    fig.tight_layout()
    
    exprs = []
    for basis_label, rhs in expr_lines:
        exprs.append(frac(nusub(basis_label), mathrm(coeff_scale.frequency_unit)) + ' = ' + rhs)
        
    return fig, exprs, amp_scale
        

def _plot_amplitude_scan_on(ax, amps_norm, coeffs_norm, threshold, max_poly_order, prefix=''):
    num_qubits = len(coeffs_norm.shape) - 1 # first dimension: amplitudes
    comp_dim = get_pauli_dim(coeffs_norm.shape[1])
    num_paulis = get_num_paulis(comp_dim)
    
    basis_labels = prod_basis_labels(num_paulis, num_qubits, symbol='',
                                     delimiter=('' if comp_dim == 2 else ','))
    
    amps_norm_fine = np.linspace(amps_norm[0], amps_norm[-1], 100)
    num_amps = amps_norm.shape[0]
    
    filled_markers = MarkerStyle.filled_markers
    num_markers = len(filled_markers)

    exprs = []
    
    ymax = 0.
    ymin = 0.

    imarker = 0
    
    amax = np.amax(np.abs(coeffs_norm), axis=0)
    min_max = np.amin(np.where(amax > threshold, amax, np.amax(amax)))
    
    for index in np.ndindex(coeffs_norm.shape[1:]):
        coeffs = coeffs_norm[(slice(None),) + index]
        
        if amax[index] < threshold:
            continue
            
        if comp_dim == 2:
            label = prefix + basis_labels[index]
        else:
            if prefix:
                label = f'{prefix},{basis_labels[index]}'
            else:
                label = basis_labels[index]
                
        plot_label = f'${label}$'
            
        if amax[index] > 50. * min_max:
            plot_scale = 0.1
            plot_label += r' ($\times 0.1$)'
        else:
            plot_scale = 1.
            
        ymax = max(ymax, np.amax(coeffs * plot_scale))
        ymin = min(ymin, np.amin(coeffs * plot_scale))
            
        even = np.sum(coeffs[:num_amps // 2] * coeffs[-num_amps // 2:]) > 0.
        
        if even:
            curve = _poly_even
            p0 = np.zeros(max_poly_order // 2 + 1)
        else:
            curve = _poly_odd
            p0 = np.zeros((max_poly_order + 1) // 2)
        
        popt, _ = sciopt.curve_fit(curve, amps_norm, coeffs, p0=p0)
        
        pathcol = ax.scatter(amps_norm, coeffs * plot_scale, marker=filled_markers[imarker % num_markers], label=label)
        ax.plot(amps_norm_fine, curve(amps_norm_fine, *popt) * plot_scale, color=pathcol.get_edgecolor())
        
        expr_rhs = ''
        
        for p, order in zip(popt, range(0 if even else 1, 2 * len(popt), 2)):
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
                pexp = r'\left(' + pstr[:epos] + r' \times 10^{' + f'{power}' + r'}\right)'
                
            if p < 0.:
                pexp = '-' + pexp
            
            if order == 0:
                expr_rhs += pexp
            elif order == 1:
                expr_rhs += pexp + 'A'
            else:
                if p > 0.:
                    pexp = '+' + pexp

                expr_rhs += pexp + f'A^{order}'
                    
        exprs.append((label, expr_rhs))
        
        imarker += 1

    ax.set_ylim(ymin * 1.5, ymax * 1.2)
    ax.legend()

    return exprs


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
