from typing import Optional
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..paulis import (get_num_paulis, make_generalized_paulis, make_prod_basis,
                      prod_basis_labels, unravel_basis_index, get_l0_projection)
from ..utils import matrix_ufunc
from .common import make_ueff

def heff_expr(
    coefficients: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = None
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
        
    Returns:
        A LaTeX expression string for the effective Hamiltonian.
    """
    num_qubits = len(coefficients.shape)
    labels = prod_basis_labels(coefficients.shape[0], num_qubits, symbol=symbol)
        
    maxval = np.amax(np.abs(coefficients))
    for base, unit in [(1.e+9, 'GHz'), (1.e+6, 'MHz'), (1.e+3, 'kHz'), (1., 'Hz')]:
        norm = 2. * np.pi * base
        if maxval > norm:
            if threshold is None:
                threshold = norm * 1.e-2
            break
            
    if threshold is None:
        raise RuntimeError(f'Passed coefficients with maxabs = {maxval}')
            
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
            expr += f'{abs(coeff) / norm:.3f}{labels[index]}'
        else:
            if len(coefficients.shape) == 2:
                denom = '2'
            else:
                denom = '2^{%d}' % (len(coefficients.shape) - 1)
                
            expr += f'{abs(coeff) / norm:.3f}' + (r'\frac{%s}{%s}' % (labels[index], denom))
        
    return (r'\frac{H_{\mathrm{eff}}}{2 \pi \mathrm{%s}} = ' % unit) + expr


def coeffs_graph(
    coefficients: np.ndarray,
    symbol: Optional[str] = None,
    threshold: Optional[float] = None,
    ignore_identity: bool = True
) -> mpl.figure.Figure:
    
    num_qubits = len(coefficients.shape)
    labels = prod_basis_labels(coefficients.shape[0], num_qubits, symbol=symbol)
        
    maxval = np.amax(np.abs(coefficients))
    for base, unit in [(1.e+9, 'GHz'), (1.e+6, 'MHz'), (1.e+3, 'kHz'), (1., 'Hz')]:
        norm = 2. * np.pi * base
        if maxval > norm:
            if threshold is None:
                threshold = norm * 1.e-2
            break
            
    if threshold is None:
        raise RuntimeError(f'Passed coefficients with maxabs = {maxval}')
    
    if ignore_identity:
        coefficients = coefficients.copy()
        coefficients.reshape(-1)[0] = 0.
        
    # Negative threshold specified -> relative to max
    if threshold < 0.:
        threshold = np.amax(np.abs(coefficients)) * (-threshold)
        
    flat_indices = np.argsort(-np.abs(coefficients.reshape(-1)))
    nterms = np.count_nonzero(np.abs(coefficients) > threshold)
    indices = np.unravel_index(flat_indices[:nterms], coefficients.shape)
    
    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), coefficients[indices] / norm)
    
    xticks = np.char.add(np.char.add('$', labels), '$')
        
    ax.set_xticks(np.arange(nterms), labels=xticks[indices])
    ax.set_ylabel(r'$\nu/(2\pi{' + unit + '})$')
    
    return fig


def inspect_iterative_fit(
    filename: str,
    threshold: float = 0.01
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
    
    tscale = 1.e+6
    tlist *= tscale
    coeffs /= tscale
    heff_coeffs /= tscale
    
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

    figures = []
    
    for iloop in range(num_loops):
        fig, axes = _make_figure(ilogu_coeffs[iloop], threshold)
        figures.append(fig)
        fig.suptitle(f'Iteration {iloop} (last_valid_it {last_valid_it[iloop]})', fontsize=16)
        
        ## First row: global digest plots
        # log(cr) versus log(com)
        ax = axes[0, 0]
        
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
        ax = axes[0, 1]
        ax.set_xlabel('t (ns)')
        ax.plot(tlist, np.sort(ilogvs[iloop], axis=1))

        ## Second row and on: individual pauli coefficients
        ibases = _plot_ilogu_coeffs(axes, ilogu_coeffs[iloop], threshold, tlist, num_sim_levels, comp_dim, num_qubits)
        
        iax = np.ravel_multi_index((1, 1), axes.shape)
        for ibase in ibases:
            ax = axes.reshape(-1)[iax]
            iax += 1

            ymin, ymax = ax.ylim()
            
            # Show cumul-over-max and coeff ratio values
            ax.text(tlist[10], ymin + (ymax - ymin) * 0.1, f'com {com[iloop, ibase]:.3f} cr {coeff_ratio[iloop, ibase]:.3f}')
        
            # Draw the fit line for successful fits
            if success[iloop, ibase]:
                ax.plot(tlist, coeffs[iloop, ibase] * tlist)
                
            if selected[iloop, ibase]:
                ax.tick_params(color='magenta', labelcolor='magenta')
                for spine in ax.spines.values():
                    spine.set(edgecolor='magenta')
                    
    return figures


def inspect_maximize_fidelity(
    filename: str,
    threshold: float = 0.01
):
    
    with h5py.File(filename, 'r') as source:
        num_qubits = source['num_qubits'][()]
        num_sim_levels = source['num_sim_levels'][()]
        comp_dim = source['comp_dim'][()]
        time_evolution = source['time_evolution'][()]
        tlist = source['tlist'][()]
        heff_coeffs = source['heff_coeffs'][()]
        final_fidelity = source['final_fidelity'][()]
        try:
            loss = source['loss'][()]
            grad = source['grad'][()]
        except KeyError:
            loss = None
            grad = None
    
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    # Flattened list of basis operators excluding the identity operator
    basis_list = basis.reshape(-1, *basis.shape[-2:])[1:]
    basis_size = basis_list.shape[0]
   
    tscale = 1.e+6
    tlist *= tscale
    heff_coeffs /= tscale
        
    ilogus = matrix_ufunc(lambda u: -np.angle(u), time_evolution)
    ilogu_coeffs = np.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.
    
    ueff_dagger = make_ueff(heff_coeffs.reshape(-1)[1:], basis_list, tlist, num_qubits, phase_factor=1.)
    u_ueffdag = np.matmul(time_evolution, ueff_dagger)
    ilogu_ueffdags, ilogvs = matrix_ufunc(lambda u: -np.angle(u), u_ueffdag, with_diagonals=True)
    ilogu_ueffdag_coeffs = np.tensordot(ilogu_ueffdags, basis_list, ((1, 2), (2, 1))).real / 2.

    figures = []
    
    ## First figure: original time evolution and Heff
    fig, axes = _make_figure(ilogu_coeffs, threshold)
    figures.append(fig)
    fig.suptitle(r'Original time evolution vs $H_{eff}$')
    
    ## First row: global digest plots
    # fidelity
    ax = axes[0, 0]
    ax.set_xlabel(r'$t ({\mu}s)$')
    ax.set_ylabel('fidelity')
    ax.plot(tlist, final_fidelity)

    if loss is not None:
        ax = axes[0, 1]
        ax.set_xlabel('steps')
        ax.set_ylabel('loss')
        ax.plot(loss)

        ax = axes[0, 2]
        ax.set_xlabel('steps')
        ax.set_ylabel('max(abs(grad))')
        ax.plot(np.amax(np.abs(grad), axis=1))
        
    ## Second row and on: individual pauli coefficients
    ibases = _plot_ilogu_coeffs(axes, ilogu_coeffs, threshold, tlist, num_sim_levels, comp_dim, num_qubits)
    
    iax = np.ravel_multi_index((1, 1), axes.shape)
    for ibase in ibases:
        ax = axes.reshape(-1)[iax]
        iax += 1
        basis_index = unravel_basis_index(ibase + 1, num_sim_levels, num_qubits)
        ax.plot(tlist, heff_coeffs[basis_index] * tlist)
    
    ## Second figure: subtracted unitaries
    fig, axes = _make_figure(ilogu_ueffdag_coeffs, threshold)    
    figures.append(fig)
    fig.suptitle('Unitary-subtracted time evolution')

    ## First row: global digest plots
    ax = axes[0, 0]
    ax.set_xlabel(r'$t ({\mu}s)$')
    ax.set_ylabel(r'$ilog(U(t)U_{eff}^{\dagger}(t))$ eigenphases')
    ax.plot(tlist, ilogvs)

    ## Second row and on: individual pauli coefficients
    _plot_ilogu_coeffs(axes, ilogu_ueffdag_coeffs, threshold, tlist, num_sim_levels, comp_dim, num_qubits)

    return figures


def _make_figure(coeffs_data, threshold):
    num_axes = np.count_nonzero(np.amax(np.abs(coeffs_data), axis=0) > threshold)
    
    nx = np.floor(np.sqrt(num_axes + 1)).astype(int)
    nx = min(nx, 12)
    ny = np.ceil((num_axes + 1) / nx).astype(int) + 1

    return plt.subplots(ny, nx, figsize=(nx * 3, ny * 3))


def _plot_ilogu_coeffs(axes, coeffs_data, threshold, tlist, num_sim_levels, comp_dim, num_qubits):
    num_paulis = get_num_paulis(num_sim_levels)
    num_comp_paulis = get_num_paulis(comp_dim)
    basis_size = num_paulis ** num_qubits - 1

    labels = prod_basis_labels(num_paulis, num_qubits)
    
    ymax = np.amax(coeffs_data)
    ymin = np.amin(coeffs_data)
    vrange = ymax - ymin
    ymax += 0.2 * vrange
    ymin -= 0.2 * vrange

    if comp_dim != num_sim_levels:
        l0p = get_l0_projection(comp_dim, num_sim_levels)
        l0_projection = l0p
        for _ in range(num_qubits - 1):
            l0_projection = np.kron(l0_projection, l0p)

        l0_coeffs = np.matmul(coeffs_data, l0_projection[1:])

        # lambda 0 projection onto computational subspace
        ax = axes[1, 0]
        ax.set_title(f'${labels.reshape(-1)[0]}$ (projected)')
        ax.set_xlabel(r'$t ({\mu}s)$')
        ax.plot(tlist, l0_coeffs)
        ax.set_ylim(ymin, ymax)
        
    ibases = []

    iax = np.ravel_multi_index((1, 1), axes.shape)
    for ibase in range(basis_size):
        if np.amax(np.abs(coeffs_data[:, ibase])) < threshold:
            continue
            
        ibases.append(ibase)

        basis_index = unravel_basis_index(ibase + 1, num_sim_levels, num_qubits)

        ax = axes.reshape(-1)[iax]
        iax += 1
        ax.set_title(f'${labels[basis_index]}$')
        ax.set_xlabel(r'$t ({\mu}s)$')
        ax.plot(tlist, coeffs_data[:, ibase])
        ax.set_ylim(ymin, ymax)

        if (np.array(basis_index) < num_comp_paulis).all():
            for spine in ax.spines.values():
                spine.set(linewidth=2.)

    return ibases
