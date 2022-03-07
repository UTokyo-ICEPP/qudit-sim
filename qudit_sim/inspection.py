import numpy as np
import h5py
import matplotlib.pyplot as plt

from .paulis import get_num_paulis, pauli_labels, unravel_basis_index, get_l0_projection

def inspect_iterative_fit(
    filename: str,
    ignore_coeffs_below: float = 0.
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
    labels = pauli_labels(num_paulis)
    basis_labels = list(labels)
    for _ in range(1, num_qubits):
        basis_labels = list(b + p for b in basis_labels for p in labels)
    
    basis_size = num_paulis ** num_qubits - 1
    
    max_heff_coeff = np.amax(np.abs(heff_coeffs), axis=1, keepdims=True)
    max_heff_coeff = np.repeat(max_heff_coeff, heff_coeffs.shape[1], axis=1)
    coeff_ratio = np.zeros_like(coeffs)
    np.divide(np.abs(coeffs), max_heff_coeff, out=coeff_ratio, where=(max_heff_coeff > 0.))
    
    is_update_candidate = success & (com < max_com)
    coeff_nonnegligible = (coeff_ratio > min_coeff_ratio)
    is_update_candidate[1:] &= coeff_nonnegligible[1:] | (last_valid_it[1:, None] != tlist.shape[0])

    update_indices = np.argsort(np.where(is_update_candidate, -np.abs(coeffs), 0.))
    
    log_com = np.zeros_like(com)
    np.log10(com, out=log_com, where=(com > 0.))
    log_cr = np.ones_like(coeff_ratio) * np.amin(coeff_ratio, where=(coeff_ratio > 0.), initial=np.amax(coeff_ratio))
    np.log10(coeff_ratio, out=log_cr, where=(coeff_ratio > 0.))

    if comp_dim != num_sim_levels:
        # get the lambda 0 projection onto computational subspace
        num_comp_paulis = get_num_paulis(comp_dim)
        l0p = get_l0_projection(comp_dim, num_sim_levels)
        l0_projection = l0p
        for _ in range(num_qubits - 1):
            l0_projection = np.kron(l0_projection, l0p)

        l0_projection = l0_projection[1:]

        l0_coeffs = np.tensordot(ilogu_coeffs, l0_projection, (2, 0))
    
    figures = []
    
    for iloop in range(ilogu_coeffs.shape[0]):
        num_axes = np.count_nonzero(np.amax(np.abs(ilogu_coeffs[iloop]), axis=0) > ignore_coeffs_below)

        nx = np.floor(np.sqrt(num_axes + 1)).astype(int)
        nx = min(nx, 12)
        ny = np.ceil((num_axes + 1) / nx).astype(int) + 1
        
        fig, axes = plt.subplots(ny, nx, figsize=(nx * 3, ny * 3))
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
        
        num_candidates = min(num_update_per_iteration, np.count_nonzero(is_update_candidate[iloop]))

        selected = update_indices[iloop, :num_candidates]
        
        ymax = np.amax(ilogu_coeffs[iloop])
        ymin = np.amin(ilogu_coeffs[iloop])
        vrange = ymax - ymin
        ymax += 0.2 * vrange
        ymin -= 0.2 * vrange
        
        if comp_dim != num_sim_levels:
            # lambda 0 projection onto computational subspace
            ax = axes[1, 0]
            ax.set_title(f'${basis_labels[0]}$ (projected)')
            ax.plot(tlist, l0_coeffs[iloop])
            ax.set_ylim(ymin, ymax)
            
        iax = 1
        for ibase in range(basis_size):
            if np.amax(np.abs(ilogu_coeffs[iloop, :, ibase])) < ignore_coeffs_below:
                continue
            
            ax = axes[(iax // nx) + 1, iax % nx]
            iax += 1
            ax.set_title(f'${basis_labels[ibase + 1]}$')
            ax.plot(tlist, ilogu_coeffs[iloop, :, ibase])
            ax.text(tlist[10], ymin + (ymax - ymin) * 0.1, f'com {com[iloop, ibase]:.3f} cr {coeff_ratio[iloop, ibase]:.3f}')

            if success[iloop, ibase]:
                ax.plot(tlist, coeffs[iloop, ibase] * tlist)
                
            ax.set_ylim(ymin, ymax)

            if ibase in selected:
                ax.tick_params(color='magenta', labelcolor='magenta')
                for spine in ax.spines.values():
                    spine.set(edgecolor='magenta')
                    
            basis_index = unravel_basis_index(ibase + 1, num_sim_levels, num_qubits)
            if (np.array(basis_index) < num_comp_paulis).all():
                for spine in ax.spines.values():
                    spine.set(linewidth=2.)
                    
    return figures


def inspect_maximize_fidelity(
    filename: str,
    ignore_coeffs_below: float = 0.
):
    
    with h5py.File(filename, 'r') as source:
        num_qubits = source['num_qubits'][()]
        num_sim_levels = source['num_sim_levels'][()]
        comp_dim = source['comp_dim'][()]
        time_evolution = source['time_evolution'][()]
        tlist = source['tlist'][()]
        heff_coeffs = source['heff_coeffs'][()]
        final_fidelity = source['final_fidelity'][()]
    
    paulis = make_generalized_paulis(num_sim_levels)
    basis = make_prod_basis(paulis, num_qubits)
    # Flattened list of basis operators excluding the identity operator
    basis_list = basis.reshape(-1, *basis.shape[-2:])[1:]
    
    num_paulis = paulis.shape[0]
    labels = pauli_labels(num_paulis)
    basis_labels = list(labels)
    for _ in range(1, num_qubits):
        basis_labels = list(b + p for b in basis_labels for p in labels)
        
    basis_size = basis_list.shape[0]
        
    ilogus = matrix_ufunc(lambda u: -np.angle(u), time_evolution)
    ilogu_coeffs = np.tensordot(ilogus, basis_list, ((1, 2), (2, 1))).real / 2.
    
    ueff_dagger = make_ueff(heff_coeffs.reshape(-1)[1:], basis_list, num_qubits, tlist, phase_factor=1.)
    u_ueffdag = np.matmul(time_evolution, ueff_dagger)
    ilogu_ueffdags, ilogvs = matrix_ufunc(lambda u: -np.angle(u), u_ueffdag, with_diagonals=True)
    ilogu_ueffdag_coeffs = np.tensordot(ilogu_ueffdags, basis_list, ((1, 2), (2, 1))).real / 2.

    if comp_dim != num_sim_levels:
        # get the lambda 0 projection onto computational subspace
        num_comp_paulis = get_num_paulis(comp_dim)
        l0p = get_l0_projection(comp_dim, num_sim_levels)
        l0_projection = l0p
        for _ in range(num_qubits - 1):
            l0_projection = np.kron(l0_projection, l0p)

        l0_projection = l0_projection[1:]
        
    figures = []
    
    ## First figure: original time evolution and Heff
    ## Second figure: subtracted unitaries
    
    for ifig, coeffs_data in enumerate([ilogu_coeffs, ilogu_ueffdag_coeffs]):
        num_axes = np.count_nonzero(np.amax(np.abs(coeffs_data), axis=0) > ignore_coeffs_below)
    
        nx = np.floor(np.sqrt(num_axes + 1)).astype(int)
        nx = min(nx, 12)
        ny = np.ceil((num_axes + 1) / nx).astype(int) + 1
        
        fig, axes = plt.subplots(ny, nx, figsize=(nx * 3, ny * 3))
        figures.append(fig)
        
        if ifig == 0:
            fig.suptitle(r'Original time evolution vs $H_{eff}$')
        else:
            fig.suptitle('Unitary-subtracted time evolution')

        ## First row: global digest plots
        ax = axes[0, 0]
        ax.set_xlabel(r'$t ({\mu}s)$')
        
        if ifig == 0:
            # fidelity
            ax.set_ylabel('fidelity')
            ax.plot(tlist * 1.e+6, final_fidelity)
        else:
            ax.set_ylabel(r'$ilog(U(t)U_{eff}^{\dagger}(t))$ eigenphases')
            ax.plot(tlist * 1.e+6, ilogvs)

        ## Second row and on: individual pauli coefficients
        ymax = np.amax(coeffs_data)
        ymin = np.amin(coeffs_data)
        vrange = ymax - ymin
        ymax += 0.2 * vrange
        ymin -= 0.2 * vrange

        if comp_dim != num_sim_levels:
            l0_coeffs = np.tensordot(coeffs_data, l0_projection, (1, 0))

            # lambda 0 projection onto computational subspace
            ax = axes[1, 0]
            ax.set_title(f'${basis_labels[0]}$ (projected)')
            ax.set_xlabel(r'$t ({\mu}s)$')
            ax.plot(tlist * 1.e+6, l0_coeffs)
            ax.set_ylim(ymin, ymax)

        iax = 1
        for ibase in range(basis_size):
            if np.amax(np.abs(coeffs_data[:, ibase])) < ignore_coeffs_below:
                continue

            basis_index = unravel_basis_index(ibase + 1, num_sim_levels, num_qubits)

            ax = axes[(iax // nx) + 1, iax % nx]
            iax += 1
            ax.set_title(f'${basis_labels[ibase + 1]}$')
            ax.set_xlabel(r'$t ({\mu}s)$')
            ax.plot(tlist * 1.e+6, coeffs_data[:, ibase])
            if ifig == 0:
                ax.plot(tlist * 1.e+6, heff_coeffs[basis_index] * tlist)
            ax.set_ylim(ymin, ymax)

            if (np.array(basis_index) < num_comp_paulis).all():
                for spine in ax.spines.values():
                    spine.set(linewidth=2.)

    return figures
