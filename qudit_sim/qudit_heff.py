from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable
import string
import sys
from functools import partial
import numpy as np
import scipy
import scipy.optimize as sciopt
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jscila
from iminuit import Minuit
import optax
import h5py
import qutip as qtp

from .paulis import make_generalized_paulis, make_prod_basis
from .pulse_sim import run_pulse_sim

#jax.config.update('jax_enable_x64', True)

def process_fidelity(time_evolution, basis, coeffs, tlist, numpy=np):
    num_qubits = len(coeffs.shape)
    axes = list(range(num_qubits))
    heff = numpy.tensordot(basis, coeffs, (axes, axes)) / (2 ** (num_qubits - 1))
    
    return _process_fidelity(time_evolution, heff, tlist, numpy)


def _process_fidelity(time_evolution, heff, tlist, numpy):
    if numpy is np:
        eigh = np.linalg.eig
    elif numpy is jnp:
        eigh = jax.scipy.linalg.eigh
        
    heff_t = tlist[:, None, None] * numpy.tile(heff[None, ...], (tlist.shape[0], 1, 1))
    
    def unitarize(mat):
        eigvals, eigvecs = eigh(mat)
        dd = numpy.tile(eigvals[:, None, :], (1, eigvecs.shape[1], 1))
        return numpy.matmul(eigvecs * numpy.exp(1.j * dd), eigvecs.conj().transpose(0, 2, 1))
    
    udag_t = unitarize(heff_t)
    
    tr_u_udag = numpy.trace(numpy.matmul(time_evolution, udag_t), axis1=1, axis2=2)
    fidelity = (numpy.square(tr_u_udag.real) + numpy.square(tr_u_udag.imag)) / (heff.shape[-1] ** 2)
    
    return fidelity


def maximize_process_fidelity(
    time_evolution: np.ndarray,
    paulis: np.ndarray,
    num_qubits: int,
    tlist: np.ndarray,
    save_result_to: Optional[str] = None,
    optimizer: Union[optax.GradientTransformation, str] = optax.adam(0.01),
    init: Union[str, np.ndarray] = 'log_unitary_fit',
    max_updates: int = 1000,
    convergence: float = 1.e-6,
    **kwargs
) -> np.ndarray:
    
    basis = jnp.array(make_prod_basis(paulis, num_qubits))
    
    # Compute the loss from the flattened coeffs array
    basis_no_id = basis.reshape(-1, basis.shape[-2], basis.shape[-1])[1:]
    
    if isinstance(init, str) and init == 'log_unitary_fit':
        initial = fit_to_log_unitary(
            time_evolution,
            paulis,
            num_qubits,
            tlist,
            save_result_to=save_result_to,
            **kwargs)[1:] * tlist[-1]
    else:
        initial = jnp.array(init * tlist[-1])

    time_evolution = jnp.asarray(time_evolution)
    tlist_norm = jnp.array(tlist / tlist[-1])
    
    def loss_fn(coeffs):
        heff = jnp.tensordot(basis_no_id, coeffs, (0, 0)) / (2 ** (num_qubits - 1))
        fidelity = _process_fidelity(time_evolution[1:], heff, tlist_norm[1:], jnp)
        return 1. - 1. / tlist_norm.shape[0] - jnp.mean(fidelity)
    
    if optimizer == 'minuit':
        minimizer = Minuit(loss_fn, initial, grad=jax.grad(loss_fn))
        minimizer.strategy = 0
        minimizer.migrad()

        coeffs = np.concatenate(([0.], minimizer.values / tlist[-1])).reshape(basis.shape[:-2])
        
    else:
        loss_and_grad = jax.jit(jax.value_and_grad(lambda params: loss_fn(params['c'])))

        params = {'c': initial}
        opt_state = optimizer.init(params)

        loss_values = np.empty(max_updates, dtype='f8')

        @jax.jit
        def step(params, opt_state):
            loss, gradient = loss_and_grad(params)
            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, gradient

        for iup in range(max_updates):
            params, opt_state, loss, gradient = step(params, opt_state)

            loss_values[iup] = loss

            if jnp.amax(jnp.abs(gradient['c'])) < convergence:
                break

        if save_result_to:
            with h5py.File(f'{save_result_to}.h5', 'a') as out:
                out.create_dataset('loss', data=loss_values)
                out.create_dataset('num_updates', data=np.array(iup))

        coeffs = np.concatenate(([0.], params['c'] / tlist[-1])).reshape(basis.shape[:-2])
        
    return coeffs


def fit_to_log_unitary(
    time_evolution: np.ndarray,
    paulis: np.ndarray,
    num_qubits: int,
    tlist: np.ndarray,
    save_result_to: Optional[str] = None,
    fit_tol: float = 0.01,
    min_xpoints: int = 10,
    warn_fit_failure: bool = True
) -> np.ndarray:
    """
            fit_tol: Tolerance factor for the linear fit. The function tries to iteratively find a time interval
            where the best fit line `f(t)` satisfies `abs(sum(U(t) - f(t)) / sum(U(t))) < fit_tol`.
    """
    
    ## Take the log of the time evolution operator

    eigvals, eigcols = np.linalg.eig(time_evolution)
    eigrows = np.conjugate(np.transpose(eigcols, axes=(0, 2, 1)))
    
    omega_t = -np.angle(eigvals) # list of energy eigenvalues (mod 2pi) times t

    # Find the first t where an eigenvalue does a 2pi jump
    omega_min = np.amin(omega_t, axis=1)
    omega_max = np.amax(omega_t, axis=1)

    margin = 0.1

    min_hits_minus_pi = np.asarray(omega_min < -np.pi + margin).nonzero()[0]
    if len(min_hits_minus_pi) == 0:
        tmax_min = omega_t.shape[0]
    else:
        tmax_min = min_hits_minus_pi[0]
    
    max_hits_pi = np.asarray(omega_max > np.pi - margin).nonzero()[0]
    if len(max_hits_pi) == 0:
        tmax_max = omega_t.shape[0]
    else:
        tmax_max = max_hits_pi[0]
        
    tmax = min(tmax_min, tmax_max)
    
    heff_t = (eigcols * np.tile(omega_t[:, np.newaxis], (1, omega_t.shape[1], 1))) @ eigrows
    
    ## Extract the (generalized) Pauli components
    
    prod_basis = make_prod_basis(paulis, num_qubits)

    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    pauli_coeffs_t = np.einsum(f'txy,{qubit_indices}yx->t{qubit_indices}', heff_t, prod_basis).real
    # Divide the trace by two to account for the normalization of the generalized Paulis
    pauli_coeffs_t /= 2.

    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'a') as out:
            out.create_dataset('omega', data=omega_t)
            out.create_dataset('eigcols', data=eigcols)
            out.create_dataset('pauli_coeffs', data=pauli_coeffs_t)
            out.create_dataset('tmax', data=np.array([tmax]))
            out.create_dataset('fit_success', shape=pauli_coeffs_t.shape[1:], dtype='i')
            out.create_dataset('fit_range', shape=(pauli_coeffs_t.shape[1:] + (2,)), dtype='i')
            out.create_dataset('fit_residual', shape=pauli_coeffs_t.shape[1:], dtype='f8')
    
    ## Do a linear fit to each component
    
    num_paulis = paulis.shape[0]
    
    pauli_coeffs = np.zeros(pauli_coeffs_t.shape[1:])
    
    # This is probably not the most numpythonic way of indexing the array..
    time_series_list = pauli_coeffs_t.reshape(pauli_coeffs_t.shape[0], np.prod(pauli_coeffs.shape)).T

    line = lambda a, x: a * x
    for ic, coeffs_t in enumerate(time_series_list):
        icm = np.unravel_index(ic, pauli_coeffs.shape)
        
        # Iteratively determine the interval that yields a fit within tolerance
        start = 0
        end = tmax
        min_residual = None
        while True:
            xdata = tlist[start:end]
            ydata = coeffs_t[start:end]
            
            if xdata.shape[0] <= min_xpoints:
                if warn_fit_failure:
                    sys.stderr.write(f'Linear fit for coefficient of {icm} did not yield a'
                                     f' reliable result (minimum residual = {min_residual}).\n')
                
                if save_result_to:
                    with h5py.File(f'{save_result_to}.h5', 'a') as out:
                        out['fit_success'][icm] = 0
                        out['fit_range'][icm] = [start, end]
                        out['fit_residual'][icm] = residual
                        
                elif warn_fit_failure:
                    sys.stderr.write(' Run the function with the save_result_to option and'
                                     ' check the raw output.\n')
                popt = np.array([0.])
                break
            """
            # ydata is usually oscillatory - make the envelope array and determine p0 from the average
            local_max = []
            local_min = []
            grad = 0
            for t in range(1, ydata.shape[0]):
                if ydata[t] - ydata[t - 1] > 0.:
                    if grad < 0:
                        local_min.append((t - 1, ydata[t - 1]))
                        
                    grad = 1
                    
                elif ydata[t] - ydata[t - 1] < 0.:
                    if grad > 0:
                        local_max.append((t - 1, ydata[t - 1]))

                    grad = -1
                    
            env_high = np.empty_like(ydata)
            tprev = 0
            yprev = 0.
            for t, y in local_max:
                env_high[tprev:t] = np.linspace(yprev, y, t - tprev, endpoint=False)
                tprev = t
                yprev = y
            
            env_low = np.empty_like(ydata)
            tprev = 0
            yprev = 0.
            for t, y in local_min:
                env_low[tprev:t] = np.linspace(yprev, y, t - tprev, endpoint=False)
                tprev = t
                yprev = y                
                
            envrange = min(local_max[-1][0], local_min[-1][0])
            average = (env_high[:envrange] + env_low[:envrange]) * 0.5
            p0 = np.mean((env_high[1:envrange] - env_high[:envrange - 1]) / (xdata[1] - xdata[0]))
            """            
            p0 = None
            popt, _ = sciopt.curve_fit(line, xdata, ydata, p0=p0)
            
            residual = abs(np.sum(ydata - popt[0] * xdata) / np.sum(ydata))
            if min_residual is None or residual < min_residual:
                min_residual = residual
                
            if residual < fit_tol:
                if save_result_to:
                    with h5py.File(f'{save_result_to}.h5', 'a') as out:
                        out['fit_success'][icm] = 1
                        out['fit_range'][icm] = [start, end]
                        out['fit_residual'][icm] = residual
                    
                break
                
            start += int(xdata.shape[0] * 0.1)
            end -= int(xdata.shape[0] * 0.1)
                
        pauli_coeffs[icm] = popt[0]

    return pauli_coeffs


def find_heff(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    extraction_fn: Callable = maximize_process_fidelity,
    extraction_params: Dict = dict(),
    save_result_to: Optional[str] = None,
) -> np.ndarray:
    """Run a pulse simulation with constant drives and extract the Pauli components of the effective Hamiltonian.
    
    QuTiP `sesolve` applied to the identity matrix will give the time evolution operator :math:`U_H(t)` according
    to the rotating-wave Hamiltonian :math:`H` at each time point. If an effective Hamiltonian 
    :math:`H_{\mathrm{eff}}` is to be found, the evolution should be approximatable with
    :math:`\exp(-i H_{\mathrm{eff}} t)`. This function takes the matrix-log of calculated :math:`U_H(t)`, extracts
    the Pauli coefficients at each time point, and performs a linear fit to each coefficient as a function of time.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of `pulse_sim.make_hamiltonian_components` for details.
        drive_def: Drive definition. See the docstring of `pulse_sim.DriveExprGen` for details. Argument `'amplitude'` for
            each channel must be a constant expression (float or string).
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
            :math:`(\lambda_i \otimes \lambda_j \otimes \dots)/2^{n-1}` of the effective Hamiltonian.
    """
    
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    for key, value in drive_def.items():
        try:
            amp_factor = value['amplitude']
        except KeyError:
            raise RuntimeError(f'Missing amplitude specification for drive on channel {key}')

        if isinstance(amp_factor, str) and 't' in amp_factor:
            raise RuntimeError(f'Cannot use time-dependent amplitude (found in channel {key})')
            
    ## Evolve the identity operator to obtain the time evolution operators
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)

    result = run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=(10, 400),
        save_result_to=save_result_to)
    
    time_evolution = np.concatenate(list(np.expand_dims(state.full(), axis=0) for state in result.states))
    
    if save_result_to:
        with h5py.File(f'{save_result_to}.h5', 'w') as out:
            out.create_dataset('time_evolution', data=time_evolution)
            out.create_dataset('tlist', data=result.times)
            
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)

    pauli_coeffs = extraction_fn(
        time_evolution,
        paulis,
        num_qubits,
        result.times,
        save_result_to=save_result_to,
        **extraction_params)

    return pauli_coeffs


def find_gate(
    qubits: Sequence[int],
    params: Dict[str, Any],
    drive_def: Dict[int, Dict[str, float]],
    tlist: np.ndarray,
    num_sim_levels: int = 2,
    comp_dim: int = 2,
    save_result_to: Optional[str] = None
) -> np.ndarray:
    """Run a pulse simulation and return the log of the resulting unitary.
    
    This function computes the time evolution operator :math:`U_{\mathrm{pulse}}` effected by the drive pulse
    and returns :math:`i \log U_{\mathrm{pulse}}`, projected onto the computational space if the simulation is
    performed with more levels than computational dimension. The returned value is given as an array of Pauli
    coefficients.

    Args:
        qubits: List of qudits to include in the Hamiltonian.
        params: Hamiltonian parameters. See the docstring of `pulse_sim.make_hamiltonian_components` for details.
        drive_def: Drive definition. See the docstring of `pulse_sim.DriveExprGen` for details.
        tlist: Time points to use in the simulation.
        num_sim_levels: Number of oscillator levels in the simulation.
        comp_dim: Dimensionality of the computational space.
        save_result_to: File name (without the extension) to save the simulation and extraction results to.
        
    Returns:
        An array with the value at index `[i, j, ..]` corresponding to the coefficient of
        :math:`\lambda_i \otimes \lambda_j \otimes \dots` in :math:`i log U_{\mathrm{pulse}}`.
    """
    
    ## Validate and format the input
    
    if isinstance(qubits, int):
        qubits = (qubits,)
        
    num_qubits = len(qubits)
    
    assert comp_dim <= num_sim_levels, 'Number of levels in simulation cannot be less than computational dimension'
    
    for key, value in drive_def.items():
        try:
            amp_factor = value['amplitude']
        except KeyError:
            raise RuntimeError(f'Missing amplitude specification for drive on channel {key}')

    ## Evolve the identity operator to obtain the evolution operator corresponding to the pulse
    
    psi0 = qtp.tensor([qtp.qeye(num_sim_levels)] * num_qubits)

    result = run_pulse_sim(
        qubits, params, drive_def,
        psi0=psi0,
        tlist=tlist,
        save_result_to=save_result_to)
    
    ## Take the log of the evolution operator

    # Apparently sesolve always store the states for all time points regardless of the options..
    unitary = result.states[-1]
    
    eigvals, eigcols = np.linalg.eig(unitary)
    eigrows = np.conjugate(np.transpose(eigcols))
    
    ilog_diagonal = np.diag(-np.angle(eigvals))

    ilog_u = eigcols @ ilog_diagonal @ eigrows
    
    ## Extract the (generalized) Pauli components
    
    paulis = make_generalized_paulis(comp_dim, matrix_dim=num_sim_levels)
    prod_basis = make_prod_basis(paulis, num_qubits)
    
    # Compute the inner product (trace of matrix product) with the prod_basis at each time point
    # Implicitly using the 17-qubit limit in assuming that the indices of the basis won't reach x
    qubit_indices = string.ascii_letters[:num_qubits]
    pauli_coeffs = np.einsum(f'xy,{qubit_indices}yx->{qubit_indices}', ilog_u, prod_basis).real
    # Divide the trace by two to account for the normalization of the generalized Paulis
    pauli_coeffs /= 2.
    
    return pauli_coeffs


