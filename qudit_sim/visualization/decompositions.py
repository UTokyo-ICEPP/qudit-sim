"""Visualizations of Pauli decompositions of Hamiltonians and gates."""

from typing import List, Optional, Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import numpy as np
try:
    get_ipython()
except NameError:
    HAS_IPYTHON = False
    PrintReturnType = str
else:
    HAS_IPYTHON = True
    from IPython.display import Latex
    PrintReturnType = Latex

from rqutils.math import matrix_angle
import rqutils.paulis as paulis
from rqutils.qprint import QPrintPauli

from ..basis import change_basis, matrix_labels
from ..scale import FrequencyScale
from ..sim_result import PulseSimResult


def print_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    symbols: Optional[Union[str, List[str]]] = None,
    precision: int = 3,
    threshold: float = 1.e-3,
    lhs_label: Optional[str] = None,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto,
    basis: Optional[str] = None
) -> PrintReturnType:
    r"""Compose a LaTeX expression of the effective Hamiltonian from the Pauli components.

    Args:
        components: Array of Pauli components returned by find_heff.
        uncertainties: Array of component uncertainties.
        symbol: Symbol to use instead of :math:`\lambda` for the matrices.
        precision: Number of digits below the decimal point to show.
        threshold: Ignore terms with absolute components below this value relative to the given
            scale (if >0) or to the maximum absolute component (if <0).
        lhs_label: Left-hand-side label.
        scale: Normalize the components with the frequency scale. If None, components are taken
            to be dimensionless. If `FrequencyScale.auto`, scale is found from the maximum absolute
            value of the components. String `'pi'` is also allowed, in which case the components are
            normalized by :math:`\pi`.
        basis: Represent the components in the given matrix basis.

    Returns:
        A representation object for a LaTeX expression or an expression string for the effective
        Hamiltonian.
    """
    if basis is not None:
        components = change_basis(components, to_basis=basis)
        if symbols is None:
            pauli_dim = tuple(np.around(np.sqrt(components.shape)).astype(int))
            symbols = list(matrix_labels(basis, dim) for dim in pauli_dim)

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
        amp_cutoff = -threshold

    if uncertainties is not None:
        if basis is not None:
            uncertainties = change_basis(uncertainties, to_basis=basis)

        selected = np.nonzero(np.abs(components) > amp_cutoff * max_abs)
        unc = np.zeros_like(uncertainties)
        unc[selected] = uncertainties[selected] / scale_omega

        central = QPrintPauli(components, amp_format=f'.{precision}f',
                              amp_cutoff=amp_cutoff, symbol=symbols)

        uncert = QPrintPauli(unc, amp_format=f'.{precision}f',
                             amp_cutoff=0., symbol=symbols)

        if HAS_IPYTHON:
            return Latex(fr'\begin{{split}} {lhs_label} & = {central.latex(env=None)} \\'
                         + fr' & \pm {uncert.latex(env=None)} \end{{split}}')
        else:
            return f'{lhs_label}  = {central}\n{" " * len(lhs_label)} +- {uncert}'

    else:
        pobj = QPrintPauli(components, amp_format=f'.{precision}f', amp_cutoff=amp_cutoff,
                           lhs_label=lhs_label, symbol=symbols)

        if HAS_IPYTHON:
            return Latex(pobj.latex())
        else:
            return str(pobj)


def plot_components(
    components: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    symbols: Optional[Union[str, List[str]]] = None,
    threshold: float = 1.e-2,
    scale: Union[FrequencyScale, str, None] = FrequencyScale.auto,
    ignore_identity: bool = True,
    basis: Optional[str] = None
) -> mpl.figure.Figure:
    """Plot the Hamiltonian components as a bar graph in the decreasing order in the absolute value.

    Args:
        components: Array of Pauli components returned by find_heff.
        uncertainties: Array of component uncertainties.
        symbols: Symbols to use instead of the numeric indices for the matrices.
        threshold: Ignore terms with absolute components below this value relative to the given
            scale (if >0) or to the maximum absolute component (if <0).
        scale: Normalize the components with the frequency scale. If None, components are taken
            to be dimensionless. If `FrequencyScale.auto`, scale is found from the maximum absolute
            value of the components. String `'pi'` is also allowed, in which case the components are
            normalized by :math:`\pi`.
        ignore_identity: Ignore the identity term.
        basis: Represent the components in the given matrix basis.

    Returns:
        A Figure object containing the bar graph.
    """
    pauli_dim = tuple(np.around(np.sqrt(components.shape)).astype(int))
    num_qudits = len(components.shape)

    if basis is not None:
        components = change_basis(components, to_basis=basis)
        if symbols is None:
            symbols = list(matrix_labels(basis, dim) for dim in pauli_dim)

    max_abs = np.amax(np.abs(components))

    if scale is FrequencyScale.auto:
        scale = FrequencyScale.find_energy_scale(max_abs)

    if num_qudits == 1:
        power_of_two_sup = ''
    elif num_qudits == 2:
        power_of_two_sup = '/2'
    else:
        power_of_two_sup = f'/ 2^{{{num_qudits - 1}}}'

    if scale is None:
        scale_omega = 1.
        ylabel = fr'$\theta {power_of_two_sup}$'
    elif scale == 'pi':
        scale_omega = np.pi
        ylabel = fr'$\theta {power_of_two_sup}/\pi$'
    else:
        scale_omega = scale.pulsatance_value
        # If we normalize by 2*pi*frequency, the displayed values are in frequency
        ylabel = fr'$\nu {power_of_two_sup}\,(2\pi\,\mathrm{{{scale.frequency_unit}}})$'

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

    # Renormalize the components to account for the power-of-two suppression
    components /= 2 ** (num_qudits - 1)

    if uncertainties is None:
        yerr = None
    else:
        uncertainties = uncertainties / scale_omega
        if ignore_identity:
            uncertainties[identity_index] = 0.

        uncertainties /= 2 ** (num_qudits - 1)

        yerr = uncertainties[indices]

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(nterms), components[indices], yerr=yerr)

    ax.axhline(0., color='black', linewidth=0.5)

    if symbols is None:
        symbols = [''] * len(pauli_dim)
        delimiter = r'\,' if pauli_dim[0] == 2 else ','
    else:
        delimiter = r'\,'

    labels = paulis.labels(pauli_dim, symbol=symbols, delimiter=delimiter, norm=False)

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
    title: str = '',
    basis: Optional[str] = None,
    symbols: Optional[Union[str, List[str]]] = None,
    smooth: bool = False
) -> Tuple[List[Tuple[int, ...]], mpl.figure.Figure]:
    r"""Plot the Pauli components of the generator of a time evolution as a function of time.

    The time evolution, time points, and the operator dimension can either be passed as a simulation
    result object or individually.

    Args:
        sim_result: Simulation result object. If not None, ``time_evolution``, ``tlist``, and
            ``dim`` are ignored.
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
        basis: Represent the components in the given matrix basis.
        symbols: Symbols to use instead of the numeric indices for the matrices.
        smooth: (Experimental) Perform a fit to every single time evolution unitary with the initial
            value of the generator components taken from the previous step.

    Returns:
        The indices of the plotted components and the plot figure.
    """
    if sim_result is not None:
        time_evolution = sim_result.states
        tlist = sim_result.times
        dim = sim_result.frame.dim

    if tscale is FrequencyScale.auto:
        tscale = FrequencyScale.find_time_scale(tlist[-1])

    if tscale is not None:
        tlist = tlist * tscale.frequency_value

    if differential:
        time_evolution = time_evolution[1:] @ time_evolution[:-1].transpose((0, 2, 1)).conjugate()
        tlist = tlist[1:]

    if smooth:
        components, ev = smooth_components(time_evolution, dim)
        tlist = tlist
    else:
        generator, ev = matrix_angle(time_evolution, with_diagonals=True)
        components = paulis.components(-1. * generator, dim=dim).real

    if basis is not None:
        components = change_basis(components, to_basis=basis, num_qudits=len(dim))
        if symbols is None:
            symbols = list(matrix_labels(basis, d) for d in dim)

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
        labels = paulis.labels(dim, symbol=symbols, norm=False)

        if align_ylim:
            indices_array = tuple(zip(*select_components))
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


def smooth_components(
    time_evolution: np.ndarray,
    dim: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify the generator components and eigenvalues of the time evolution unitaries through fits."""
    def loss_fn(params, unitary):
        hermitian = paulis.compose(params['components'], dim, npmod=jnp)
        ansatz = jax.scipy.linalg.expm(1.j * hermitian)
        return -jnp.square(jnp.abs(jnp.trace(unitary @ ansatz)))

    value_and_grad = jax.value_and_grad(loss_fn)
    grad_trans = optax.adam(0.005)

    @jax.jit
    def step(opt_params, opt_state, unitary):
        loss, gradient = value_and_grad(opt_params, unitary)
        updates, opt_state = grad_trans.update(gradient, opt_state)
        new_params = optax.apply_updates(opt_params, updates)
        return new_params, opt_state, loss, gradient

    components = np.empty((time_evolution.shape[0],) + tuple(np.square(dim)))
    eigenvalues = np.empty(time_evolution.shape[:2])

    initial = paulis.components(-matrix_angle(time_evolution[0]), dim=dim).real

    max_update = 10000
    window = 20
    losses = np.empty(window)

    for itime in range(time_evolution.shape[0]):
        opt_params = {'components': jnp.array(initial)}
        opt_state = grad_trans.init(opt_params)

        for iup in range(max_update):
            new_params, opt_state, loss, gradient = step(opt_params, opt_state, time_evolution[itime])

            losses[iup % window] = loss
            if iup > window and np.amax(losses) - np.amin(losses) < 1.e-4:
                break

            opt_params = new_params

        print(f'done in {iup} steps.')

        initial = opt_params['components']

        components[itime] = np.array(opt_params['components'])

        hermitian = paulis.compose(components[itime], dim=dim)
        eigenvalues[itime] = np.linalg.eigh(hermitian)[0]

    return components, eigenvalues
