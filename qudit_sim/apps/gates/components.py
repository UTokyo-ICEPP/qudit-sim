"""Gate component extraction functions."""

from typing import Any, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from ...config import config
from ...sim_result import PulseSimResult
from ...unitary import closest_unitary, truncate_matrix


def gate_and_fidelity(
    sim_result: PulseSimResult,
    comp_dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> Tuple[np.ndarray, float]:
    r"""Get the gate unitary from the final state of a simulation result.

    Args:
        sim_result: Pulse simulation result.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in the
            simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`) and
        the fidelity of the truncated gate.
    """
    gate = sim_result.states[-1]
    frame = sim_result.frame

    if isinstance(comp_dim, int):
        comp_dim = (comp_dim,) * frame.num_qudits

    if comp_dim is None:
        return gate, 1.

    truncated = truncate_matrix(gate, frame.dim, comp_dim)
    return closest_unitary(truncated, with_fidelity=True)


def gate_components_from_log(
    sim_result: PulseSimResult,
    comp_dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    r"""Compute the Pauli components of the generator of the unitary obtained from the simulation.

    Args:
        sim_result: Pulse simulation result or a list thereof.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`).
    """
    gate, _ = gate_and_fidelity(sim_result, comp_dim)

    if comp_dim is None:
        comp_dim = sim_result.frame.dim
    elif isinstance(comp_dim, int):
        comp_dim = (comp_dim,) * sim_result.frame.num_qudits

    components = paulis.components(-matrix_angle(gate), comp_dim).real

    return components


def gate_components(
    sim_result: PulseSimResult,
    initial_guess: np.ndarray,
    optimizer: str = 'adam',
    optimizer_args: Optional[Any] = 0.005,
    max_update: int = 1000,
    convergence: float = 1.e-5,
    convergence_window: int = 5,
    with_loss: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Find the components of log unitary through fidelity maximization."""
    gate = sim_result.states[-1]
    frame = sim_result.frame

    comp_dim = tuple(map(int, np.around(np.sqrt(initial_guess.shape))))

    trunc_gate = truncate_matrix(gate, frame.dim, comp_dim)

    def loss_fn(params):
        hermitian = paulis.compose(params['components'], comp_dim, npmod=jnp)
        ansatz = jax.scipy.linalg.expm(1.j * hermitian)
        trace = jnp.trace(trunc_gate @ ansatz)
        return -(jnp.square(trace.real) + jnp.square(trace.imag))

    value_and_grad = jax.value_and_grad(loss_fn)

    ## Set up the optimizer, loss & grad functions, and the parameter update function
    if not isinstance(optimizer_args, tuple):
        optimizer_args = (optimizer_args,)

    grad_trans = getattr(optax, optimizer)(*optimizer_args)

    @jax.jit
    def step(opt_params, opt_state):
        loss, gradient = value_and_grad(opt_params)
        updates, opt_state = grad_trans.update(gradient, opt_state)
        new_params = optax.apply_updates(opt_params, updates)

        return new_params, opt_state, loss

    jax_device = jax.devices()[config.jax_devices[0]]

    opt_params = {'components': initial_guess}
    with jax.default_device(jax_device):
        opt_state = grad_trans.init(opt_params)
        step = step.lower(opt_params, opt_state).compile()

    losses = np.zeros(max_update)

    for iup in range(max_update):
        # https://github.com/google/jax/issues/11478
        # default_device is thread-local
        with jax.default_device(jax_device):
            opt_params, opt_state, loss = step(opt_params, opt_state)

        losses[iup] = loss

        if iup >= convergence_window:
            window_losses = losses[iup - convergence_window:iup + 1]
            change = np.amax(window_losses) - np.amin(window_losses)

            if change < convergence:
                break

    losses = losses[:iup + 1]

    components = np.array(opt_params['components'])

    if with_loss:
        return components, losses

    return components
