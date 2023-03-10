from typing import Union, List, Tuple, Optional, Any
import numpy as np
import jax
import jax.numpy as jnp
import optax

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from ...sim_result import PulseSimResult
from ...unitary import truncate_matrix, closest_unitary
from ...config import config


def gate_and_fidelity(
    sim_result: PulseSimResult,
    comp_dim: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    r"""Get the gate unitary from the final state of a simulation result.

    Args:
        sim_result: Pulse simulation result.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`) and
        the fidelity of the truncated gate.
    """
    gate = sim_result.states[-1]

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        truncated = truncate_matrix(gate, len(sim_result.dim), sim_result.dim[0], comp_dim)
        return closest_unitary(truncated, with_fidelity=True)
    else:
        return gate, 1.


def gate_components_from_log(
    sim_result: PulseSimResult,
    comp_dim: Optional[int] = None
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
        components_dim = sim_result.dim
    else:
        components_dim = (comp_dim,) * len(sim_result.dim)

    components = paulis.components(-matrix_angle(gate), components_dim).real

    return components


def gate_components(
    sim_result: PulseSimResult,
    target: np.ndarray,
    optimizer: str = 'adam',
    optimizer_args: Optional[Any] = 0.005,
    max_update: int = 1000,
    convergence: float = 1.e-5,
    convergence_window: int = 5,
    with_loss: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    jax_device = jax.devices()[config.jax_devices[0]]

    num_qudits = len(sim_result.dim)
    comp_dim = np.sqrt(target.shape[0]).astype(int)
    dim = (comp_dim,) * num_qudits

    trunc_gate = truncate_matrix(sim_result.states[-1], num_qudits, sim_result.dim[0], comp_dim)
    trunc_gate = jax.device_put(trunc_gate, device=jax_device)

    def loss_fn(params):
        with jax.default_device(jax_device):
            hermitian = paulis.compose(params['components'], dim, npmod=jnp)
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
        # https://github.com/google/jax/issues/11478
        # default_device is thread-local
        with jax.default_device(jax_device):
            loss, gradient = value_and_grad(opt_params)
            updates, opt_state = grad_trans.update(gradient, opt_state)
            new_params = optax.apply_updates(opt_params, updates)

        return new_params, opt_state, loss

    opt_params = {'components': target}
    with jax.default_device(jax_device):
        opt_state = grad_trans.init(opt_params)

    losses = np.zeros(max_update)

    for iup in range(max_update):
        opt_params, opt_state, loss = step(opt_params, opt_state)

        losses[iup] = loss

        if iup >= convergence_window:
            window_losses = losses[iup - convergence_window:iup + 1]
            change = np.amax(window_losses) - np.amin(window_losses)

            if change < convergence:
                break

    components = np.array(opt_params['components'])

    if with_loss:
        return components, losses
    else:
        return components
