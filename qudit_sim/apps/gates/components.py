from typing import Union, List, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from ...sim_result import PulseSimResult
from ...config import config

def get_gate(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: Optional[int] = None
) -> Union[Tuple[np.ndarray, float], List[Tuple[np.ndarray, float]]]:
    r"""Get the gate unitary from the final state of (a) simulation result(s).

    Args:
        sim_result: Pulse simulation result or a list thereof.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`) and
        the fidelity of the truncated gate, or a list thereof if `sim_result` is a list.
    """
    if isinstance(sim_result, list):
        return list(get_gate(res, comp_dim) for res in sim_result)

    gate = sim_result.states[-1]

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        num_qudits = len(sim_result.dim)
        comp_dim_tuple = (comp_dim,) * num_qudits
        flattened_grid_indices = np.indices(comp_dim_tuple).reshape(num_qudits, -1)
        trunc_indices = np.ravel_multi_index(flattened_grid_indices, sim_result.dim)
        # example: (4, 4) -> (3, 3): trunc_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        trunc_gate = gate[trunc_indices]

        # Find the closest unitary to the truncation
        v, _, wdag = np.linalg.svd(trunc_gate)

        # Reconstruct (reunitarize) the gate
        reco_gate = v @ wdag

        fidelity = np.square(np.abs(np.trace(trunc_gate @ reco_gate.conjugate().T)))
        fidelity /= np.square(np.prod(comp_dim_tuple))

        gate = reco_gate

    else:
        fidelity = 1.

    return gate, fidelity


def gate_components_from_log(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: Optional[int] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    r"""Compute the Pauli components of the generator of the unitary obtained from the simulation.

    Args:
        sim_result: Pulse simulation result or a list thereof.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`), or a
        list of such arrays if `sim_result` is a list.
    """
    if isinstance(sim_result, list):
        return list(gate_components_from_log(res, comp_dim) for res in sim_result)

    gate, _ = get_gate(sim_result, comp_dim)

    components = paulis.components(-matrix_angle(gate), sim_result.dim).real

    return components


def gate_components(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    heff: np.ndarray,
    approx_time: float,
    max_update: int = 1000,
    comp_dim: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    if isinstance(sim_result, list):
        return list(gate_components(res, heff, approx_time, max_update, comp_dim)
                    for res in sim_result)

    jax_device = jax.devices()[config.jax_devices[0]]

    gate, _ = get_gate(sim_result, comp_dim)

    unitary = jax.device_put(gate, device=jax_device)

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        comp_dim_tuple = (comp_dim,) * len(sim_result.dim)
        heff = paulis.truncate(heff, comp_dim_tuple)
    else:
        comp_dim_tuple = sim_result.dim

    def loss_fn(params):
        with jax.default_device(jax_device):
            hermitian = paulis.compose(params['components'], comp_dim_tuple, npmod=jnp)
            ansatz = jax.scipy.linalg.expm(1.j * hermitian)
            return -jnp.square(jnp.abs(jnp.trace(unitary @ ansatz)))

    value_and_grad = jax.value_and_grad(loss_fn)
    grad_trans = optax.adam(0.001)

    @jax.jit
    def step(opt_params, opt_state):
        # https://github.com/google/jax/issues/11478
        # default_device is thread-local
        with jax.default_device(jax_device):
            loss, gradient = value_and_grad(opt_params)
            updates, opt_state = grad_trans.update(gradient, opt_state)
            new_params = optax.apply_updates(opt_params, updates)

        return new_params, opt_state, loss, gradient

    opt_params = {'components': heff * approx_time}
    with jax.default_device(jax_device):
        opt_state = grad_trans.init(opt_params)

    losses = np.zeros(max_update)

    for iup in range(max_update):
        new_params, opt_state, loss, gradient = step(opt_params, opt_state)

        losses[iup] = loss
        opt_params = new_params

    return np.array(opt_params['components']), losses
