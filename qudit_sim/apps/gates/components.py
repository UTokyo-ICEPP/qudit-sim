from typing import Union, List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from ...sim_result import PulseSimResult
from ...unitary import truncate_matrix, closest_unitary
from ...config import config
from ...parallel import parallel_map

def gate_and_fidelity(
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
        return list(gate_and_fidelity(res, comp_dim) for res in sim_result)

    gate = sim_result.states[-1]

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        truncated = truncate_matrix(gate, len(sim_result.dim), sim_result.dim[0], comp_dim)
        return closest_unitary(truncated, with_fidelity=True)
    else:
        return gate, 1.


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

    gate, _ = gate_and_fidelity(sim_result, comp_dim)

    components = paulis.components(-matrix_angle(gate), sim_result.dim).real

    return components


def gate_components(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    heff: np.ndarray,
    approx_time: Optional[float] = None,
    max_update: int = 1000
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    if isinstance(sim_result, list):
        args = sim_result
        common_args = (heff, approx_time, max_update)
        return parallel_map(gate_components, args=args, common_args=common_args, thread_based=True)

    if approx_time is None:
        approx_time = sim_result.times[-1] * 0.5

    jax_device = jax.devices()[config.jax_devices[0]]

    comp_dim = np.sqrt(heff.shape[0]).astype(int)
    dim = (comp_dim,) * len(sim_result.dim)

    gate, _ = gate_and_fidelity(sim_result, comp_dim)

    unitary = jax.device_put(gate, device=jax_device)

    def loss_fn(params):
        with jax.default_device(jax_device):
            hermitian = paulis.compose(params['components'], dim, npmod=jnp)
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
