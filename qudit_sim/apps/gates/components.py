from typing import Union, List, Optional
import numpy as np

from rqutils.math import matrix_angle
import rqutils.paulis as paulis

from ...sim_result import PulseSimResult

def gate_components(
    sim_result: Union[PulseSimResult, List[PulseSimResult]],
    comp_dim: Optional[int] = None
) -> np.ndarray:
    r"""Compute the Pauli components of the generator of the unitary obtained from the simulation.

    Args:
        sim_result: Pulse simulation result or a list thereof.
        comp_dim: Interpret the result in the given matrix dimension. If None, dimension in
            the simulation is used.

    Returns:
        Array of Pauli components of the generator of the gate (:math:`i \mathrm{log} U`), or a
        list of such arrays if `sim_result` is an array.
    """
    if isinstance(sim_result, list):
        return list(_single_gate_components(res, comp_dim) for res in sim_result)
    else:
        return _single_gate_components(sim_result, comp_dim)


def _single_gate_components(sim_result: PulseSimResult, comp_dim):
    gate = sim_result.states[-1]
    components = paulis.components(-matrix_angle(gate), sim_result.dim).real

    if comp_dim is not None and sim_result.dim[0] != comp_dim:
        components = paulis.truncate(components, (comp_dim,) * len(sim_result.dim))

    return components
