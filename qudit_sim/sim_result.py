"""Simulation result class and I/O."""

from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import h5py

@dataclass(frozen=True)
class PulseSimResult:
    """Return type of pulse_sim.

    See the docstring of pulse_sim for why this class is necessary.
    """
    times: np.ndarray
    expect: Union[List[np.ndarray], None]
    states: Union[np.ndarray, None]
    dim: Tuple[int, ...]
    frame: Tuple[Frame, ...]


def save_sim_result(filename: str, result: PulseSimResult):
    """Save the pulse simulation result to an HDF5 file."""
    with h5py.File(filename, 'w') as out:
        out.create_dataset('times', data=result.times)
        if result.expect is not None:
            out.create_dataset('expect', data=result.expect)
        if result.states is not None:
            out.create_dataset('states', data=result.states)
        out.create_dataset('dim', data=np.array(result.dim, dtype=int))
        out.create_dataset('frame', data=np.array([[frame.frequency, frame.phase] for frame in result.frame]))


def load_sim_result(filename: str) -> PulseSimResult:
    """Load the pulse simulation result from an HDF5 file."""
    with h5py.File(filename, 'r') as source:
        times = source['times'][()]
        try:
            expect = source['expect'][()]
        except KeyError:
            expect = None
        try:
            states = source['states'][()]
        except KeyError:
            states = None
        dim = tuple(source['dim'][()])
        frame = tuple(Frame(d[0], d[1]) for d in source['frame'][()])

    return PulseSimResult(times, expect, states, dim, frame)
