"""Simulation result class and I/O."""

from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import h5py

from .frame import SystemFrame

@dataclass(frozen=True)
class PulseSimResult:
    """Return type of pulse_sim.

    See the docstring of pulse_sim for why this class is necessary.
    """
    times: np.ndarray
    expect: Union[List[np.ndarray], None]
    states: Union[np.ndarray, None]
    frame: SystemFrame


def save_sim_result(filename: str, result: PulseSimResult):
    """Save the pulse simulation result to an HDF5 file."""
    with h5py.File(filename, 'w') as out:
        out.create_dataset('times', data=result.times)
        if result.expect is not None:
            out.create_dataset('expect', data=result.expect)
        if result.states is not None:
            out.create_dataset('states', data=result.states)
        out.create_dataset('qudit_ids', data=np.array(list(result.frame.keys()), dtype=object))
        out.create_dataset('frame', data=np.array([[frame.frequency, frame.phase] for frame in result.frame.values()]))


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
        qudit_ids = source['qudit_ids'][()]
        frame = SystemFrame({qid: QuditFrame(d[0], d[1]) for qid, d in zip(qudit_ids, source['frame'][()])})

    return PulseSimResult(times, expect, states, frame)
