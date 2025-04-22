"""Simulation result class and I/O."""

from dataclasses import dataclass
from typing import Union
import h5py
import numpy as np

from .frame import QuditFrame, SystemFrame


@dataclass(frozen=True)
class PulseSimResult:
    """Return type of pulse_sim.

    See the docstring of pulse_sim for why this class is necessary.
    """
    times: np.ndarray
    expect: Union[list[np.ndarray], None]
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
        out.create_dataset('qudit_ids', data=list(result.frame.keys()))
        frame = out.create_group('frame')
        for qudit_id, qudit_frame in result.frame.items():
            data = np.array([qudit_frame.frequency, qudit_frame.phase])
            frame.create_dataset(qudit_id, data=data)


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

        frame = {}
        for qudit_id in source['qudit_ids'].asstr():  # pylint: disable=no-member
            frame_data = source['frame'][qudit_id]
            frame[qudit_id] = QuditFrame(frame_data[0], frame_data[1])

    return PulseSimResult(times, expect, states, SystemFrame(frame))
