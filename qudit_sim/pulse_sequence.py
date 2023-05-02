r"""
================================================
Pulse sequence (:mod:`qudit_sim.pulse_sequence`)
================================================

.. currentmodule:: qudit_sim.pulse_sequence

Implementation of pulse sequence.
"""
import copy
from typing import Any, Dict, Union
import numpy as np

from .drive import Delay
from .expression import TimeFunction
from .pulse import Pulse


class PulseSequence(list):
    """Pulse sequence.

    This class represents a sequence of instructions (pulse, delay, frequency/phase shift/set)
    given to a single channel.
    """

    @property
    def duration(self):
        d = sum(inst.value for inst in self if isinstance(inst, Delay))
        d += sum(inst.duration for inst in self if isinstance(inst, Pulse))
        return d

    def __str__(self):
        return f'PulseSequence([{", ".join(str(inst) for inst in self)}])'

    def envelope(self, t: Union[float, np.ndarray], args: Dict[str, Any] = dict()) -> np.ndarray:
        """Return the envelope of the sequence as a function of time.

        This function is mostly for visualization purposes. Phase and frequency information is lost in the
        returned array.

        Args:
            t: Time or array of time points.
            args: Second argument to the pulse envelope functions.

        Returns:
            Pulse sequence envelope (complex) as a function of time.
        """
        funclist = list()
        time = 0.

        for inst in self:
            if isinstance(inst, Delay):
                funclist.append((time, 0.))
                time += inst.value
            elif isinstance(inst, TimeFunction):
                fn = copy.copy(inst)
                fn.tzero = time
                funclist.append((time, fn))
                time += inst.duration

        funclist.append((time, None))

        result = 0.
        for time, func in funclist[:-1]:
            if isinstance(func, TimeFunction):
                result = np.where(t > time, func(t, args), result)
            else:
                result = np.where(t > time, func, result)

        result = np.where(t > funclist[-1][0], 0., result)

        return result
