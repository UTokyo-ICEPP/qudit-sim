r"""
====================================================
Frame (:mod:`qudit_sim.frame`)
====================================================

.. currentmodule:: qudit_sim.frame

See :doc:`/hamiltonian` for theoretical background.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import qutip as qtp

@dataclass(frozen=True)
class QuditFrame:
    """Frame specification for a single qudit."""
    frequency: np.ndarray
    phase: np.ndarray

    @property
    def num_levels(self) -> int:
        return self.frequency.shape[0] + 1

FrameSpec = Union[str, Dict[str, QuditFrame], Sequence[QuditFrame]]

class SystemFrame(dict):
    """Frame specification of a multi-qudit system."""
    def __init__(
        self,
        frame_spec: Optional[FrameSpec] = None,
        hgen: Optional['HamiltonianBuilder'] = None
    ):
        if hgen is None:
            if isinstance(frame_spec, dict):
                super().__init__(frame_spec)
            else:
                raise RuntimeError('frame_spec must be a dict if hgen is not provided')
        else:
            if isinstance(frame_spec, dict):
                super().__init__({qid: frame_spec[qid] for qid in hgen.qudit_ids()})
            elif isinstance(frame_spec, str):
                super().__init__(self.compute_frame(frame_spec, hgen))
            else:
                super().__init__({qid: frame_spec[iq] for iq, qid in enumerate(hgen.qudit_ids())})

    @property
    def num_qudits(self) -> int:
        return len(self)

    @property
    def dim(self) -> Tuple[int, ...]:
        return tuple(qudit_frame.num_levels for qudit_frame in self.values())

    @property
    def frequencies(self) -> np.ndarray:
        return np.array(list(qudit_frame.frequency for qudit_frame in self.values()))

    @property
    def phases(self) -> np.ndarray:
        return np.array(list(qudit_frame.phase for qudit_frame in self.values()))

    @staticmethod
    def compute_frame(frame_name: str, hgen: 'HamiltonianBuilder') -> Dict[str, QuditFrame]:
        """Compute the frequencies of a named frame."""
        drive_frame = False

        if frame_name.startswith('drive'):
            # drive|base_frame
            frame_name = frame_name[6:]
            if not frame_name:
                frame_name = 'dressed'

            drive_frame = True

        qudit_ids = hgen.qudit_ids()

        no_coupling = (sum(hgen.coupling(q1, q2) for q1 in qudit_ids for q2 in qudit_ids) == 0.)

        if (frame_name == 'dressed' or frame_name.startswith('noiz')) and no_coupling:
            frame_name = 'qudit'

        if frame_name == 'qudit':
            frequencies = hgen.free_frequencies()
        elif frame_name == 'lab':
            frequencies = {qid: np.zeros(hgen.qudit_params(qid).num_levels - 1)
                           for qid in qudit_ids}
        elif frame_name == 'dressed':
            frequencies = hgen.dressed_frequencies()
        elif frame_name.startswith('noiz'):
            if len(frame_name) > 4:
                comp_dim = int(frame_name[4:])
            else:
                comp_dim = 0

            frequencies = hgen.noiz_frequencies(comp_dim=comp_dim)
        else:
            raise ValueError(f'Global frame {frame_name} is not defined')

        if drive_frame:
            for qid, drives in hgen.drive().items():
                if len(drives) == 1:
                    frequencies[qid] = np.full(hgen.qudit_params(qid).num_levels - 1,
                                               drives[0].frequency)
                elif len(drives) != 0:
                    raise RuntimeError(f'Qudit {qid} has more than one drive terms')

        return {qid: QuditFrame(frequencies[qid], np.zeros(hgen.qudit_params(qid).num_levels - 1))
                for qid in qudit_ids}

    def set_frequency(
        self,
        qudit_id: str,
        frequency: np.ndarray
    ):
        current = self[qudit_id]
        self[qudit_id] = QuditFrame(frequency, current.phase)

    def set_phase(
        self,
        qudit_id: str,
        phase: np.ndarray
    ):
        current = self[qudit_id]
        self[qudit_id] = QuditFrame(current.frequency, phase)

    def frame_change_operator(
        self,
        from_frame: FrameSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the change-of-frame operator.

        Change of frame from

        .. math::

            U_{f}(t) = \bigotimes_j \exp \left[ i \sum_l \left( \Xi_j^l t + \Phi_j^l \right)
                                                         | l \rangle_j \langle l |_j \right]

        to

        .. math::

            U_{g}(t) = \bigotimes_j \exp \left[ i \sum_l \left( \Eta_j^l t + \Psi_j^l \right)
                                                         | l \rangle_j \langle l |_j \right]

        is effected by

        .. math::

            V_{gf}(t) = U_g(t) U_f^{\dagger}(t) = \bigotimes_j \exp \left[
                                                  i \sum_l \left\{ (\Eta_j^l - \Xi_j^l) t
                                                  + (\Psi_j^l - \Phi_j^l) \right\}
                                                  | l \rangle_j \langle l |_j \right].

        Args:
            from_frame: Frame to change from.

        Returns:
            Flattened arrays corresponding to :math:`[\sum_{j} \Xi_j^{l_j}]_{\{l_j\}}` and
            :math:`[\sum_{j} \Phi_j^{l_j}]_{\{l_j\}}`.
        """
        if not isinstance(from_frame, SystemFrame):
            from_frame = SystemFrame(from_frame)

        energies = []
        offsets = []
        for qudit_id in self.keys():
            frequencies = self[qudit_id].frequency - from_frame[qudit_id].frequency
            energies.append(np.concatenate(([0.], np.cumsum(frequencies))))

            phases = self[qudit_id].phase - from_frame[qudit_id].phase
            offsets.append(np.concatenate(([0.], np.cumsum(phases))))

        en_diagonal = add_outer_multi(energies)
        offset_diagonal = add_outer_multi(offsets)

        return en_diagonal, offset_diagonal

    def change_frame(
        self,
        tlist: np.ndarray,
        obj: Union[np.ndarray, qtp.Qobj],
        from_frame: FrameSpec,
        objtype: str = 'evolution',
        t0: Optional[float] = None
    ) -> np.ndarray:
        r"""Apply the change-of-frame unitaries to states, unitaries, hamiltonians, and observables.

        Args:
            tlist: 1D array of shape ``(T,)`` representing the time points.
            obj: Qobj or an array of shape either ``(D)`` (state), ``(T, D)`` (state evolution),
                ``(T, D, D)`` (evolution operator, time-dependent Hamiltonian, or an observable),
                ``(D, D)`` (evolution operator for a single time point, Hamiltonian, or an
                observable), ``(N, D, D)`` (multiple observables), or ``(N, T, D, D)``
                (multiple time-dependent observables), where ``D`` is the dimension of the quantum
                system (``prod(system_dim)``) and ``N`` is the number of observables.
            from_frame: Frame to change from.
            objtype: ``'state'``, ``'evolution'``, ``'hamiltonian'``, or ``'observable'``. Ignored
                when the type can be unambiguously inferred from the shape of ``obj``.
            t0: Initial time for the frame change of the evolution operator. If None, ``tlist[0]``
                is used.

        Returns:
            An array representing the frame-changed object.
        """
        if isinstance(obj, qtp.Qobj):
            if obj.isket or obj.isbra:
                obj = np.squeeze(obj.full())
            else:
                obj = obj.full()

        ## Validate and determine the object shape & type
        state_dim = np.prod(list(qudit_frame.num_levels for qudit_frame in self.values()))
        shape_consistent = True
        type_valid = True

        final_shape = obj.shape

        if len(obj.shape) == 1:
            if obj.shape[0] != state_dim:
                shape_consistent = False

            if objtype == 'state':
                obj = obj[None, :]
                final_shape = tlist.shape + final_shape

            else:
                type_valid = False

        elif len(obj.shape) == 2:
            if obj.shape[1] != state_dim:
                shape_consistent = False

            if state_dim != tlist.shape[0] and obj.shape[0] == tlist.shape[0]:
                # Unambiguously determined
                objtype = 'state'

            if objtype == 'state':
                if obj.shape[0] != tlist.shape[0]:
                    shape_consistent = False

            elif objtype in ['evolution', 'hamiltonian', 'observable']:
                if obj.shape[0] != state_dim:
                    shape_consistent = False

                obj = obj[None, ...]
                final_shape = tlist.shape + final_shape

            else:
                type_valid = False

        elif len(obj.shape) == 3:
            if objtype in ['evolution', 'hamiltonian']:
                if obj.shape[0] != tlist.shape[0] or obj.shape[1:] != (state_dim, state_dim):
                    shape_consistent = False

            elif objtype == 'observable':
                if obj.shape[1:] != (state_dim, state_dim):
                    shape_consistent = False

                if obj.shape[0] != tlist.shape[0]:
                    # Ambiguity - we may be talking about T observables, but that's rather improbable
                    obj = obj[:, None, ...]
                    final_shape = (obj.shape[0], tlist.shape[0]) + obj.shape[1:]

            else:
                type_valid = False

        elif len(obj.shape) == 4:
            if obj.shape[1:] != (tlist.shape[0], state_dim, state_dim):
                shape_consistent = False

            if objtype != 'observable':
                type_valid = False

        else:
            shape_consistent = False

        if not type_valid:
            raise ValueError(f'Invalid objtype {objtype} for an obj of shape {obj.shape}')
        if not shape_consistent:
            raise ValueError(f'Inconsistent obj shape {obj.shape} for objtype {objtype} and tlist'
                             f' length {tlist.shape[0]}')

        if not isinstance(from_frame, SystemFrame):
            from_frame = SystemFrame(from_frame)

        en_diagonal, offset_diagonal = self.frame_change_operator(from_frame)
        cof_op_diag = np.exp(1.j * (en_diagonal[None, :] * tlist[:, None] + offset_diagonal))

        if objtype == 'state':
            # Left-multiplying by a diagonal is the same as element-wise multiplication
            obj = cof_op_diag * obj

        elif objtype == 'evolution':
            # Right-multiplying by the inverse of a diagonal unitary = element-wise multiplication
            # of the columns by the conjugate
            if t0 is None:
                cof_op_diag_t0_conj = cof_op_diag[0].conjugate()
            else:
                cof_op_diag_t0_conj = np.exp(-1.j * (en_diagonal * t0 + offset_diagonal))

            obj = cof_op_diag[:, :, None] * obj * cof_op_diag_t0_conj

        elif objtype == 'hamiltonian':
            obj = cof_op_diag[:, :, None] * obj * cof_op_diag[:, None, :].conjugate()
            obj -= np.diag(en_diagonal)

        elif objtype == 'observable':
            obj = cof_op_diag[:, :, None] * obj * cof_op_diag[:, None, :].conjugate()

        return obj.reshape(final_shape)

    def is_lab_frame(self) -> bool:
        value = True
        for qudit_frame in self.values():
            value &= np.allclose(qudit_frame.frequency, 0.) and np.allclose(qudit_frame.phase, 0.)

        return value


def add_outer_multi(arr):
    # Compute the summation "kron"
    diagonal = np.zeros(1)
    for subarr in arr:
        diagonal = np.add.outer(diagonal, subarr).reshape(-1)

    return diagonal
