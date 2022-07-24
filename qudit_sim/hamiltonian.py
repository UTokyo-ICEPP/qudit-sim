r"""
====================================================
Hamiltonian builder (:mod:`qudit_sim.hamiltonian`)
====================================================

.. currentmodule:: qudit_sim.hamiltonian

See :doc:`/hamiltonian` for theoretical background.
"""

from typing import Any, Dict, Sequence, List, Tuple, Callable, Optional, Union, Hashable
from dataclasses import dataclass
import copy
import numpy as np
import qutip as qtp

from .pulse import PulseSequence, SetFrequency
from .drive import DriveTerm, cos_freq, sin_freq

@dataclass(frozen=True)
class Frame:
    """Frame specification for a single level gap of a qudit."""
    frequency: np.ndarray
    phase: np.ndarray

FrameSpec = Union[str, Dict[Hashable, Frame], Sequence[Frame]]

@dataclass(frozen=True)
class QuditParams:
    """Parameters defining a qudit."""
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float
    drive_weight: np.ndarray


def add_outer_multi(arr):
    # Compute the summation "kron"
    diagonal = np.zeros(1)
    for subarr in arr:
        diagonal = np.add.outer(diagonal, subarr).reshape(-1)

    return diagonal


class HamiltonianBuilder:
    r"""Hamiltonian with static transverse couplings and external drive terms.

    The class has two option flags that alter the Hamiltonian-building behavior:

    - compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
    - use_rwa: If True, apply the rotating-wave approximation to the drive Hamiltonian.

    Args:
        num_levels: Number of energy levels to consider.
        qudits: If passing ``params`` to initialize the Hamiltonian, list of qudit numbers to include.
        params: Hamiltonian parameters given by IBMQ ``backend.configuration().hamiltonian['vars']``, optionally
            augmented with ``'crosstalk'``, which should be a ``dict`` of form ``{(j, k): z}`` specifying the crosstalk
            factor ``z`` (complex corresponding to :math:`\alpha_{jk} e^{i\rho_{jk}}`) of drive on qudit :math:`j` seen
            by qudit :math:`k`. :math:`j` and :math:`k` are qudit ids given in ``qudits``.
        default_frame: Default global frame to use. ``set_global_frame(default_frame, keep_phase=True)`` is executed
            each time ``add_qudit`` or ``add_coupling`` is called.
    """
    def __init__(
        self,
        num_levels: int = 2,
        qudits: Optional[Union[int, Sequence[int]]] = None,
        params: Optional[Dict[str, Any]] = None,
        default_frame: str = 'dressed'
    ) -> None:
        self._num_levels = num_levels

        self.default_frame = default_frame

        # Makes use of dict order guarantee from python 3.7
        self._qudit_params = dict()
        self._coupling = dict()
        self._crosstalk = dict()
        self._drive = dict()
        self._frame = dict()

        self.compile_hint = True
        self.use_rwa = False

        if qudits is None:
            return

        if isinstance(qudits, int):
            qudits = (qudits,)

        for q in qudits:
            self.add_qudit(params[f'wq{q}'], params[f'delta{q}'], params[f'omegad{q}'], qudit_id=q)

        for q1, q2 in zip(qudits[:-1], qudits[1:]):
            try:
                coupling = params[f'jq{min(q1, q2)}q{max(q1, q2)}']
            except KeyError:
                continue

            self.add_coupling(q1, q2, coupling)

        if 'crosstalk' in params:
            for (qsrc, qtrg), factor in params['crosstalk'].items():
                self.add_crosstalk(qsrc, qtrg, factor)

    @property
    def num_levels(self) -> int:
        """Number of considered energy levels."""
        return self._num_levels

    @property
    def num_qudits(self) -> int:
        """Number of qudits."""
        return len(self._qudit_params)

    def qudit_ids(self) -> List[Hashable]:
        """List of qudit IDs."""
        return list(self._qudit_params.keys())

    def qudit_id(self, idx: int) -> Hashable:
        """Qudit ID for the given index."""
        return self.qudit_ids()[idx]

    def qudit_index(self, qudit_id: Hashable) -> int:
        """Qudit index."""
        return next(idx for idx, qid in enumerate(self._qudit_params) if qid == qudit_id)

    def qudit_params(self, qudit_id: Hashable) -> QuditParams:
        """Qudit parameters."""
        return self._qudit_params[qudit_id]

    def coupling(self, q1: Hashable, q2: Hashable) -> float:
        """Coupling constant between the two qudits."""
        return self._coupling[frozenset({q1, q2})]

    def crosstalk(self, source: Hashable, target: Hashable) -> complex:
        """Crosstalk factor from source to target."""
        return self._crosstalk[(source, target)]

    def drive(
        self,
        qudit_id: Optional[Hashable] = None
    ) -> Union[Dict[Hashable, List[DriveTerm]], List[DriveTerm]]:
        """Drive terms for the qudit."""
        if qudit_id is None:
            return copy.deepcopy(self._drive)
        else:
            return list(self._drive[qudit_id])

    def frame(
        self,
        qudit_id: Optional[Hashable] = None
    ) -> Union[Dict[Hashable, Frame], Frame]:
        """Frame of the qudit."""
        if qudit_id is None:
            return copy.deepcopy(self._frame)
        else:
            return self._frame[qudit_id]

    def add_qudit(
        self,
        qubit_frequency: float,
        anharmonicity: float,
        drive_amplitude: float,
        qudit_id: Optional[Hashable] = None,
        position: Optional[int] = None,
        drive_weight: Optional[np.ndarray] = None
    ) -> None:
        r"""Add a qudit to the system.

        Args:
            qubit_frequency: Qubit frequency.
            anharmonicity: Anharmonicity.
            drive_amplitude: Base drive amplitude in rad/s.
            qudit_id: Identifier for the qudit. If None, the position (order of addition) is used.
            position: If an integer, the qudit is inserted into the specified position.
            drive_weight: The weights given to each level transition. Default is :math:`\sqrt{l}`
        """
        if drive_weight is None:
            drive_weight = np.sqrt(np.arange(self.num_levels, dtype=float)[1:])

        params = QuditParams(qubit_frequency=qubit_frequency, anharmonicity=anharmonicity,
                             drive_amplitude=drive_amplitude, drive_weight=drive_weight)

        if qudit_id is None:
            if position is None:
                qudit_id = len(self._qudit_params)
            else:
                qudit_id = position

        if qudit_id in self._qudit_params:
            raise KeyError(f'Qudit id {qudit_id} already exists.')

        if position is None:
            self._qudit_params[qudit_id] = params
        elif position > len(self._qudit_params):
            raise IndexError(f'Position {position} greater than number of existing parameters')
        else:
            qudit_params = dict(list(self._qudit_params.items())[:position])
            qudit_params[qudit_id] = params
            qudit_params.update(list(self._qudit_params.items())[position:])

            self._qudit_params.clear()
            self._qudit_params.update(qudit_params)

        self._drive[qudit_id] = list()
        self._crosstalk[(qudit_id, qudit_id)] = 1.
        self.set_frame(qudit_id)

        self.set_global_frame(self.default_frame, keep_phase=True)

    def identity_op(self) -> qtp.Qobj:
        return qtp.tensor([qtp.qeye(self._num_levels)] * self.num_qudits)

    def eigenvalues(self) -> np.ndarray:
        r"""Compute the energy eigenvalues of the static Hamiltonian (free and coupling).

        Eigenvalues are ordered so that element ``[i, j, ...]`` corresponds to the energy of the state with the
        largest component of :math:`|i\rangle_0 |j\rangle_1 \dots`, where :math:`|l\rangle_k` is the :math:`l`-th
        energy eigenstate of the free qudit :math:`k`.

        Returns:
            An array of energy eigenvalues.
        """
        hgen = self.copy(clear_drive=True)

        # Move to the lab frame first to build a fully static H0+Hint
        hgen.set_global_frame('lab')
        hamiltonian = hgen.build()[0]

        eigvals, unitary = np.linalg.eigh(hamiltonian.full())
        # hamiltonian == unitary @ np.diag(eigvals) @ unitary.T.conjugate()

        # Row index of the biggest contributor to each column
        k = np.argmax(np.abs(unitary), axis=1)
        # Reordered eigenvalues (makes the corresponding change-of-basis unitary closest to identity)
        eigvals = eigvals[k]
        # Reshape the array to the qudit-product structure
        eigvals = eigvals.reshape((self.num_levels,) * self.num_qudits)

        return eigvals

    def free_frequencies(
        self,
        qudit_id: Optional[Hashable] = None
    ) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        """Return the free-qudit frequencies.

        Args:
            qudit_id: Qudit ID. If None, a dict of free frequencies for all qudits is returned.

        Returns:
            The free-qudit frequency of the specified qudit ID, or a full mapping of qudit ID to frequency arrays.
        """
        if qudit_id is None:
            frequencies = dict()
            for qudit_id, params in self._qudit_params.items():
                frequencies[qudit_id] = params.qubit_frequency + np.arange(self._num_levels - 1) * params.anharmonicity

        else:
            params = self._qudit_params[qudit_id]
            frequencies = params.qubit_frequency + np.arange(self._num_levels - 1) * params.anharmonicity

        return frequencies

    def dressed_frequencies(
        self,
        qudit_id: Optional[Hashable] = None
    ) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        """Return the dressed-qudit frequencies.

        Args:
            qudit_id: Qudit ID. If None, a dict of dressed frequencies for all qudits is returned.

        Returns:
            The dressed-qudit frequency of the specified qudit ID, or a full mapping of qudit ID to frequency arrays.
        """
        if len(self._coupling) == 0:
            return self.free_frequencies(qudit_id)

        eigvals = self.eigenvalues()

        frequencies = dict()

        for iq, qid in enumerate(self._qudit_params):
            # [0, ..., 0, :, 0, ..., 0]
            indices = (0,) * iq + (slice(None),) + (0,) * (self.num_qudits - iq - 1)
            energies = eigvals[indices]

            frequencies[qid] = np.diff(energies)

        if qudit_id is None:
            return frequencies
        else:
            return frequencies[qudit_id]

    def noiz_frequencies(
        self,
        qudit_id: Optional[Hashable] = None,
        comp_dim: int = 0
    ) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        r"""Return the no-IZ frequencies.

        No-IZ frame cancels the phase drift of a qudit on average, i.e., when the other qudits are in a fully
        mixed state. The frame defining matrix is

        .. math::

            D_{\mathrm{noIZ}} = \sum_{j=1}^{n} \sum_{l} \frac{1}{L^{n-1}}
                                \mathrm{tr} \left[ \left(I_{\hat{j}} \otimes | l \rangle_j \langle l |_j\right) E \right]
                                I_{\hat{j}} \otimes | l \rangle_j \langle l |_j.

        For ``comp_dim = d != 0``, the trace above is replaced with the pseudo-trace

        .. math::

            \mathrm{ptr}^{j}_{d} (\cdot) = \bigotimes_{k!=j} \sum_{l_k < d} \langle l_k |_k \cdot | l_k \rangle_k.

        Args:
            qudit_id: Qudit ID. If None, a dict of no-IZ frequencies for all qudits is returned.

        Returns:
            The no-IZ frequency of the specified qudit ID, or a full mapping of qudit ID to frequency arrays.
        """
        if len(self._coupling) == 0:
            return self.free_frequencies(qudit_id)

        if comp_dim <= 0:
            comp_dim = self.num_levels

        eigvals = self.eigenvalues()

        frequencies = dict()

        for iq, qid in enumerate(self._qudit_params):
            indexing = (slice(comp_dim),) * (self.num_qudits - 1)
            traceable_form = np.moveaxis(eigvals, iq, -1)[indexing]
            partial_trace = np.sum(traceable_form.reshape(-1, self.num_levels), axis=0)
            partial_trace /= comp_dim ** (self.num_qudits - 1)

            frequencies[qid] = np.diff(partial_trace)

        if qudit_id is None:
            return frequencies
        else:
            return frequencies[qudit_id]

    def set_frame(
        self,
        qudit_id: Hashable,
        frequency: Optional[np.ndarray] = None,
        phase: Optional[Union[np.ndarray, float]] = None
    ) -> None:
        """Set the frame for the qudit.

        The arrays must be of size `num_levels - 1`.

        Args:
            qudit_id: Qudit ID.
            frequency: Frame frequency for each level gap. If None, set to qudit-frame frequencies.
            phase: Frame phase shift, either as a single global value or an array specifying the
                phase shift for each level gap. If None, set to zero.
        """
        if frequency is None:
            frequency = self.free_frequencies(qudit_id)

        if phase is None:
            phase = np.zeros(self._num_levels - 1)
        elif isinstance(phase, float):
            phase = np.full(self._num_levels - 1, phase)

        self._frame[qudit_id] = Frame(frequency=frequency, phase=phase)

    def _frame_arrays(self, frame_spec: FrameSpec) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(frame_spec, str):
            drive_frame = False

            if (frame_spec == 'dressed' or frame_spec.startswith('noiz')) and len(self._coupling) == 0:
                frame_spec = 'qudit'

            elif frame_spec == 'drive':
                frame_spec = self.default_frame
                drive_frame = True

            qudit_ids = self.qudit_ids()
            frequencies = None

            if frame_spec == 'qudit':
                freqs = self.free_frequencies()
                frequencies = np.array([freqs[qudit_id] for qudit_id in qudit_ids])
            elif frame_spec == 'lab':
                frequencies = np.zeros((self.num_qudits, self._num_levels - 1))
            elif frame_spec == 'dressed':
                freqs = self.dressed_frequencies()
                frequencies = np.array([freqs[qudit_id] for qudit_id in qudit_ids])
            elif frame_spec.startswith('noiz'):
                if len(frame_spec) > 4:
                    comp_dim = int(frame_spec[4:])
                else:
                    comp_dim = 0

                freqs = self.noiz_frequencies(comp_dim=comp_dim)
                frequencies = np.array([freqs[qudit_id] for qudit_id in qudit_ids])

            if drive_frame:
                for qid, drives in self._drive.items():
                    if len(drives) == 1:
                        idx = self.qudit_index(qid)
                        frequencies[idx] = np.full(self._num_levels - 1, drives[0].frequency)
                    elif len(drives) != 0:
                        raise RuntimeError(f'Qudit {qid} has more than one drive terms')

            elif frequencies is None:
                raise ValueError(f'Global frame {frame_spec} is not defined')

            phases = np.zeros((self.num_qudits, self._num_levels - 1))

        elif isinstance(frame_spec, dict):
            qudit_ids = self.qudit_ids()
            frequencies = np.array([frame_spec[qudit_id].frequency for qudit_id in qudit_ids])
            phases = np.array([frame_spec[qudit_id].phase for qudit_id in qudit_ids])

        else:
            frequencies = np.array([frame.frequency for frame in frame_spec])
            phases = np.array([frame.phase for frame in frame_spec])

        return frequencies, phases

    def set_global_frame(self, frame_spec: FrameSpec, keep_phase: bool = False) -> None:
        r"""Set frames for all qudits globally.

        The allowed frame names are:

        - 'qudit': Set frame frequencies to the individual qudit level gaps disregarding the couplings.
          Equivalent to calling `set_frame(qid, frequency=None, phase=None)` for all `qid`.
        - 'lab': Set frame frequencies to zero.
        - 'dressed': Diagonalize the static Hamiltonian (in the lab frame) and set the frame frequencies
          to cancel the phase drifts of single-qudit excitations.
        - 'drive': For qudits with a drive term, set the frame frequency for all levels to that of the drive.
          Use the default frame for qudits without a drive. Exception is raised if there are qudits with
          more than one drive terms.

        Args:
            frame_spec: Frame specification in dict or list form, or a frame name ('qudit', 'lab',
            'dressed', or 'drive').
        """
        frequencies, phases = self._frame_arrays(frame_spec)

        qudit_ids = self.qudit_ids()

        if keep_phase:
            current_frame = self.frame()
            phases = np.array([current_frame[qudit_id].phase for qudit_id in qudit_ids])

        for idx, (frequency, phase) in enumerate(zip(frequencies, phases)):
            self.set_frame(qudit_ids[idx], frequency=frequency, phase=phase)

    def frame_change_operator(
        self,
        from_frame: FrameSpec,
        to_frame: Optional[FrameSpec] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the change-of-frame operator.

        Change of frame from

        .. math::

            U_{f}(t) = \bigotimes_j \exp \left[ i \sum_l \left( \Xi_j^l t + \Phi_j^l \right) | l \rangle_j \langle l |_j \right]

        to

        .. math::

            U_{g}(t) = \bigotimes_j \exp \left[ i \sum_l \left( \Eta_j^l t + \Psi_j^l \right) | l \rangle_j \langle l |_j \right]

        is effected by

        .. math::

            V_{gf}(t) = U_g(t) U_f^{\dagger}(t) = \bigotimes_j \exp \left[ i \sum_l \left\{ (\Eta_j^l - \Xi_j^l) t + (\Psi_j^l - \Phi_j^l) \right\} | l \rangle_j \langle l |_j \right].

        Args:
            from_frame: Specification of original frame.
            to_frame: Specification of new frame. If None, the current frame is used.

        Returns:
            Flattened arrays corresponding to :math:`[\sum_{j} \Xi_j^{l_j}]_{\{l_j\}}` and :math:`[\sum_{j} \Phi_j^{l_j}]_{\{l_j\}}`.
        """
        if to_frame is None:
            frequencies = np.array([frame.frequency for frame in self._frame.values()])
            phases = np.array([frame.phase for frame in self._frame.values()])
        else:
            frequencies, phases = self._frame_arrays(to_frame)

        from_freq, from_phase = self._frame_arrays(from_frame)

        frequencies -= from_freq
        phases -= from_phase

        energies = np.concatenate((np.zeros(self.num_qudits)[:, None], np.cumsum(frequencies, axis=1)), axis=1)
        offsets = np.concatenate((np.zeros(self.num_qudits)[:, None], np.cumsum(phases, axis=1)), axis=1)

        en_diagonal = add_outer_multi(energies)
        offset_diagonal = add_outer_multi(offsets)

        return en_diagonal, offset_diagonal

    def change_frame(
        self,
        tlist: np.ndarray,
        obj: Union[np.ndarray, qtp.Qobj],
        from_frame: FrameSpec,
        to_frame: Optional[FrameSpec] = None,
        objtype: str = 'evolution',
        t0: Optional[float] = None
    ) -> np.ndarray:
        r"""Apply the change-of-frame unitaries to states, unitaries, hamiltonians, and observables.

        Args:
            tlist: 1D array of shape ``(T,)`` representing the time points.
            obj: Qobj or an array of shape either ``(D)`` (state), ``(T, D)`` (state evolution), ``(T, D, D)``
                (evolution operator, time-dependent Hamiltonian, or an observable), ``(D, D)``
                (evolution operator for a single time point, Hamiltonian, or an observable), ``(N, D, D)``
                (multiple observables), or ``(N, T, D, D)`` (multiple time-dependent observables), where ``D``
                is the dimension of the quantum system (``num_levels ** num_qudits``) and ``N`` is the number
                of observables.
            from_frame: Specification of original frame.
            to_frame: Specification of new frame. If None, the current frame is used.
            objtype: ``'state'``, ``'evolution'``, ``'hamiltonian'``, or ``'observable'``. Ignored when
                the type can be unambiguously inferred from the shape of ``obj``.
            t0: Initial time for the frame change of the evolution operator. If None, ``tlist[0]`` is used.

        Returns:
            An array representing the frame-changed object.
        """
        if isinstance(obj, qtp.Qobj):
            if obj.isket or obj.isbra:
                obj = np.squeeze(obj.full())
            else:
                obj = obj.full()

        ## Validate and determine the object shape & type
        state_dim = self.num_levels ** self.num_qudits
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
                    # Ambiguity - we may be talking about T observables, but that's rather unprobable
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
            raise ValueError(f'Inconsistent obj shape {obj.shape} for objtype {objtype} and tlist length {tlist.shape[0]}')

        en_diagonal, offset_diagonal = self.frame_change_operator(from_frame, to_frame)
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
            obj = cof_op_diag[:, :, None] * obj * cof_op_diag[:, None, :].conjugate() - np.diag(en_diagonal)

        elif objtype == 'observable':
            obj = cof_op_diag[:, :, None] * obj * cof_op_diag[:, None, :].conjugate()

        return obj.reshape(final_shape)

    def add_coupling(self, q1: Hashable, q2: Hashable, value: float) -> None:
        """Add a coupling term between two qudits."""
        self._coupling[frozenset({q1, q2})] = value

        self.set_global_frame(self.default_frame, keep_phase=True)

    def add_crosstalk(self, source: Hashable, target: Hashable, factor: complex) -> None:
        r"""Add a crosstalk term from the source channel to the target qudit.

        Args:
            source: Qudit ID of the source channel.
            target: Qudit ID of the target qudit.
            factor: Crosstalk coefficient (:math:`\alpha_{jk} e^{i\rho_{jk}}`).
        """
        self._crosstalk[(source, target)] = factor

    def add_drive(
        self,
        qudit_id: Hashable,
        frequency: Optional[float] = None,
        amplitude: Union[float, complex, str, np.ndarray, Callable, None] = 1.+0.j,
        constant_phase: Optional[float] = None,
        sequence: Optional[PulseSequence] = None
    ) -> None:
        r"""Add a drive term.

        Args:
            qudit_id: Qudit to apply the drive to.
            frequency: Carrier frequency of the drive. Required when using ``amplitude`` or when ``sequence`` does not
                start from ``SetFrequency``.
            amplitude: Function :math:`r(t)`. Ignored if ``sequence`` is set.
            constant_phase: The phase value of ``amplitude`` when it is a str or a callable and is known to have a
                constant phase. None otherwise. Ignored if ``sequence`` is set.
            sequence: Pulse sequence of the drive.
        """
        if sequence is not None:
            drive = PulseSequence(sequence)
            if frequency is not None:
                drive.insert(0, SetFrequency(frequency))

        else:
            if frequency is None or amplitude is None:
                raise RuntimeError('Frequency and amplitude must be set if not using a PulseSequence.')

            drive = DriveTerm(frequency=frequency, amplitude=amplitude, constant_phase=constant_phase)

        self._drive[qudit_id].append(drive)

    def clear_drive(self) -> None:
        """Remove all drives."""
        for drives in self._drive.values():
            drives.clear()

    def build(
        self,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Return the list of Hamiltonian terms passable to qutip.sesolve.

        Args:

            tlist: If not None, all callable Hamiltonian coefficients are called with `(tlist, args)` and the
                resulting arrays are instead passed to sesolve.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        hstatic = self.build_hdiag()
        hint = self.build_hint(tlist=tlist)
        hdrive = self.build_hdrive(tlist=tlist, args=args)

        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint.pop(0)
        if hdrive and isinstance(hdrive[0], qtp.Qobj):
            hstatic += hdrive.pop(0)

        hamiltonian = []

        if np.any(hstatic.data.data):
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            hamiltonian.append(hstatic)

        hamiltonian.extend(hint)
        hamiltonian.extend(hdrive)

        return hamiltonian

    def _qudit_hfree(self, params: QuditParams) -> np.ndarray:
        hfree = np.arange(self._num_levels) * (params.qubit_frequency - params.anharmonicity / 2.)
        hfree += np.square(np.arange(self._num_levels)) * params.anharmonicity / 2.
        return hfree

    def build_hdiag(self) -> qtp.Qobj:
        """Build the diagonal term of the Hamiltonian.

        Returns:
            A Qobj representing Hdiag. The object may be empty if the qudit frame is used.
        """
        hdiag = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        for qudit_id, frame in self._frame.items():
            energy_offset = np.cumsum(self.free_frequencies(qudit_id)) - np.cumsum(frame.frequency)
            # If in qudit frame, energy_offset is zero for all levels
            if np.allclose(energy_offset, np.zeros_like(energy_offset)):
                continue

            diagonal = np.concatenate((np.zeros(1), energy_offset))
            qudit_op = qtp.Qobj(inpt=np.diag(diagonal))

            ops = [qtp.qeye(self._num_levels)] * self.num_qudits
            ops[self.qudit_index(qudit_id)] = qudit_op

            hdiag += qtp.tensor(ops)

        return hdiag

    def build_hint(
        self,
        tlist: Optional[np.ndarray] = None
    ) -> List:
        """Build the interaction Hamiltonian.

        Args:
            tlist: Array of time points. If provided and `self.compile_hint` is False, all dynamic coefficients
                are taken as functions and called with `(tlist, None)`, and the resulting arrays are returned.

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        hint = list()
        hstatic = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        for (q1, q2), coupling in self._coupling.items():
            p1 = self._qudit_params[q1]
            p2 = self._qudit_params[q2]

            iq1 = self.qudit_index(q1)
            iq2 = self.qudit_index(q2)

            for l1, l2 in np.ndindex((self._num_levels - 1, self._num_levels - 1)):
                # exp(i (xi_{p1,l1} - xi_{p2,l2}) t) sqrt(l1 + 1) sqrt(l2 + 1) |l1+1>|l2><l1|<l2+1| + h.c.

                # Annihilator terms for this level combination
                ann1 = np.sqrt(l1 + 1) * qtp.basis(self._num_levels, l1) * qtp.basis(self._num_levels, l1 + 1).dag()
                ann2 = np.sqrt(l2 + 1) * qtp.basis(self._num_levels, l2) * qtp.basis(self._num_levels, l2 + 1).dag()

                ops = [qtp.qeye(self._num_levels)] * self.num_qudits
                ops[iq1] = ann1.dag()
                ops[iq2] = ann2

                op = qtp.tensor(ops)
                frequency = 0.

                for qudit_id, level, sign in [(q1, l1, 1.), (q2, l2, -1.)]:
                    frame = self._frame[qudit_id]

                    if frame.phase[level] != 0.:
                        op *= np.exp(sign * 1.j * frame.phase[level])

                    frequency += sign * frame.frequency[level]

                if np.isclose(frequency, 0.):
                    hstatic += coupling * (op + op.dag())

                else:
                    h_x = coupling * (op + op.dag())
                    h_y = coupling * 1.j * (op - op.dag())

                    if self.compile_hint:
                        hint.append([h_x, f'cos({frequency}*t)'])
                        hint.append([h_y, f'sin({frequency}*t)'])
                    else:
                        fn_x = cos_freq(frequency)
                        fn_y = sin_freq(frequency)
                        if tlist is None:
                            hint.append([h_x, fn_x])
                            hint.append([h_y, fn_y])
                        else:
                            hint.append([h_x, fn_x(tlist, None)])
                            hint.append([h_y, fn_y(tlist, None)])

        if np.any(hstatic.data.data):
            hint.insert(0, hstatic)

        return hint

    def build_hdrive(
        self,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Build the drive Hamiltonian.

        Args:
            tlist: Array of time points. Required when at least one drive amplitude is given as an array.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        hdrive = list()
        hstatic = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        # Construct the Qobj for each qudit/level first
        qops = list()

        for iq, qudit_id in enumerate(self._frame):
            frame = self._frame[qudit_id]

            for level in range(self._num_levels - 1):
                cre = np.sqrt(level + 1) * qtp.basis(self._num_levels, level + 1) * qtp.basis(self._num_levels, level).dag()

                ops = [qtp.qeye(self._num_levels)] * self.num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)

                if frame.phase[level] != 0.:
                    op *= np.exp(1.j * frame.phase[level])

                qops.append((qudit_id, op, frame.frequency[level]))

        # Loop over the drive channels
        for channel, drives in self._drive.items():
            ch_params = self._qudit_params[channel]

            for drive in drives:
                if not isinstance(drive, PulseSequence) and isinstance(drive.amplitude, np.ndarray) and tlist is None:
                    raise RuntimeError('Time points array is needed to build a Hamiltonian with array-based drive.')

                # Loop over the driven operators
                for qudit_id, creation_op, frame_frequency in qops:
                    drive_base = ch_params.drive_amplitude / 2.

                    try:
                        # Crosstalk between a qudit and itself is set to 1 in add_qudit
                        drive_base *= self._crosstalk[(channel, qudit_id)]
                    except KeyError:
                        # This qudit gets no drive
                        continue

                    # Hamiltonian term can be split in xy (static and/or real envelope) or creation/annihilation (otherwise)
                    fn_x, fn_y = drive.generate_fn(frame_frequency, drive_base, self.use_rwa)

                    h_x = creation_op + creation_op.dag()
                    h_y = 1.j * (creation_op - creation_op.dag())

                    if isinstance(fn_x, (float, complex)):
                        hstatic += fn_x * h_x
                        hstatic += fn_y * h_y

                    elif isinstance(fn_x, str):
                        hdrive.append([h_x, fn_x])
                        if fn_y:
                            hdrive.append([h_y, fn_y])

                    elif isinstance(fn_x, np.ndarray):
                        if np.any(fn_x):
                            hdrive.append([h_x, fn_x])
                        if np.any(fn_y):
                            hdrive.append([h_y, fn_y])

                    else:
                        if tlist is None:
                            hdrive.append([h_x, fn_x])
                            if fn_y is not None:
                                hdrive.append([h_y, fn_y])
                        else:
                            hdrive.append([h_x, fn_x(tlist, args)])
                            if fn_y is not None:
                                hdrive.append([h_y, fn_y(tlist, args)])

        if np.any(hstatic.data.data):
            hdrive.insert(0, hstatic)

        return hdrive

    def make_tlist(
        self,
        points_per_cycle: int,
        num_cycles: Optional[int] = None,
        duration: Optional[float] = None,
        num_points: Optional[int] = None,
        frame: Optional[FrameSpec] = None
    ) -> np.ndarray:
        r"""Build a list of time points using the maximum frequency in the Hamiltonian.

        If the Hamiltonian is static, uses the maximum level spacing.

        Use one of ``num_cycles``, ``duration``, or ``num_points`` to specify the total number of time points.

        Args:
            points_per_cycle: Number of points per cycle at the highest frequency.
            num_cycles: Number of overall cycles.
            duration: Maximum value of the tlist.
            num_points: Total number of time points including 0.
            frame: If specified, the frame in which to find the maximum frequency.

        Returns:
            Array of time points.
        """
        if len([p for p in [num_cycles, duration, num_points] if p is not None]) != 1:
            raise RuntimeError('One and only one of num_cycles, duration, or num_points must be set')

        max_frequency = 0.

        if frame is not None:
            hgen = self.copy()
            hgen.set_global_frame(frame)
        else:
            hgen = self

        for q1, q2 in hgen._coupling.keys():
            max_freq_diffs = np.amax(np.abs(np.subtract.outer(hgen._frame[q1].frequency, hgen._frame[q2].frequency)))
            max_frequency = max(max_frequency, max_freq_diffs)

        for qid, drives in hgen._drive.items():
            if len(drives) == 0:
                continue

            drive_freqs = np.array(list(drive.frequency for drive in drives))
            if self.use_rwa:
                rel_sign = -1
            else:
                rel_sign = 1

            max_frame_drive_freqs = np.amax(np.abs(np.add.outer(rel_sign * drive_freqs, hgen._frame[qid].frequency)))
            max_frequency = max(max_frequency, max_frame_drive_freqs)

        if max_frequency == 0.:
            eigvals = self.eigenvalues()

            for axis in range(self.num_qudits):
                level_gaps = np.diff(eigvals, axis=axis)
                max_frequency = max(max_frequency, np.amax(level_gaps))

        cycle = 2. * np.pi / max_frequency

        if num_cycles is not None:
            duration = cycle * num_cycles
            num_points = points_per_cycle * num_cycles + 1

        elif duration is not None:
            num_points = np.round(points_per_cycle * duration / cycle).astype(int) + 1

        elif num_points is not None:
            duration = cycle * (num_points - 1) / points_per_cycle

        return np.linspace(0., duration, num_points)

    def copy(self, clear_drive: bool = False):
        instance = HamiltonianBuilder(self._num_levels)

        instance.default_frame = self.default_frame

        instance.compile_hint = self.compile_hint
        instance.use_rwa = self.use_rwa

        # Not sure if copy.deepcopy keeps the dict ordering, so we insert by hand
        instance._qudit_params.update(self._qudit_params.items())
        instance._coupling.update(self._coupling.items())
        instance._crosstalk.update(self._crosstalk.items())
        for qudit_id in self._qudit_params:
            instance._drive[qudit_id] = list()
        if not clear_drive:
            for qudit_id, drive in self._drive.items():
                instance._drive[qudit_id].extend(drive)

        instance._frame.update(self._frame.items())

        return instance

    def make_scan(
        self,
        scan_type: str,
        values: Sequence,
        **kwargs
    ) -> List['HamiltonianBuilder']:
        """Build a list of copies of self varied over a single attribute.

        The argument ``scan_type`` determines which attribute to make variations over. Implemented scan types are
        - ``'amplitude'``: Drive amplitudes. Elements of ``values`` are passed to the ``amplitude`` parameter of ``add_drive``.
        - ``'frequency'``: Drive frequencies. Elements of ``values`` are passed to the ``frequency`` parameter of ``add_drive``.
        - ``'coupling'``: Qudit-qudit couplings. Elements of ``values`` are passed to the ``value`` parameter of ``add_coupling``.

        In all cases, the remaining arguments to the respective functions must be given in the ``kwargs`` of this method.

        Args:
            scan_type: ``'amplitude'``, ``'frequency'``, or ``'coupling'``.
            values: A list of values. One copy of self is created for each value.
            kwargs: Remaining arguments to the function to be called on each copy.

        Returns:
            A list of copies of self varied over the specified attribute.
        """
        copies = list(self.copy() for _ in range(len(values)))

        if scan_type == 'amplitude':
            for value, instance in zip(values, copies):
                instance.add_drive(qudit_id=kwargs['qudit_id'],
                                   frequency=kwargs.get('frequency'),
                                   amplitude=value,
                                   constant_phase=kwargs.get('constant_phase'))

        elif scan_type == 'frequency':
            for value, instance in zip(values, copies):
                instance.add_drive(qudit_id=kwargs['qudit_id'],
                                   frequency=value,
                                   amplitude=kwargs.get('amplitude', 1.+0.j),
                                   constant_phase=kwargs.get('constant_phase'))

        elif scan_type == 'coupling':
            for value, instance in zip(values, copies):
                instance.add_coupling(q1=kwargs['q1'], q2=kwargs['q2'], value=value)

        return copies
