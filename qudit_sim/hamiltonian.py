r"""
====================================================
Hamiltonian builder (:mod:`qudit_sim.hamiltonian`)
====================================================

.. currentmodule:: qudit_sim.hamiltonian

See :doc:`/hamiltonian` for theoretical background.
"""

# TODO: Qubit optimization

# When only considering qubits, anharmonicity terms are dropped, making the formulae above somewhat simpler. In particular,
# when one or more drives are resonant with qubit frequencies, some terms in the interaction and drive Hamiltonians
# will have common frequencies (i.e. :math:`\delta_{jk}` and `\epsilon_{kj}` coincide). We should look into how to detect
# and exploit such cases in the future.

from typing import Any, Dict, Sequence, List, Tuple, Callable, Optional, Union, Hashable
import copy
from dataclasses import dataclass
import numpy as np
import qutip as qtp

from .pulse import PulseSequence
from .drive import DriveTerm, cos_freq, sin_freq
from .util import Frame

FrameSpec = Union[str, Dict[Hashable, Frame], Sequence[Frame]]

@dataclass(frozen=True)
class QuditParams:
    """Parameters defining a qudit."""
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float

class MaxFrequencyNotSet(Exception):
    pass

class HamiltonianBuilder:
    r"""Hamiltonian with static transverse couplings and external drive terms.

    Args:
        num_levels: Number of energy levels to consider.
        qudits: If passing ``params`` to initialize the Hamiltonian, list of qudit numbers to include.
        params: Hamiltonian parameters given by IBMQ ``backend.configuration().hamiltonian['vars']``, optionally
            augmented with ``'crosstalk'``, which should be a ``dict`` of form ``{(j, k): z}`` specifying the crosstalk
            factor ``z`` (complex corresponding to :math:`\alpha_{jk} e^{i\rho_{jk}}`) of drive on qudit :math:`j` seen
            by qudit :math:`k`. :math:`j` and :math:`k` are qudit ids given in ``qudits``.
        default_frame: Default global frame to use. ``set_global_frame(default_frame)`` is executed each time
            ``add_qudit`` or ``add_coupling`` is called.
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

        self._max_frequency_int = None
        self._max_frequency_drive = None

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

    @property
    def max_frequency(self) -> float:
        """Maximum frequency appearing in this Hamiltonian."""
        if self._max_frequency_int is None or self._max_frequency_drive is None:
            raise MaxFrequencyNotSet('Call build_hint and build_hdrive first')

        return max(self._max_frequency_int, self._max_frequency_drive)

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

    def drive(self, qudit_id: Hashable) -> List[DriveTerm]:
        """Drive terms for the qudit."""
        return list(self._drive[qudit_id])

    def frame(self, qudit_id: Hashable) -> Frame:
        """Frame of the qudit."""
        return self._frame[qudit_id]

    def add_qudit(
        self,
        qubit_frequency: float,
        anharmonicity: float,
        drive_amplitude: float,
        qudit_id: Optional[Hashable] = None,
        position: Optional[int] = None
    ) -> None:
        """Add a qudit to the system.

        Args:
            qubit_frequency: Qubit frequency.
            anharmonicity: Anharmonicity.
            drive_amplitude: Base drive amplitude in rad/s.
            qudit_id: Identifier for the qudit. If None, the position (order of addition) is used.
            position: If an integer, the qudit is inserted into the specified position.
        """
        params = QuditParams(qubit_frequency=qubit_frequency, anharmonicity=anharmonicity,
                             drive_amplitude=drive_amplitude)

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

        self.set_global_frame(self.default_frame)

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
        current_frame = dict(self._frame)

        # Move to the lab frame first to build a fully static H0+Hint
        self.set_global_frame('lab')

        hamiltonian = self.build_hdiag() + self.build_hint()[0]
        eigvals, unitary = np.linalg.eigh(hamiltonian.full())
        # hamiltonian == unitary @ np.diag(eigvals) @ unitary.T.conjugate()

        # Row index of the biggest contributor to each column
        k = np.argmax(np.abs(unitary), axis=1)
        # Reordered eigenvalues (makes the corresponding change-of-basis unitary closest to identity)
        eigvals = eigvals[k]
        # Reshape the array to extract the single-qudit excitation subspaces
        eigvals = eigvals.reshape((self.num_levels,) * self.num_qudits)

        frequencies = dict()

        for iq, qid in enumerate(self._qudit_params):
            # [0, ..., 0, :, 0, ..., 0]
            indices = (0,) * iq + (slice(None),) + (0,) * (self.num_qudits - iq - 1)
            energies = eigvals[indices]

            frequencies[qid] = np.diff(energies)

        self._frame = current_frame

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

        self._max_frequency_int = None
        self._max_frequency_drive = None

    def set_global_frame(self, frame_spec: FrameSpec) -> None:
        r"""Set frames for all qudits globally.

        The allowed frame names are:

        - 'qudit': Set frame frequencies to the individual qudit level gaps disregarding the couplings.
          Equivalent to calling `set_frame(qid, frequency=None, phase=None)` for all `qid`.
        - 'lab': Set frame frequencies to zero.
        - 'dressed': Diagonalize the static Hamiltonian (in the lab frame) and set the frame frequencies
          to cancel the phase drifts of single-qudit excitations.

        Args:
            frame_spec: Frame specification in dict or list form, or a frame name ('qudit', 'lab', or 'dressed').
        """
        if isinstance(frame_spec, str):
            if frame_spec == 'dressed' and len(self._coupling) == 0:
                frame_spec = 'qudit'

            if frame_spec == 'qudit':
                for qid in self._qudit_params:
                    self.set_frame(qid)

            elif frame_spec == 'lab':
                for qid in self._qudit_params:
                    self.set_frame(qid, frequency=np.zeros(self._num_levels - 1))

            elif frame_spec == 'dressed':
                frequencies = self.dressed_frequencies()

                for qid, freq in frequencies.items():
                    self.set_frame(qid, frequency=freq)

            else:
                raise ValueError(f'Global frame {frame_spec} is not defined')

        elif isinstance(frame_spec, dict):
            for qudit_id, frame in frame_spec.items():
                self.set_frame(qudit_id, frequency=frame.frequency, phase=frame.phase)

        else:
            qudit_ids = self.qudit_ids()
            for idx, frame in enumerate(frame_spec):
                self.set_frame(qudit_ids[idx], frequency=frame.frequency, phase=frame.phase)

    def change_frame(
        self,
        tlist: np.ndarray,
        states: np.ndarray,
        from_frame: FrameSpec,
        to_frame: Optional[FrameSpec] = None
    ) -> np.ndarray:
        """Apply the change-of-frame unitaries to states.

        Args:
            tlist: 1D array of time points.
            states: Array of shape either ``(T, D, D)`` (operator evolution) or ``(T, D)`` (state evolution).
            from_frame: Specification of original frame.
            to_frame: Specification of new frame. If None, the current frame is used.

        Returns:
            An array corresponding to ``states`` with the change of frame applied.
        """
        def frame_arrays(frame_spec):
            if isinstance(frame_spec, str):
                if frame_spec == 'dressed' and len(self._coupling) == 0:
                    frame_spec = 'qudit'

                qudit_ids = self.qudit_ids()

                if frame_spec == 'qudit':
                    freqs = self.free_frequencies()
                    frequencies = np.array([freqs[qudit_id] for qudit_id in qudit_ids])
                elif frame_spec == 'lab':
                    frequencies = np.zeros((self.num_qudits, self._num_levels - 1))
                elif frame_spec == 'dressed':
                    freqs = self.dressed_frequencies()
                    frequencies = np.array([freqs[qudit_id] for qudit_id in qudit_ids])
                else:
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

        if to_frame is None:
            frequencies = np.array([frame.frequency for frame in self._frame.values()])
            phases = np.array([frame.phase for frame in self._frame.values()])
        else:
            frequencies, phases = frame_arrays(to_frame)

        from_freq, from_phase = frame_arrays(from_frame)

        frequencies -= from_freq
        phases -= from_phase

        energies = np.concatenate((np.zeros(self.num_qudits)[:, None], np.cumsum(frequencies, axis=1)), axis=1)
        offsets = np.concatenate((np.zeros(self.num_qudits)[:, None], np.cumsum(phases, axis=1)), axis=1)

        en_diagonal = np.zeros(1)
        for en_arr in energies:
            en_diagonal = np.add.outer(en_diagonal, en_arr).reshape(-1)

        offset_diagonal = np.zeros(1)
        for off_arr in offsets:
            offset_diagonal = np.add.outer(offset_diagonal, off_arr).reshape(-1)

        diagonals = np.exp(1.j * (en_diagonal[None, :] * tlist[:, None] + offset_diagonal))

        if len(states.shape) == 2:
            # Left-multiplying by a diagonal is the same as element-wise multiplication
            return diagonals * states

        elif len(states.shape) == 3:
            # Right-multiplying by the inverse of a diagonal unitary = element-wise multiplication
            # of the columns by the conjugate
            return diagonals[:, :, None] * states * diagonals[0].conjugate()

        else:
            raise ValueError('Invalid shape of states array')


    def add_coupling(self, q1: Hashable, q2: Hashable, value: float) -> None:
        """Add a coupling term between two qudits."""
        self._coupling[frozenset({q1, q2})] = value

        self.set_global_frame(self.default_frame)

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
        amplitude: Union[float, complex, str, np.ndarray, PulseSequence, Callable, None] = 1.+0.j,
        constant_phase: Optional[float] = None
    ) -> None:
        r"""Add a drive term.

        Args:
            qudit_id: Qudit to apply the drive to.
            frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
                that starts with SetFrequency.
            amplitude: Function `r(t)`.
            constant_phase: The phase value of `amplitude` when it is a str or a callable and is known
                to have a constant phase. None otherwise.
        """
        drive = DriveTerm(frequency=frequency, amplitude=amplitude, constant_phase=constant_phase)
        self._drive[qudit_id].append(drive)

    def clear_drive(self) -> None:
        """Remove all drives."""
        for drives in self._drive.values():
            drives.clear()

        self._max_frequency_drive = None

    def build(
        self,
        rwa: bool = True,
        compile_hint: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Return the list of Hamiltonian terms passable to qutip.sesolve.

        Args:
            rwa: If True, apply the rotating-wave approximation.
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
            tlist: If not None, all callable Hamiltonian coefficients are called with `(tlist, args)` and the
                resulting arrays are instead passed to sesolve.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        hstatic = self.build_hdiag()
        hint = self.build_hint(compile_hint=compile_hint, tlist=tlist, args=args)
        hdrive = self.build_hdrive(rwa=rwa, tlist=tlist, args=args)

        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint.pop(0)
        if hdrive and isinstance(hdrive[0], qtp.Qobj):
            hstatic += hdrive.pop(0)

        hamiltonian = []

        if hstatic != qtp.Qobj():
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
        hdiag = qtp.Qobj()

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
        compile_hint: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Build the interaction Hamiltonian.

        Args:
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
            tlist: Array of time points. If provided and `compile_hint=False`, all callable coefficients
                are called with `(tlist, args)` and the resulting arrays are returned.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        self._max_frequency_int = 0.

        hint = list()
        hstatic = qtp.Qobj()

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

                    if compile_hint:
                        hint.append([h_x, f'cos({frequency}*t)'])
                        hint.append([h_y, f'sin({frequency}*t)'])
                    else:
                        fn_x = cos_freq(frequency)
                        fn_y = sin_freq(frequency)
                        if tlist is None:
                            hint.append([h_x, fn_x])
                            hint.append([h_y, fn_y])
                        else:
                            hint.append([h_x, fn_x(tlist, args)])
                            hint.append([h_y, fn_y(tlist, args)])

                    self._max_frequency_int = max(self._max_frequency_int, abs(frequency))

        if hstatic != qtp.Qobj():
            hint.insert(0, hstatic)

        return hint

    def build_hdrive(
        self,
        rwa: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Build the drive Hamiltonian.

        Args:
            rwa: If True, apply the rotating-wave approximation.
            tlist: Array of time points. Required when at least one drive amplitude is given as an array.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        self._max_frequency_drive = 0.

        hdrive = list()
        hstatic = qtp.Qobj()

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
                if isinstance(drive.amplitude, np.ndarray) and tlist is None:
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
                    fn_x, fn_y, term_max_frequency = drive.generate_fn(frame_frequency, drive_base, rwa)

                    h_x = creation_op + creation_op.dag()
                    h_y = 1.j * (creation_op - creation_op.dag())

                    if isinstance(fn_x, (float, complex)):
                        if fn_x != 0.:
                            hstatic += fn_x * h_x
                        if fn_y != 0.:
                            hstatic += fn_y * h_y

                    elif isinstance(fn_x, str):
                        hdrive.append([h_x, fn_x])
                        if fn_y:
                            hdrive.append([h_y, fn_y])

                    elif isinstance(fn_x, np.ndarray):
                        if np.any(fn_x != 0.):
                            hdrive.append([h_x, fn_x])
                        if np.any(fn_y != 0.):
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

                    self._max_frequency_drive = max(self._max_frequency_drive, term_max_frequency)

        if hstatic != qtp.Qobj():
            hdrive.insert(0, hstatic)

        return hdrive

    def make_tlist(
        self,
        points_per_cycle: int,
        num_cycles: Optional[int] = None,
        duration: Optional[float] = None,
        num_points: Optional[int] = None,
        rwa: bool = True
    ) -> np.ndarray:
        r"""Build a list of time points using the maximum frequency in the Hamiltonian.

        If the maximum frequency is not determined yet, ``self.build(rwa)`` is called first to compute one.

        When the maximum frequency is 0 (i.e. single-qubit simulation with resonant drive), return the time
        range from 0 to :math:`2 \pi / \sqrt{\mathrm{tr}(H_{\mathrm{stat}} H_{\mathrm{stat}}) / 2^{n}}.

        Use one of ``num_cycles``, ``duration``, or `num_points` to specify the total number of time points.

        Args:
            points_per_cycle: Number of points per cycle at the highest frequency.
            num_cycles: Number of overall cycles.
            duration: Maximum value of the tlist.
            num_points: Total number of time points.
            rwa: Whether to use the rotating wave approximation to find the maximum frequency.

        Returns:
            Array of time points.
        """
        if len([p for p in [num_cycles, duration, num_points] if p is not None]) != 1:
            raise RuntimeError('One and only one of num_cycles, duration, or num_points must be set')

        try:
            frequency = self.max_frequency

        except MaxFrequencyNotSet:
            self.build_hint()
            self.build_hdrive(rwa=rwa)
            frequency = self.max_frequency

        if frequency == 0.:
            hamiltonian = self.build(rwa=rwa)
            if not hamiltonian:
                raise RuntimeError('Cannot determine the tlist')

            hstat = hamiltonian[0]
            amp2 = np.trace(hstat.full() @ hstat.full()).real / (2 ** self.num_qudits)
            frequency = np.sqrt(amp2)

        cycle = 2. * np.pi / frequency

        if num_cycles is not None:
            duration = cycle * num_cycles
            num_points = points_per_cycle * num_cycles

        elif duration is not None:
            num_points = int(points_per_cycle * duration / cycle)

        elif num_points is not None:
            duration = cycle * num_points / points_per_cycle

        return np.linspace(0., duration, num_points)


    def make_scan(
        self,
        scan_type: str,
        values: Sequence,
        **kwargs
    ) -> List['HamiltonianBuilder']:
        """Build a list of copies of self varied over a single attribute.

        The argument `scan_type` determines which attribute to make variations over. Implemented scan types are
        - `'amplitude'`: Drive amplitudes. Elements of `values` are passed to the `amplitude` parameter of `add_drive`.
        - `'frequency'`: Drive frequencies. Elements of `values` are passed to the `frequency` parameter of `add_drive`.
        - `'coupling'`: Qudit-qudit couplings. Elements of `values` are passed to the `value` parameter of `add_coupling`.

        In all cases, the remaining arguments to the respective functions must be given in the `kwargs` of this method.

        Args:
            scan_type: `'amplitude'`, `'frequency'`, or `'coupling'`.
            values: A list of values. One copy of self is created for each value.
            kwargs: Remaining arguments to the function to be called on each copy.

        Returns:
            A list of copies of self varied over the specified attribute.
        """
        copies = list(copy.deepcopy(self) for _ in range(len(values)))

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
