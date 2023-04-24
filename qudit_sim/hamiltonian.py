r"""
====================================================
Hamiltonian builder (:mod:`qudit_sim.hamiltonian`)
====================================================

.. currentmodule:: qudit_sim.hamiltonian

See :doc:`/hamiltonian` for theoretical background.
"""

from typing import Any, Dict, Sequence, List, Callable, Optional, Union, Hashable
from types import ModuleType
from numbers import Number
from dataclasses import dataclass
import copy
import numpy as np
import qutip as qtp

from .expression import ParameterExpression, TimeFunction, ConstantFunction
from .drive import DriveTerm, SetFrequency, HamiltonianCoefficient, CosFunction, SinFunction
from .frame import FrameSpec, SystemFrame

QobjCoeffPair = List[Union[qtp.Qobj, HamiltonianCoefficient]]

@dataclass(frozen=True)
class QuditParams:
    """Parameters defining a qudit."""
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float
    drive_weight: np.ndarray


class Hamiltonian(list):
    """Hamiltonian list with function evaluation subroutine."""
    def evaluate_coeffs(
        self,
        tlist: np.ndarray,
        args: Optional[Dict[str, Any]] = None,
        npmod: ModuleType = np
    ) -> List[Union[qtp.Qobj, QobjCoeffPair]]:
        """
        Evaluate functional Hamiltonian coefficients at given time points.

        Args:
            tlist: If not None, all callable Hamiltonian coefficients are called with `(tlist, args)` and the
                resulting arrays are instead passed to sesolve.
            args: Arguments to the callable coefficients.

        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        evaluated = list()
        hstatic = qtp.Qobj()
        for term in self:
            if isinstance(term, list):
                if isinstance(term[1], ConstantFunction):
                    hstatic += term[0] * term[1](0., args, npmod)
                elif isinstance(term[1], TimeFunction):
                    evaluated.append([term[0], term[1](tlist, args, npmod)])
                else:
                    evaluated.append(list(term))
            else:
                hstatic += term

        evaluated.insert(0, hstatic)

        return evaluated


class HamiltonianBuilder:
    r"""Hamiltonian with static transverse couplings and external drive terms.

    The class has one option flag that alter the Hamiltonian-building behavior:

    - use_rwa: If True, apply the rotating-wave approximation to the drive Hamiltonian.

    Args:
        num_levels: Number of energy levels to consider.
        qudits: If passing ``params`` to initialize the Hamiltonian, list of qudit numbers to include.
        params: Hamiltonian parameters given by IBMQ ``backend.configuration().hamiltonian['vars']``, optionally
            augmented with ``'crosstalk'``, which should be a ``dict`` of form ``{(j, k): z}`` specifying the crosstalk
            factor ``z`` (complex corresponding to :math:`\alpha_{jk} e^{i\rho_{jk}}`) of drive on qudit :math:`j` seen
            by qudit :math:`k`. :math:`j` and :math:`k` are qudit ids given in ``qudits``.
    """
    def __init__(
        self,
        num_levels: int = 2,
        qudits: Optional[Union[int, Sequence[int]]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        self._num_levels = num_levels

        # Makes use of dict order guarantee from python 3.7
        self._qudit_params = dict()
        self._coupling = dict()
        self._crosstalk = dict()
        self._drive = dict()

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
        return self._coupling.get(frozenset({q1, q2}), 0.)

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
            drive_weight: The weights given to each level transition. Default is :math:`\sqrt{l + 1}`
        """
        if drive_weight is None:
            drive_weight = np.sqrt(np.arange(1, self.num_levels, dtype=float))

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
        hstatic = self.build_hdiag(frame='lab') + self.build_hint(frame='lab')[0]

        eigvals, unitary = np.linalg.eigh(hstatic.full())
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

    def add_coupling(self, q1: Hashable, q2: Hashable, value: float) -> None:
        """Add a coupling term between two qudits."""
        self._coupling[frozenset({q1, q2})] = value

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
        sequence: Optional[List[Any]] = None
    ) -> None:
        r"""Add a drive term.

        Args:
            qudit_id: Qudit to apply the drive to.
            frequency: Carrier frequency of the drive. Required when using ``amplitude`` or when ``sequence`` does not
                start from ``SetFrequency``.
            amplitude: Function :math:`r(t)`. Ignored if ``sequence`` is set.
            sequence: Pulse sequence of the drive.
        """
        self._drive[qudit_id].append(
            DriveTerm(frequency=frequency, amplitude=amplitude, sequence=sequence)
        )

    def clear_drive(self) -> None:
        """Remove all drives."""
        for drives in self._drive.values():
            drives.clear()

    def build(
        self,
        frame: FrameSpec = 'dressed',
        as_timefn: bool = False
    ) -> Hamiltonian:
        """Return the list of Hamiltonian terms passable to qutip.sesolve.

        Args:
            frame: System frame to build the Hamiltonian in.
            as_timefn: If True, force the coefficients to be TimeFunctions.

        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        hstatic = self.build_hdiag(frame=frame)
        hint = self.build_hint(frame=frame, as_timefn=as_timefn)
        hdrive = self.build_hdrive(frame=frame, as_timefn=as_timefn)

        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint.pop(0)
        if hdrive and isinstance(hdrive[0], qtp.Qobj):
            hstatic += hdrive.pop(0)

        hamiltonian = Hamiltonian()

        if np.any(hstatic.data.data):
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            hamiltonian.append(hstatic)

        hamiltonian.extend(hint)
        hamiltonian.extend(hdrive)

        return hamiltonian

    def build_hstatic(
        self,
        frame: FrameSpec = 'dressed',
    ) -> qtp.Qobj:
        """Build only the static Hamiltonian term in the given frame.

        Args:
            frame: System frame to build the Hamiltonian in.

        Returns:
            A Qobj representing Hstat.
        """
        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        hstatic = self.build_hdiag(frame=frame)
        hint = self.build_hint(frame=frame)

        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint[0]

        return hstatic

    def build_hdiag(
        self,
        frame: FrameSpec = 'dressed'
    ) -> qtp.Qobj:
        """Build the diagonal term of the Hamiltonian.

        Args:
            frame: System frame to build the Hamiltonian in.

        Returns:
            A Qobj representing Hdiag.
        """
        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        hdiag = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        for qudit_id in self.qudit_ids():
            energy_offset = np.cumsum(self.free_frequencies(qudit_id))
            energy_offset -= np.cumsum(frame[qudit_id].frequency)

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
        frame: FrameSpec = 'dressed',
        as_timefn: bool = False
    ) -> Hamiltonian:
        """Build the interaction Hamiltonian.

        Args:
            frame: System frame to build the Hamiltonian in.
            as_timefn: Return the coefficients as TimeFunctions (True) or string expressions (False).

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        hint = Hamiltonian()
        hstatic = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        for (q1, q2), coupling in self._coupling.items():
            p1 = self._qudit_params[q1]
            p2 = self._qudit_params[q2]

            iq1 = self.qudit_index(q1)
            iq2 = self.qudit_index(q2)

            for l1, l2 in np.ndindex((self._num_levels - 1, self._num_levels - 1)):
                # exp(i (xi_{p1,l1} - xi_{p2,l2}) t) lambda_l1 lambda_l2 |l1+1>|l2><l1|<l2+1| + h.c.

                # Annihilator terms for this level combination
                ann1 = p1.drive_weight[l1] * qtp.basis(self._num_levels, l1) * qtp.basis(self._num_levels, l1 + 1).dag()
                ann2 = p2.drive_weight[l2] * qtp.basis(self._num_levels, l2) * qtp.basis(self._num_levels, l2 + 1).dag()

                ops = [qtp.qeye(self._num_levels)] * self.num_qudits
                ops[iq1] = ann1.dag()
                ops[iq2] = ann2

                op = qtp.tensor(ops)
                frequency = 0.

                for qudit_id, level, sign in [(q1, l1, 1.), (q2, l2, -1.)]:
                    op *= np.exp(sign * 1.j * frame[qudit_id].phase[level])
                    frequency += sign * frame[qudit_id].frequency[level]

                if np.isclose(frequency, 0.):
                    hstatic += coupling * (op + op.dag())

                else:
                    h_x = coupling * (op + op.dag())
                    h_y = coupling * 1.j * (op - op.dag())

                    if as_timefn:
                        hint.append([h_x, CosFunction(frequency)])
                        hint.append([h_y, SinFunction(frequency)])
                    else:
                        hint.append([h_x, f'cos({frequency}*t)'])
                        hint.append([h_y, f'sin({frequency}*t)'])

        if np.any(hstatic.data.data):
            hint.insert(0, hstatic)

        return hint

    def build_hdrive(
        self,
        frame: FrameSpec = 'dressed',
        as_timefn: bool = False
    ) -> Hamiltonian:
        """Build the drive Hamiltonian.

        Args:
            frame: System frame to build the Hamiltonian in.
            as_timefn: If True, force the coefficients to be TimeFunctions. Raises a TypeError for a drive with
                an incompatible type is

        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        hdrive = Hamiltonian()
        hstatic = qtp.tensor([qtp.qzero(self._num_levels)] * self.num_qudits)

        # Construct the Qobj for each qudit/level first
        qops = list()

        for iq, (qudit_id, params) in enumerate(self._qudit_params.items()):
            for level in range(self._num_levels - 1):
                cre = qtp.basis(self._num_levels, level + 1) * qtp.basis(self._num_levels, level).dag()
                cre *= params.drive_weight[level]

                ops = [qtp.qeye(self._num_levels)] * self.num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)

                op *= np.exp(1.j * frame[qudit_id].phase[level])

                qops.append((qudit_id, op, frame[qudit_id].frequency[level]))

        # Loop over the drive channels
        for channel, drives in self._drive.items():
            ch_params = self._qudit_params[channel]

            for drive in drives:
                # Loop over the driven operators
                for qudit_id, creation_op, frame_frequency in qops:
                    drive_base = ch_params.drive_amplitude / 2.

                    try:
                        # Crosstalk between a qudit and itself is set to 1 in add_qudit
                        drive_base *= self._crosstalk[(channel, qudit_id)]
                    except KeyError:
                        # This qudit gets no drive
                        continue

                    # Generate the potentially time-dependent Hamiltonian coefficients
                    fn_x, fn_y = drive.generate_fn(frame_frequency, drive_base, self.use_rwa, as_timefn=as_timefn)

                    h_x = creation_op + creation_op.dag()
                    h_y = 1.j * (creation_op - creation_op.dag())

                    if isinstance(fn_x, Number):
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
                        # TimeFunction
                        if not (isinstance(fn_x, ConstantFunction) and fn_x.value == 0.):
                            hdrive.append([h_x, fn_x])
                        if not (isinstance(fn_y, ConstantFunction) and fn_y.value == 0.):
                            hdrive.append([h_y, fn_y])

        if np.any(hstatic.data.data):
            hdrive.insert(0, hstatic)

        return hdrive

    def make_tlist(
        self,
        points_per_cycle: int,
        num_cycles: Optional[int] = None,
        duration: Optional[float] = None,
        num_points: Optional[int] = None,
        frame: FrameSpec = 'dressed',
        freq_args: Dict[str, Any] = {}
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
            freq_args: Arguments to pass to parametric drive frequencies (if there are any).

        Returns:
            Array of time points.
        """
        if len([p for p in [num_cycles, duration, num_points] if p is not None]) != 1:
            raise RuntimeError('One and only one of num_cycles, duration, or num_points must be set')

        max_frequency = 0.

        if not isinstance(frame, SystemFrame):
            frame = SystemFrame(frame, self)

        max_freq_diffs = list()
        for q1, q2 in self._coupling.keys():
            freq_diffs = np.subtract.outer(frame[q1].frequency, frame[q2].frequency)
            max_freq_diffs.append(np.amax(np.abs(freq_diffs)))

        for qid, drives in self._drive.items():
            if len(drives) == 0:
                continue

            drive_freqs = list()
            for drive in drives:
                frequency = drive.frequency
                if isinstance(frequency, ParameterExpression):
                    try:
                        args = tuple(freq_args[param] for param in frequency.parameters)
                    except KeyError:
                        raise RuntimeError(f'Frequency for a drive term of {qid} is parametric but no value'
                                           ' was given as freq_args')

                    drive_freqs.append(frequency.evaluate(args))
                else:
                    drive_freqs.append(frequency)

            drive_freqs = np.array(drive_freqs)
            if self.use_rwa:
                rel_sign = -1
            else:
                rel_sign = 1

            frame_drive_freqs = np.add.outer(rel_sign * drive_freqs, frame[qid].frequency)
            max_freq_diffs.append(np.amax(np.abs(frame_drive_freqs)))

        max_frequency = max(max_freq_diffs)

        if max_frequency == 0.:
            eigvals = self.eigenvalues()

            max_level_gaps = list()

            for axis in range(self.num_qudits):
                max_level_gaps.append(np.amax(np.diff(eigvals, axis=axis)))

            max_frequency = max(max_level_gaps)

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

        return instance
