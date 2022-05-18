r"""
======================================================================================
Hamiltonian for a statically coupled multi-qudit system (:mod:`qudit_sim.hamiltonian`)
======================================================================================

.. currentmodule:: qudit_sim.hamiltonian

Fundamentals
============

The full Hamiltonian of an :math:`n`-qudit system with static coupling and drive terms is

.. math::

    H = H_0 + H_{\mathrm{int}} + H_{\mathrm{d}},

where

.. math::

    H_0 & = \sum_{j=1}^{n} \left[ \omega_j b_j^{\dagger} b_j + \frac{\Delta_j}{2} b_j^{\dagger} b_j (b_j^{\dagger} b_j - 1) \right]
          = \sum_{j=1}^{n} \left[ \left( \omega_j - \frac{\Delta_j}{2} \right) N_j + \frac{\Delta_j}{2} N_j^2 \right], \\
    H_{\mathrm{int}} & = \sum_{j<k} J_{jk} \left( b_j^{\dagger} b_k + b_j b_k^{\dagger} \right), \\
    H_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \Omega_j \left( p_j(t) \cos (\nu_j t - \rho_{jk}) + q_j(t) \sin (\nu_j t - \rho_{jk}) \right)
                       \left( b_k^{\dagger} + b_k \right) \\
                   & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right)
                       \left( b_k^{\dagger} + b_k \right)

with :math:`b_j^{\dagger}` and :math:`b_j` the creation and annihilation operators for qudit :math:`j` and

- :math:`\omega_j`: Qubit frequency of qudit :math:`j`
- :math:`\Delta_j`: Anharmonicity of qudit :math:`j`
- :math:`J_{jk}`: Coupling between qudits :math:`j` and :math:`k`
- :math:`\Omega_j`: Base amplitude of drive in channel :math:`j`
- :math:`p_j (t), q_j (t)`: I and Q components of the pulse envelope of drive in channel :math:`j`, and :math:`r_j (t) = p_j(t) + iq_j(t)`
- :math:`\nu_j`: Local oscillator frequency of drive in channel :math:`j`
- :math:`\alpha_{jk}`: Crosstalk attenuation factor of drive in channel :math:`j` sensed by qudit :math:`k`
- :math:`\rho_{jk}`: Crosstalk phase shift of drive in channel :math:`j` sensed by qudit :math:`k`.

When considering more than a single drive frequency per channel, it can be more convenient to express
the drive Hamiltonian in the frequency domain:

.. math::

    H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left( \tilde{r}_j(\nu) e^{-i (\nu t - \rho_{jk})}
                      + \mathrm{c.c.} \right) \left( b_k^{\dagger} + b_k \right)

Change of frame
===============

Qudit frame
-----------

We move to the qudit frame through a transformation with :math:`U_q := e^{i H_0 t}`:

.. math::

    \tilde{H} & := U_q H U_q^{\dagger} + i \dot{U_q} U_q^{\dagger} \\
    & = U_q (H_{\mathrm{int}} + H_{\mathrm{d}}) U_q^{\dagger} =: \tilde{H}_{\mathrm{int}} + \tilde{H}_{\mathrm{d}}.

:math:`\tilde{H}` is the generator of time evolution for state :math:`U_q |\psi\rangle`:

.. math::

    i \frac{\partial}{\partial t} U_q |\psi\rangle & = (i \dot{U}_q + U_q H) |\psi\rangle \\
                                                   & = \tilde{H} U_q |\psi\rangle.

To write down :math:`\tilde{H}_{\mathrm{int}}` and :math:`\tilde{H}_{\mathrm{d}}` in terms of :math:`\{b_j\}_j` and :math:`\{N_j\}_j`,
we first note that :math:`U_q` can be factored into commuting subsystem unitaries:

.. math::

    U_q = \prod_j \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) N_j + \frac{\Delta_j}{2} N_j^2 \right] t \right\}
        =: \prod_j e^{i h_j t}.

Each :math:`h_j` commutes with :math:`b_k` and :math:`b_k^{\dagger}` if :math:`k \neq j`, so

.. math::

    \tilde{H}_{\mathrm{int}} & = \sum_{j<k} J_{jk} \left( \tilde{b}_j^{\dagger} \tilde{b}_k + \tilde{b}_j \tilde{b}_k^{\dagger} \right) \\
    \tilde{H}_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right)
                               \left( \tilde{b}_k^{\dagger} + \tilde{b}_k \right)

where

.. math::

    \tilde{b}_{j} & = e^{i h_j t} b_j e^{-i h_j t}, \\
    \tilde{b}_{j}^{\dagger} & = e^{i h_j t} b_j^{\dagger} e^{-i h_j t}.

By definition :math:`b_j N_j = (N_j + 1) b_j`, which implies

.. math::

    b_j e^{-i h_j t} = \exp \left\{ -i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j + 1)
                                             + \frac{\Delta_j}{2} (N_j + 1)^2) \right] t \right\} b_j

and therefore

.. math::

    \tilde{b}_{j} & = \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - (N_j + 1))
                                           + \frac{\Delta_j}{2} (N_j^2 - (N_j + 1)^2) \right] t \right\} b_j \\
                  & = e^{-i(\omega_j + \Delta_j N_j) t} b_j.

Similarly, :math:`b_j^{\dagger} N_j = (N_j - 1) b_j^{\dagger}` leads to

.. math::

    \tilde{b}_{j}^{\dagger} & = \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - (N_j - 1))
                                                     + \frac{\Delta_j}{2} (N_j^2 - (N_j - 1)^2) \right] t \right\} b_j^{\dagger} \\
                  & = e^{i(\omega_j + \Delta_j (N_j - 1)) t} b_j^{\dagger}.

The interaction Hamiltonian in the qudit frame is therefore

.. math::

    \tilde{H}_{\mathrm{int}} & = \sum_{j<k} J_{jk} \left( e^{i (\omega_j - \omega_k) t} e^{i [\Delta_j (N_j - 1) - \Delta_k N_k] t}
                                                          b_j^{\dagger} b_k + \mathrm{h.c.} \right) \\
                             & = \sum_{j<k} J_{jk} \left( e^{i (\omega_j - \omega_k) t} \sum_{lm} e^{i (\Delta_j l - \Delta_k m) t}
                                                          \sqrt{(l+1)(m+1)} | l + 1 \rangle_j \langle l |_j \otimes | m \rangle_k
                                                          \langle m + 1 |_k + \mathrm{h.c.} \right).

In the last line, we used the expansion of the annihilation operator :math:`b_j = \sum_{l} \sqrt{l+1} | l \rangle_j \langle l + 1 |_j`
and its Hermitian conjugate.

The drive Hamiltonian in the qudit frame is

.. math::

    \tilde{H}_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right)
                               \left( e^{i(\omega_k + \Delta_k (N_k - 1))t} b_k^{\dagger} + \mathrm{h.c.} \right) \\
                           & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right)
                               \sum_l \left( e^{i \omega_k t} e^{i \Delta_k l t} \sqrt{l+1} | l + 1 \rangle_k \langle l |_k + \mathrm{h.c.} \right).

Dressed frame
-------------

Even in the absense of a drive, :math:`\tilde{H}_{\mathrm{int}}` above actually causes slow phase drifts in the qudit frame. As it is difficult
to see this from a time-dependent :math:`\tilde{H}_{\mathrm{int}}`, we move back once again to the lab frame and diagonalize
:math:`H_{\mathrm{stat}} = H_0 + H_{\mathrm{int}}` as

.. math::

    H_{\mathrm{stat}} = V E V^{\dagger}.

The unitary :math:`V` is chosen to be

.. math::

    V = I + \eta

that minimizes :math:`|\eta|` while satisfying the diagonalization condition above. Given that :math:`H_{\mathrm{int}}` is off-diagonal and
:math:`|H_{\mathrm{int}}| \ll |H_0|`, this results in

.. math::

    E = H_0 + \delta

for some small diagonal :math:`\delta`.

The time evolution by :math:`H_{\mathrm{stat}}` is

.. math::

    e^{-i H_{\mathrm{stat}} t} & = V e^{-iEt} V^{\dagger} \\
                               & = e^{-iEt} + \eta e^{-iEt} + e^{-iEt} \eta^{\dagger} + \eta e^{-iEt} \eta^{\dagger}.

Because :math:`\tilde{H}_{\mathrm{int}}` is the generator of time evolution in the qudit frame

.. math::

    & T \left[ \exp \left(-i \int_{0}^{t} dt' \tilde{H}_{\mathrm{int}} (t') \right) \right] U_q(0) |\psi(0)\rangle
    = U_q(t) |\psi(t)\rangle \\
    & = U_q(t) e^{-i H_{\mathrm{stat}} t} |\psi(0)\rangle.

Therefore, for any free-Hamiltonian eigenstate :math:`|l\rangle`,

.. math::

    T \left[ \exp \left(-i \int_{0}^{t} dt' \tilde{H}_{\mathrm{int}} (t') \right) \right] |l\rangle
    = e^{-i \delta_{l} t} |l\rangle + e^{i H_0 t} \left(\eta e^{-iEt} + e^{-iEt} \eta^{\dagger} + \eta e^{-iEt} \eta^{\dagger} \right) |l\rangle,

where :math:`\delta_{l}` is the :math:`l`-th element of :math:`\delta`.

To eliminate these phase drifts, we would like to work in the frame defined by :math:`U_E = e^{i E t}`. However, this mathematically trivial change
of frame is not physically practical, because :math:`E` does not necessarily render itself to a sum of single-qudit operators, while all drive and
readout are performed in terms of individual qudits. Therefore we fall back to a frame defined by :math:`U_d = e^{i D t}`, where

.. math::

    D = \sum_{j=1}^{n} \sum_{l} \left( \langle l |_j \otimes \langle 0 |^{\otimes n-1} E | l \rangle_j \otimes | 0 \rangle^{\otimes n-1} \right)
        | l \rangle_j \langle l |_j,

which eliminates the phase drifts of single-qudit excitations. This frame rotates at "dressed" frequencies, i.e., free-qudit frequencies
shifted by the effects of inter-qudit interactions.

General frame
-------------

We can also move to an arbitrary frame specifying the frequency and phase offset for each level gap of each qudit. Let for qudit :math:`j` the frequency
and the phase offset between level :math:`l` and :math:`l+1` be :math:`\xi_{j}^{l}` and :math:`\phi_{j}^{l}`, and :math:`\Xi_{j}^{l} := \sum_{m<l} \xi_{j}^{m}`,
:math:`\Phi_{j}^{l} := \sum_{m<l} \phi_{j}^{m}`. Then the transformation unitary is

.. math::

    U_f := \exp \left[ i \sum_j \sum_l \left( \Xi_j^l t + \Phi_j^{l} \right) |l\rangle_j \langle l |_j \right].

:math:`U_f` commutes with the free Hamiltonian :math:`H_0` but :math:`i \dot{U}_f U_f^{\dagger} \neq -H_0` in general, so

.. math::

    \tilde{H} = U_f H U_f^{\dagger} + i \dot{U_f} U_f^{\dagger} = H_{\mathrm{diag}} + \tilde{H}_{\mathrm{int}} + \tilde{H}_{\mathrm{d}}.

The three terms can be expressed in terms of individual qudit levels as

.. math::

    H_{\mathrm{diag}} & = \sum_{j} \sum_{l} \left[ \left( \omega_j - \frac{\Delta_j}{2} \right) l + \frac{\Delta_j}{2} l^2 - \Xi_j^{l} \right] |l\rangle_j \langle l|_j, \\
    \tilde{H}_{\mathrm{int}} & = \sum_{j<k} J_{jk} \sum_{lm} \left( e^{i [(\xi_j^{l} - \xi_{k}^{m}) t + (\phi_j^{l} - \phi_k^{m})]} \sqrt{(l+1)(m+1)} |l+1\rangle_j \langle l|_j \otimes |m\rangle_k \langle m+1|_k + \mathrm{h.c.} \right), \\
    \tilde{H}_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i (\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \sum_l \left( e^{i (\xi_k^{l} t + \phi_k^{l})} \sqrt{l+1} |l + 1 \rangle_k \langle l |_k  + \mathrm{h.c.} \right).

Rotating-wave approximation
---------------------------

When :math:`|\nu_j + \xi_k^l| \gg |\nu_j - \xi_k^l|` for all :math:`j, k, l`, we can apply the rotating-wave approximation (RWA) to the drive Hamiltonian and ignore
the fast-oscillating terms:

.. math::

    \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{i \rho_{jk}} \sum_l e^{-i (\epsilon_{jk}^l t - \phi_{k}^l)} \sqrt{l+1} |l+1\rangle_k \langle l |_k + \mathrm{h.c.} \right),

where :math:`\epsilon_{jk}^l := \nu_j - \xi_k^l`.

The RWA drive Hamiltonian in the frequency domain is (assuming :math:`\tilde{r}_j` has support only around the frame frequencies)

.. math::

    \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left( \tilde{r}_j(\nu) e^{i \rho_{jk}} \sum_l e^{-i [(\nu - \xi_k^l) t - \phi_{k}^l]} |l+1\rangle_k \langle l |_k + \mathrm{h.c.} \right).



QuTiP implementation
====================

Time-dependent Hamiltonian in QuTiP is represented by a list of two-lists `[H, c(t)]` where `H` is a static Qobj and
`c(t)` is the time-dependent coefficient of `H`. There must be one such two-list per distinct time dependency.
As apparent from above, in the frame-transformed interaction Hamiltonian,
each level combination of each pair of coupled qudits has its own frequency. Similarly, the drive Hamiltonian has a
distinct frequency for each level of each qudit. Therefore the QuTiP Hamiltonian list typically contains a large number
of entries.

While `c(t)` can be a complex function (and therefore `H` be a non-Hermitian matrix), having `c(t)` real and `H` Hermitian
seems to be advantageous in terms of calculation speed. Therefore, in our implementation, the Hamiltonian terms are split
into symmetric (e.g. :math:`|l+1\rangle \langle l| + |l\rangle \langle l+1|`) and antisymmetric
(e.g. :math:`i(|l+1\rangle \langle l| - |l\rangle \langle l+1|)`) parts whenever possible, with the corresponding time
dependencies given by cosine and sine functions, respectively.
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

REL_FREQUENCY_EPSILON = 1.e-7

@dataclass(frozen=True)
class Frame:
    """Frame specification for a single level gap of a qudit."""
    frequency: np.ndarray
    phase: np.ndarray

@dataclass(frozen=True)
class QuditParams:
    """Parameters defining a qudit."""
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float

class HamiltonianGenerator:
    r"""Generator for a Hamiltonian with static transverse couplings and external drive terms.

    Args:
        num_levels: Number of energy levels to consider.
        qudits: If passing `params` to initialize the Hamiltonian, list of qudit numbers to include.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally
            augmented with `'crosstalk'`, which should be a `dict` of form `{(j, k): z}` specifying the crosstalk
            factor `z` (complex corresponding to :math:`\alpha_{jk} e^{i\rho_{jk}}`) of drive on qudit `j` seen
            by qudit `k`. `j` and `k` are qudit ids given in `qudits`.
        default_frame: Default global frame to use. `set_global_frame(default_frame)` is executed each time
            `add_qudit` or `add_coupling` is called.
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
        return max(self._max_frequency_int, self._max_frequency_drive)

    def qudit_params(self, qudit_id: Hashable) -> QuditParams:
        """Qudit parameters."""
        return self._qudit_params[qudit_id]

    def coupling(self, q1: Hashable, q2: Hashable) -> float:
        """Coupling constant between the two qudits."""
        return self._coupling[frozenset({self._qudit_params[q1], self._qudit_params[q2]})]

    def crosstalk(self, source: Hashable, target: Hashable) -> complex:
        """Crosstalk factor from source to target."""
        return self._crosstalk[(self._qudit_params[source], self._qudit_params[target])]

    def drive(self, qudit_id: Hashable) -> List[DriveTerm]:
        """Drive terms for the qudit."""
        return list(self._drive[self._qudit_params[qudit_id]])

    def frame(self, qudit_id: Hashable) -> Frame:
        """Frame of the qudit."""
        return self._frame[self._qudit_params[qudit_id]]

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

        self._drive[params] = list()

        self.set_global_frame(self.default_frame)

    def _qudit_frame_frequencies(self, params: QuditParams) -> np.ndarray:
        return params.qubit_frequency + np.arange(self._num_levels - 1) * params.anharmonicity

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
        params = self._qudit_params[qudit_id]

        if frequency is None:
            frequency = self._qudit_frame_frequencies(params)

        if phase is None:
            phase = np.zeros(self._num_levels - 1)
        elif isinstance(phase, float):
            phase = np.full(self._num_levels - 1, phase)

        self._frame[params] = Frame(frequency=frequency, phase=phase)

    def set_global_frame(self, frame: str) -> None:
        r"""Set frames for all qudits globally.

        The allowed frame names are:

        - 'qudit': Set frame frequencies to the individual qudit level gaps disregarding the couplings.
          Equivalent to calling `set_frame(qid, frequency=None, phase=None)` for all `qid`.
        - 'lab': Set frame frequencies to zero.
        - 'dressed': Diagonalize the static Hamiltonian (in the lab frame) and set the frame frequencies
          to cancel the phase drifts of single-qudit excitations.

        Args:
            frame: 'qudit', 'lab', or 'dressed'.
        """

        if frame == 'dressed' and len(self._coupling) == 0:
            frame = 'qudit'

        if frame == 'qudit':
            for qid in self._qudit_params.keys():
                self.set_frame(qid)

        elif frame == 'lab':
            for qid in self._qudit_params.keys():
                self.set_frame(qid, frequency=np.zeros(self._num_levels - 1))

        elif frame == 'dressed':
            # Move to the lab frame first to generate a fully static H0+Hint
            self.set_global_frame('lab')

            hamiltonian = self.generate_hdiag() + self.generate_hint()[0]
            eigvals, unitary = np.linalg.eigh(hamiltonian.full())
            # hamiltonian == unitary @ np.diag(eigvals) @ unitary.T.conjugate()

            # Row index of the biggest contributor to each column
            k = np.argmax(np.abs(unitary), axis=1)
            # Reordered eigenvalues (makes the corresponding change-of-basis unitary closest to identity)
            eigvals = eigvals[k]
            # Reshape the array to extract the single-qudit excitation subspaces
            eigvals = eigvals.reshape((self.num_levels,) * self.num_qudits)

            for iq, qid in enumerate(self._qudit_params.keys()):
                indices = (0,) * iq + (slice(None),) + (0,) * (self.num_qudits - iq - 1)
                energies = eigvals[indices]
                self.set_frame(qid, np.diff(energies))

        else:
            raise ValueError(f'Global frame {frame} is not defined')

    def add_coupling(self, q1: Hashable, q2: Hashable, value: float) -> None:
        """Add a coupling term between two qudits."""
        self._coupling[frozenset({self._qudit_params[q1], self._qudit_params[q2]})] = value

        self.set_global_frame(self.default_frame)

    def add_crosstalk(self, source: Hashable, target: Hashable, factor: complex) -> None:
        r"""Add a crosstalk term from the source channel to the target qudit.

        Args:
            source: Qudit ID of the source channel.
            target: Qudit ID of the target qudit.
            factor: Crosstalk coefficient (:math:`\alpha_{jk} e^{i\rho_{jk}}`).
        """
        self._crosstalk[(self._qudit_params[source], self._qudit_params[target])] = factor

    def add_drive(
        self,
        qudit_id: Hashable,
        frequency: Optional[float] = None,
        amplitude: Union[float, complex, str, np.ndarray, PulseSequence, Callable, None] = 1.+0.j,
        phase: Optional[float] = None
    ) -> None:
        r"""Add a drive term.

        Args:
            qudit_id: Qudit to apply the drive to.
            frequency: Carrier frequency of the drive. None is allowed if amplitude is a PulseSequence
                that starts with SetFrequency.
            amplitude: Function `r(t)`.
            phase: The phase value of `amplitude` when it is a str or a callable and is known to have
                a constant phase. None otherwise.
        """
        drive = DriveTerm(frequency=frequency, amplitude=amplitude, phase=phase)
        self._drive[self._qudit_params[qudit_id]].append(drive)

    def generate(
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
        hstatic = self.generate_hdiag()
        hint = self.generate_hint(compile_hint=compile_hint, tlist=tlist, args=args)
        hdrive = self.generate_hdrive(rwa=rwa, tlist=tlist, args=args)

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

    def generate_hdiag(self) -> qtp.Qobj:
        """Generate the diagonal term of the Hamiltonian.

        Returns:
            A Qobj representing Hdiag. The object may be empty if the qudit frame is used.
        """
        hdiag = qtp.Qobj()

        for iq, params in enumerate(self._qudit_params.values()):
            frame = self._frame[params]

            energy_offset = np.cumsum(self._qudit_frame_frequencies(params)) - np.cumsum(frame.frequency)
            # If in qudit frame, energy_offset is zero for all levels
            if np.allclose(energy_offset, np.zeros_like(energy_offset)):
                continue

            diagonal = np.concatenate((np.zeros(1), energy_offset))
            qudit_op = qtp.Qobj(inpt=np.diag(diagonal))

            ops = [qtp.qeye(self._num_levels)] * self.num_qudits
            ops[iq] = qudit_op

            hdiag += qtp.tensor(ops)

        return hdiag

    def generate_hint(
        self,
        compile_hint: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Generate the interaction Hamiltonian.

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

        params_list = list(self._qudit_params.values())

        for iq1, iq2 in np.ndindex((self.num_qudits, self.num_qudits)):
            if iq2 <= iq1:
                continue

            p1 = params_list[iq1]
            p2 = params_list[iq2]

            try:
                coupling = self._coupling[frozenset({p1, p2})]
            except KeyError:
                # no coupling between the qudits
                continue

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

                for params, level, sign in [(p1, l1, 1.), (p2, l2, -1.)]:
                    frame = self._frame[params]

                    if frame.phase[level] != 0.:
                        op *= np.exp(sign * 1.j * frame.phase[level])

                    frequency += sign * frame.frequency[level]

                if abs(frequency) < REL_FREQUENCY_EPSILON * (p1.qubit_frequency + p2.qubit_frequency) * 0.5:
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

    def generate_hdrive(
        self,
        rwa: bool = True,
        tlist: Optional[np.ndarray] = None,
        args: Optional[Dict[str, Any]] = None
    ) -> List:
        """Generate the drive Hamiltonian.

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

        for iq, params in enumerate(self._qudit_params.values()):
            frame = self._frame[params]

            for level in range(self._num_levels - 1):
                cre = np.sqrt(level + 1) * qtp.basis(self._num_levels, level + 1) * qtp.basis(self._num_levels, level).dag()

                ops = [qtp.qeye(self._num_levels)] * self.num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)

                if frame.phase[level] != 0.:
                    op *= np.exp(1.j * frame.phase[level])

                qops.append((params, op, frame.frequency[level]))

        # Loop over the drive channels
        for ch_params, drives in self._drive.items():
            for drive in drives:
                if isinstance(drive.amplitude, np.ndarray) and tlist is None:
                    raise RuntimeError('Time points array is needed to generate a Hamiltonian with array-based drive.')

                # Loop over the driven operators
                for params, creation_op, frame_frequency in qops:
                    drive_base = ch_params.drive_amplitude / 2.

                    if ch_params != params:
                        try:
                            drive_base *= self._crosstalk[(ch_params, params)]
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
        num_cycles: int
    ) -> np.ndarray:
        """Generate a list of time points using the maximum frequency in the Hamiltonian.

        When the maximum frequency is 0 (i.e. single-qubit simulation with resonant drive), return the time
        range from 0 to 2pi/sqrt(tr(Hstat . Hstat) / 2^nq).

        Args:
            points_per_cycle: Number of points per cycle at the highest frequency.
            num_cycles: Number of overall cycles.

        Returns:
            Array of time points.
        """
        if self.max_frequency == 0.:
            hamiltonian = self.generate()
            if not hamiltonian:
                raise RuntimeError('Cannot determine the tlist')

            hstat = hamiltonian[0]
            amp2 = np.trace(hstat.full() @ hstat.full()).real / (2 ** self.num_qudits)
            return np.linspace(0., 2. * np.pi / np.sqrt(amp2) * num_cycles, points_per_cycle * num_cycles)
        else:
            return np.linspace(0., 2. * np.pi / self.max_frequency * num_cycles, points_per_cycle * num_cycles)

    def make_scan(
        self,
        scan_type: str,
        values: Sequence,
        **kwargs
    ) -> List['HamiltonianGenerator']:
        """Generate a list of copies of self varied over a single attribute.

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
                                   phase=kwargs.get('phase'))

        elif scan_type == 'frequency':
            for value, instance in zip(values, copies):
                instance.add_drive(qudit_id=kwargs['qudit_id'],
                                   frequency=value,
                                   amplitude=kwargs.get('amplitude', 1.+0.j),
                                   phase=kwargs.get('phase'))

        elif scan_type == 'coupling':
            for value, instance in zip(values, copies):
                instance.add_coupling(q1=kwargs['q1'], q2=kwargs['q2'], value=value)

        return copies
