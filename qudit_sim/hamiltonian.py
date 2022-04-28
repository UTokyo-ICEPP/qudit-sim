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
                   & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \left( b_k^{\dagger} + b_k \right)

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

    H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left( \tilde{r}_j(\nu) e^{-i (\nu t - \rho_{jk})} + \mathrm{c.c.} \right) \left( b_k^{\dagger} + b_k \right)

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

    U_q = \prod_j \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) N_j + \frac{\Delta_j}{2} N_j^2 \right] t \right\} =: \prod_j e^{i h_j t}.

Each :math:`h_j` commutes with :math:`b_k` and :math:`b_k^{\dagger}` if :math:`k \neq j`, so

.. math::

    \tilde{H}_{\mathrm{int}} & = \sum_{j<k} J_{jk} \left( \tilde{b}_j^{\dagger} \tilde{b}_k + \tilde{b}_j \tilde{b}_k^{\dagger} \right) \\
    \tilde{H}_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \left( \tilde{b}_k^{\dagger} + \tilde{b}_k \right)
    
where

.. math::

    \tilde{b}_{j} & = e^{i h_j t} b_j e^{-i h_j t}, \\
    \tilde{b}_{j}^{\dagger} & = e^{i h_j t} b_j^{\dagger} e^{-i h_j t}.

By definition :math:`b_j N_j = (N_j + 1) b_j`, which implies

.. math::

    b_j e^{-i h_j t} = \exp \left\{ -i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j + 1) + \frac{\Delta_j}{2} (N_j + 1)^2) \right] t \right\} b_j
    
and therefore

.. math::

    \tilde{b}_{j} & = \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - (N_j + 1)) + \frac{\Delta_j}{2} (N_j^2 - (N_j + 1)^2) \right] t \right\} b_j \\
                  & = e^{-i(\omega_j + \Delta_j N_j) t} b_j.

Similarly, :math:`b_j^{\dagger} N_j = (N_j - 1) b_j^{\dagger}` leads to

.. math::

    \tilde{b}_{j}^{\dagger} & = \exp \left\{ i \left[\left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - (N_j - 1)) + \frac{\Delta_j}{2} (N_j^2 - (N_j - 1)^2) \right] t \right\} b_j^{\dagger} \\
                  & = e^{i(\omega_j + \Delta_j (N_j - 1)) t} b_j^{\dagger}.

The interaction Hamiltonian in the qudit frame is therefore

.. math::

    \tilde{H}_{\mathrm{int}} & = \sum_{j<k} J_{jk} \left( e^{i (\omega_j - \omega_k) t} e^{i [\Delta_j (N_j - 1) - \Delta_k N_k] t} b_j^{\dagger} b_k + \mathrm{h.c.} \right) \\
                             & = \sum_{j<k} J_{jk} \left( e^{i (\omega_j - \omega_k) t} \sum_{lm} e^{i (\Delta_j l - \Delta_k m) t} \sqrt{(l+1)(m+1)} | l + 1 \rangle_j \langle l |_j \otimes | m \rangle_k \langle m + 1 |_k + \mathrm{h.c.} \right).
    
In the last line, we used the expansion of the annihilation operator :math:`b_j = \sum_{l} \sqrt{l+1} | l \rangle_j \langle l + 1 |_j` and its Hermitian conjugate.

The drive Hamiltonian in the qudit frame is

.. math::

    \tilde{H}_{\mathrm{d}} & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \left( e^{i(\omega_k + \Delta_k (N_k - 1))t} b_k^{\dagger} + \mathrm{h.c.} \right) \\
                           & = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i(\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \sum_l \left( e^{i \omega_k t} e^{i \Delta_k l t} \sqrt{l+1} | l + 1 \rangle_k \langle l |_k + \mathrm{h.c.} \right).
    
General frame
-------------

We can move to an arbitrary frame specifying the frequency and phase offset for each level gap of each qudit. Let for qudit :math:`j` the frequency
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
from dataclasses import dataclass
import numpy as np
import qutip as qtp

from .pulse import PulseSequence
from .drive import DriveTerm, cos_freq, sin_freq

REL_FREQUENCY_EPSILON = 1.e-7

@dataclass
class Frame:
    frequency: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None


@dataclass
class QuditParams:
    qubit_frequency: float
    anharmonicity: float
    drive_amplitude: float
    frame: Frame


class HamiltonianGenerator:
    r"""Generator for a Hamiltonian with static transverse couplings and external drive terms.

    Args:
        num_levels: Number of energy levels to consider.
        qudits: If passing `params` to initialize the Hamiltonian, list of qudit numbers to include.
        params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally
            augmented with `'crosstalk'`, which should be a `dict` of form `{(j, k): z}` specifying the crosstalk
            factor `z` (complex corresponding to :math:`\alpha_{jk} e^{i\rho_{jk}}`) of drive on qudit `j` seen
            by qudit `k`. `j` and `k` are qudit ids given in `qudits`.
    """
    def __init__(
        self,
        num_levels: int = 2,
        qudits: Optional[Union[int, Sequence[int]]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.num_levels = num_levels
        
        # Makes use of dict order guarantee from python 3.7
        self.qudit_params = dict()
        self.coupling = dict()
        self.crosstalk = dict()
        self.drive = dict()
        
        self._max_frequency_int = None
        self._max_frequency_drive = None
        self._need_tlist = False
        
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
            
    def add_qudit(
        self,
        qubit_frequency: float,
        anharmonicity: float,
        drive_amplitude: float,
        qudit_id: Optional[Hashable] = None,
        position: Optional[int] = None,
        frame_frequency: Optional[np.ndarray] = None,
        frame_phase: Optional[np.ndarray] = None,
    ) -> None:
        """Add a qudit to the system.
        
        Args:
            qubit_frequency: Qubit frequency.
            anharmonicity: Anharmonicity.
            drive_amplitude: Base drive amplitude in rad/s.
            qudit_id: Identifier for the qudit. If None, the position (order of addition) is used.
            position: If an integer, the qudit is inserted into the specified position.
            frame_frequency: Frame frequency for all level spacings.
            frame_phase: Frame phase offset for all level spacings.
        """
        params = QuditParams(qubit_frequency=qubit_frequency, anharmonicity=anharmonicity,
                             drive_amplitude=drive_amplitude,
                             frame=Frame(frequency=frame_frequency, phase=frame_phase))
        
        if qudit_id is None:
            if position is None:
                qudit_id = len(self.qudit_params)
            else:
                qudit_id = position
                
        if qudit_id in self.qudit_params:
            raise KeyError(f'Qudit id {qudit_id} already exists.')

        if position is None:
            self.qudit_params[qudit_id] = params
        elif position > len(self.qudit_params):
            raise IndexError(f'Position {position} greater than number of existing parameters')
        else:
            qudit_params = dict(list(self.qudit_params.items())[:position])
            qudit_params[qudit_id] = params
            qudit_params.update(list(self.qudit_params.items())[position:])

            self.qudit_params.clear()
            self.qudit_params.update(qudit_params)
        
    def set_frame(
        self,
        qudit_id: Hashable,
        frequency: Union[np.ndarray, None],
        phase: Union[np.ndarray, None]
    ) -> None:
        """Set the frame for the qudit.
        
        The arrays must be of size `num_levels - 1`.
        """
        self.qudit_params[qudit_id].frame = Frame(frequency=frequency, phase=phase)
            
    def add_coupling(self, q1: Hashable, q2: Hashable, value: float) -> None:
        """Add a coupling term between two qudits."""
        self.coupling[frozenset({self.qudit_params[q1], self.qudit_params[q2]})] = value
            
    def add_crosstalk(self, source: Hashable, target: Hashable, factor: complex) -> None:
        r"""Add a crosstalk term from the source channel to the target qudit.
        
        Args:
            source: Qudit ID of the source channel.
            target: Qudit ID of the target qudit.
            factor: Crosstalk coefficient (:math:`\alpha_{jk} e^{i\rho_{jk}}`).
        """
        self.crosstalk[(self.qudit_params[source], self.qudit_params[target])] = factor
        
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
        self.drive[self.qudit_params[qudit_id]] = DriveTerm(frequency=frequency, amplitude=amplitude,
                                                            phase=phase)
        
    @property
    def max_frequency(self) -> float:
        """Maximum frequency appearing in this Hamiltonian."""
        return max(self._max_frequency_int, self._max_frequency_drive)
    
    @property
    def need_tlist(self) -> bool:
        return self._need_tlist

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
            tlist: If not None, all callable Hamiltonian coefficients are called with (tlist, args) and the resulting
                arrays are instead passed to sesolve.
            args: Arguments to the callable coefficients.
        
        Returns:
            A list of Hamiltonian terms that can be passed to qutip.sesolve.
        """
        if tlist is None and self._need_tlist:
            raise RuntimeError('This Hamiltonian must be generated with a tlist')
            
        hstatic = self.generate_hdiag()
        hint = self.generate_hint(compile_hint=compile_hint)
        hdrive = self.generate_hdrive(rwa=rwa)
        
        if hint and isinstance(hint[0], qtp.Qobj):
            hstatic += hint.pop(0)
        if hdrive and isinstance(hdrive[0], qtp.Qobj):
            hstatic += hdrive.pop(0)

        hamiltonian = []
        
        if hstatic != qtp.Qobj():
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            hamiltonian.append(hstatic)
            
        if tlist is not None:
            for h, f in hint + hdrive:
                if callable(f):
                    hamiltonian.append([h, f(tlist, args)])
                else:
                    hamiltonian.append([h, f])
                    
        else:
            hamiltonian.extend(hint)
            hamiltonian.extend(hdrive)
            
        return hamiltonian
        
    def generate_hdiag(self) -> qtp.Qobj:
        """Generate the diagonal term of the Hamiltonian.
        
        Returns:
            A Qobj representing Hdiag. The object may be empty if the qudit frame is used.
        """
        hdiag = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        
        for iq, params in enumerate(self.qudit_params.values()):
            if params.frame.frequency is None:
                # qudit frame -> free Hamiltonian is null
                continue
                
            qudit_hfree = np.arange(self.num_levels) * (params.qubit_frequency - params.anharmonicity / 2.)
            qudit_hfree += np.square(np.arange(self.num_levels)) * params.anharmonicity / 2.
            energy_offset = qudit_hfree - np.cumsum(np.concatenate((np.zeros(1), params.frame.frequency)))
            qudit_op = qtp.Qobj(inpt=np.diag(energy_offset))

            ops = [qtp.qeye(self.num_levels)] * num_qudits
            ops[iq] = qudit_op
            
            hdiag += qtp.tensor(ops)
            
        return hdiag
    
    def generate_hint(self, compile_hint: bool = True) -> List:
        """Generate the interaction Hamiltonian.
        
        Args:
            compile_hint: If True, interaction Hamiltonian terms are given as compilable strings.
            
        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        self._max_frequency_int = 0.

        hint = list()
        hstatic = qtp.Qobj()

        num_qudits = len(self.qudit_params)
        params_list = list(self.qudit_params.values())
        
        for iq1, iq2 in np.ndindex((num_qudits, num_qudits)):
            if iq2 <= iq1:
                continue
                
            p1 = params_list[iq1]
            p2 = params_list[iq2]

            try:
                coupling = self.coupling[frozenset({p1, p2})]
            except KeyError:
                # no coupling between the qudits
                continue

            for l1, l2 in np.ndindex((self.num_levels - 1, self.num_levels - 1)):
                # exp(i (xi_{p1,l1} - xi_{p2,l2}) t) sqrt(l1 + 1) sqrt(l2 + 1) |l1+1>|l2><l1|<l2+1| + h.c.
                
                # Annihilator terms for this level combination
                ann1 = np.sqrt(l1 + 1) * qtp.basis(self.num_levels, l1) * qtp.basis(self.num_levels, l1 + 1).dag()
                ann2 = np.sqrt(l2 + 1) * qtp.basis(self.num_levels, l2) * qtp.basis(self.num_levels, l2 + 1).dag()
                
                ops = [qtp.qeye(self.num_levels)] * num_qudits
                ops[iq1] = ann1.dag()
                ops[iq2] = ann2
                
                op = qtp.tensor(ops)
                frequency = 0.
                
                for params, level, sign in [(p1, l1, 1.), (p2, l2, -1.)]:
                    if params.frame.phase is not None:
                        op *= np.exp(sign * 1.j * params.frame.phase[level])
                        
                    if params.frame.frequency is None:
                        frequency += sign * (params.qubit_frequency + level * params.anharmonicity)
                    else:
                        frequency += sign * params.frame.frequency[level]
                        
                if abs(frequency) < REL_FREQUENCY_EPSILON * (p1.qubit_frequency + p2.qubit_frequency) * 0.5:
                    hstatic += op + op.dag()

                else:
                    h_x = coupling * (op + op.dag())
                    h_y = coupling * 1.j * (op - op.dag())
                
                    if compile_hint:
                        hint.append([h_x, f'cos({frequency}*t)'])
                        hint.append([h_y, f'sin({frequency}*t)'])
                    else:
                        hint.append([h_x, cos_freq(frequency)])
                        hint.append([h_y, sin_freq(frequency)])

                    self._max_frequency_int = max(self._max_frequency_int, abs(frequency))
                    
        if hstatic != qtp.Qobj():
            hint.insert(0, hstatic)

        return hint

    def generate_hdrive(self, rwa: bool = True) -> List:
        """Generate the drive Hamiltonian.
        
        Args:
            rwa: If True, apply the rotating-wave approximation.
            
        Returns:
            A list of Hamiltonian terms. The first entry may be a single Qobj instance if there is a static term.
            Otherwise the entries are 2-lists `[Qobj, c(t)]`.
        """
        self._max_frequency_drive = 0.
        self._need_tlist = False
        
        hdrive = list()
        hstatic = qtp.Qobj()
        
        num_qudits = len(self.qudit_params)
        
        # Construct the Qobj for each qudit/level first
        qops = list()

        for iq, params in enumerate(self.qudit_params.values()):
            for level in range(self.num_levels - 1):
                cre = np.sqrt(level + 1) * qtp.basis(self.num_levels, level + 1) * qtp.basis(self.num_levels, level).dag()

                ops = [qtp.qeye(self.num_levels)] * num_qudits
                ops[iq] = cre

                op = qtp.tensor(ops)
                
                if params.frame.phase is not None:
                    op *= np.exp(1.j * params.frame.phase[level])
                
                if params.frame.frequency is None:
                    frame_frequency = params.qubit_frequency + level * params.anharmonicity
                else:
                    frame_frequency = params.frame.frequency[level]

                qops.append((params, op, frame_frequency))

        # Loop over the drive channels
        for ch_params, drive in self.drive.items():
            if isinstance(drive.amplitude, np.ndarray):
                self._need_tlist = True
            
            # Loop over the driven operators
            for params, creation_op, frame_frequency in qops:
                drive_base = ch_params.drive_amplitude / 2.
                
                if ch_params != params:
                    try:
                        drive_base *= self.crosstalk[(ch_params, params)]
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
                        
                elif isinstance(fn_x, np.ndarray):
                    if np.any(fn_x != 0.):
                        hdrive.append([h_x, fn_x])
                    if np.any(fn_y != 0.):
                        hdrive.append([h_y, fn_y])

                else:
                    hdrive.append([h_x, fn_x])
                    hdrive.append([h_y, fn_y])
                            
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
            amp2 = np.trace(hstat @ hstat).real / (2 ** len(self.qudit_params))
            return np.linspace(0., 2. * np.pi / np.sqrt(amp2) * num_cycles, points_per_cycle * num_cycles)
        else:
            return np.linspace(0., 2. * np.pi / self.max_frequency * num_cycles, points_per_cycle * num_cycles)
