from typing import Any, Dict, Sequence, List
import collections
import copy
import re
import numpy as np
import qutip as qtp

FREQUENCY_EPSILON = 1.e-6

class ScaledExpression:
    def __init__(self, arg1, arg2=None, epsilon=1.e-6):
        self._expression = None
        
        if isinstance(arg1, ScaledExpression):
            self.scale = arg1.scale
            self._expression = arg1._expression
            self.epsilon = arg1.epsilon
            return
        
        if arg2 is None and isinstance(arg1, tuple):
            arg1, arg2 = arg1
        
        if arg2 is None:
            if isinstance(arg1, str):
                try:
                    self.scale = eval(arg1)
                except NameError:
                    self.scale = 1.
                    self._expression = arg1
            else:
                self.scale = arg1
        else:
            self.scale = arg1
            try:
                self.scale *= eval(arg2)
            except (TypeError, NameError):
                self._expression = arg2

        self.epsilon = epsilon
        
    @property
    def expression(self):
        if self._expression is None:
            return None
        elif (re.match(r'[a-z]*\(.+\)$', self._expression)
            or re.match(r'[^+-]+$', self._expression)):
            return self._expression
        else:
            return f'({self._expression})'
        
    def __str__(self):
        if self.is_zero():
            return '0'
        elif self._expression is None:
            return f'{self.scale}'
        elif self.scale == 1.:
            return self.expression
        elif self.scale == -1.:
            return f'-{self.expression}'
        else:
            return f'{self.scale}*{self.expression}'
        
    def __repr__(self):
        return f'ScaledExpression({self.scale}, {self._expression}, {self.epsilon})'
        
    def __add__(self, rhs):
        if isinstance(rhs, ScaledExpression):
            if self._expression is None and rhs._expression is None:
                return ScaledExpression(self.scale + rhs.scale)
            elif self.is_zero():
                return ScaledExpression(rhs)
            elif rhs.is_zero():
                return ScaledExpression(self)
            else:
                if rhs.scale < 0.:
                    return ScaledExpression(1., f'{self}-{-rhs}')
                else:
                    return ScaledExpression(1., f'{self}+{rhs}')
        else:
            if isinstance(rhs, str):
                return self.__add__(ScaledExpression(1., rhs))
            else:
                return self.__add__(ScaledExpression(rhs))
            
    def __sub__(self, rhs):
        return self.__add__(-rhs)
    
    def __neg__(self):
        return ScaledExpression(-self.scale, self._expression, self.epsilon)
        
    def __mul__(self, rhs):
        if isinstance(rhs, ScaledExpression):
            num = self.scale * rhs.scale

            if self._expression is None:
                if rhs._expression is None:
                    return ScaledExpression(num)
                else:
                    return ScaledExpression(num, rhs.expression)
            else:
                if rhs._expression is None:
                    return ScaledExpression(num, self.expression)
                else:
                    return ScaledExpression(num, f'{self.expression}*{rhs.expression}')
        else:
            if isinstance(rhs, str):
                return self.__mul__(ScaledExpression(1., rhs))
            else:
                return self.__mul__(ScaledExpression(rhs))
            
    def __abs__(self):
        if self.is_zero():
            return ScaledExpression(0., None, self.epsilon)
        elif self._expression is None:
            return ScaledExpression(abs(self.scale), None, self.epsilon)
        else:
            return ScaledExpression(abs(self.scale), f'abs({self._expression})', self.epsilon)
        
    def scale_abs(self):
        return ScaledExpression(abs(self.scale), self._expression, self.epsilon)
            
    def is_zero(self):
        return abs(self.scale) < self.epsilon
    

class ComplexExpression:
    def __init__(self, real, imag):
        self.real = ScaledExpression(real)
        self.imag = ScaledExpression(imag)
            
    def __str__(self):
        return f'{self.real}+{self.imag}j'
    
    def __repr__(self):
        return f'ComplexExpression({self.real}, {self.imag})'
        
    def __mul__(self, rhs):
        if isinstance(rhs, ComplexExpression) or isinstance(rhs, complex):
            real = self.real * rhs.real - self.imag * rhs.imag
            imag = self.real * rhs.imag + self.imag * rhs.real
            return ComplexExpression(real, imag)
        else:
            return ComplexExpression(self.real * rhs, self.imag * rhs)
        
    def __add__(self, rhs):
        if isinstance(rhs, ComplexExpression) or isinstance(rhs, complex):
            real = self.real + rhs.real
            imag = self.imag + rhs.imag
            return ComplexExpression(real, imag)
        else:
            return ComplexExpression(self.real + rhs, self.imag)
        
    def __getitem__(self, key):
        if key == 0:
            return self.real
        elif key == 1:
            return self.imag
        else:
            raise IndexError(f'Index {key} out of range')
            
    def __abs__(self):
        if self.real.is_zero():
            return abs(self.imag)
        elif self.imag.is_zero():
            return abs(self.real)
        elif self.real.expression == self.imag.expression:
            scale = np.sqrt(np.square(self.real.scale) + np.square(self.imag.scale))
            return ScaledExpression(scale, f'abs({self.real._expression})')
        else:
            r2 = self.real * self.real
            i2 = self.imag * self.imag
            m2 = r2 + i2
            return ScaledExpression(f'np.sqrt({m2})')
            
    def is_zero(self):
        return self.real.is_zero() and self.imag.is_zero()
    
    def angle(self):
        if self.imag.is_zero():
            return np.arctan2(0., self.real.scale)
        elif self.real.is_zero():
            return np.arctan2(self.imag.scale, 0.)
        elif self.real.expression == self.imag.expression:
            return np.arctan2(self.imag.scale, self.real.scale)
        else:
            raise ValueError('Angle cannot be defined for non-static phase ComplexExpression')
            
    def polar(self):
        phase = self.angle()
        
        if self.imag.is_zero():
            radial = self.real.scale_abs()
        elif self.real.is_zero():
            radial = self.imag.scale_abs()
        elif self.real.expression == self.imag.expression:
            scale = np.sqrt(np.square(self.real.scale) + np.square(self.imag.scale))
            radial = ScaledExpression(scale, self.real._expression, self.real.epsilon)

        return radial, phase
    

class RWAHamiltonian:
    r"""Rotating-wave approximation Hamiltonian in the qudit frame.

    **Full Hamiltonian:**
    
    The full Hamiltonian of the :math:`n`-qudit system is
    
    .. math::
    
        H = H_0 + H_{\mathrm{int}} + H_{\mathrm{d}},
        
    where
    
    .. math::
    
        H_0 & = \sum_{j=0}^{n} \left[ \omega_j b_j^{\dagger} b_j + \frac{\Delta_j}{2} b_j^{\dagger} b_j (b_j^{\dagger} b_j - 1) \right] \\
        & = \sum_{j=0}^{n} \left[ \left( \omega_j - \frac{\Delta_j}{2} \right) N_j + \frac{\Delta_j}{2} N_j^2 \right],
        
        H_{\mathrm{int}} = \sum_{jk} J_{jk} \left( b_j^{\dagger} b_k + b_j b_k^{\dagger} \right),
        
        H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \left( p_j(t) \cos (\nu_j t) + q_j(t) \sin (\nu_j t) \right) \left( e^{i\phi_{jk}} b_k^{\dagger} + e^{-i\phi_{jk}} b_k \right),
        
    with
    
    - :math:`\omega_j`: Frequency of the :math:`j`th qubit
    - :math:`\Delta_j`: Anharmonicity of the :math:`j`th qudit
    - :math:`J_{jk}`: Coupling between qudits :math:`j` and :math:`k`
    - :math:`\alpha_{jk}`: Crosstalk attenuation factor of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\phi_{jk}`: Crosstalk phase shift of drive in channel :math:`j` sensed by qudit :math:`k`
    - :math:`\Omega_j`: Base amplitude of drive in channel :math:`j`
    - :math:`p_j (t), q_j (t)`: I and Q components of the pulse envelope of drive in channel :math:`j`
    - :math:`\nu_j`: Local oscillator frequency of drive in channel :math:`j`.
    
    When considering more than a single drive frequency per channel, it can be more convenient to express
    the drive Hamiltonian in the frequency domain:
    
    .. math::
    
        H_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \int d\nu \left( \tilde{p}_j(\nu) \cos (\nu t) + \tilde{q}_j(\nu) \sin (\nu t) \right) \left( e^{i\phi_{jk}} b_k^{\dagger} + e^{-i\phi_{jk}} b_k \right)
    
    **Qudit-frame Hamiltonian with Rotating-wave approximation:**
    
    We move to the qudit frame through a transformation with :math:`U_q := e^{i H_0 t}`:
    
    .. math::
    
        \tilde{H} & := U_q H U_q^{\dagger} + i \dot{U_q} U_q^{\dagger} \\
        & = U_q (H_{\mathrm{int}} + H_{\mathrm{d}}) U_q^{\dagger} =: \tilde{H}_{\mathrm{int}} + \tilde{H}_{\mathrm{d}}.
    
    Using :math:`b N = (N + 1) b` and :math:`b^{\dagger}N = (N - 1) b^{\dagger}`, we have

    .. math::
    
        b_j H_0^n = & \left\{ \sum_{k \neq j} \left[ \left( \omega_k - \frac{\Delta_k}{2} \right) N_k + \frac{\Delta_k}{2} N_k^2 \right] \right. \\
        + \left. \left( \omega_j - \frac{\Delta_j}{2} \right) (N_j + 1) + \frac{\Delta_j}{2} (N_j + 1)^2 \right\}^n b_j \\
        = & \left[ H_0 + \omega_j + \Delta_j N_j \right]^n b_j,

        b_j^{\dagger} H_0^n = & \left\{ \sum_{k \neq j} \left[ \left( \omega_k - \frac{\Delta_k}{2} \right) N_k + \frac{\Delta_k}{2} N_k^2 \right] \right. \\
        + \left. \left( \omega_j - \frac{\Delta_j}{2} \right) (N_j - 1) + \frac{\Delta_j}{2} (N_j - 1)^2 \right\}^n b_j^{\dagger} \\
        = & \left[ H_0 - \omega_j - \Delta_j (N_j - 1) \right]^n b_j^{\dagger}.
              
    The interaction Hamiltonian in the qudit frame is therefore
    
    .. math::
    
        \tilde{H}_{\mathrm{int}} = \sum_{jk} J_{jk} \left( e^{i \delta_{jk} t} e^{i [\Delta_j (N_j - 1) - \Delta_k N_k] t} b_j^{\dagger} b_k \right. \\
        \left. + e^{-i \delta_{jk} t} e^{-i [\Delta_j N_j - \Delta_k (N_k - 1)] t} b_j b_k^{\dagger} \right).

    The drive Hamiltonian is
    
    .. math::
    
        \tilde{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \Omega_j \left( p_j(t) \cos (\nu_j t) + q_j(t) \sin (\nu_j t) \right) \left( e^{i (\omega_k t + \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + e^{-i (\omega_k t + \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right),
    
    and with the rotating wave approximation (RWA)
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left[ r_j(t) e^{-i (\epsilon_{jk} t - \phi_{jk})} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + r^{*}_j(t) e^{i (\epsilon_{jk} t - \phi_{jk})} e^{-i \Delta_k N_k t} b_k \right],
        
    where :math:`\epsilon_{jk} = \nu_j - \omega_k` and :math:`r_j(t) = p_j(t) + i q_j(t)`.
    
    The RWA drive Hamiltonian in the frequency domain is (assuming :math:`\tilde{p}_j` and :math:`\tilde{q}_j` have
    support only around the qudit frequencies)
    
    .. math::
    
        \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left[ \tilde{r}_j(\nu) e^{i [(\omega_k - \nu) t + \phi_{jk}]} e^{i \Delta_k (N_k - 1) t} b_k^{\dagger} \right. \\
        \left. + \tilde{r}^{*}_j(\nu) e^{-i [(\omega_k - \nu) t + \phi_{jk}]} e^{-i \Delta_k N_k t} b_k \right].
    
    **QuTiP implementation:**
    
    Time-dependent Hamiltonian in QuTiP is represented by a two-tuple `(H, c(t))` where `H` is a static Qobj and `c(t)`
    is the time-dependent coefficient of `H`. This function returns two lists of tuples, corresponding to the 
    interaction and drive Hamiltonians, with the total list length corresponding to the number of distinct
    time dependencies in the RWA Hamiltonian :math:`\tilde{H}_{\mathrm{int}} + \bar{H}_{\mathrm{d}}`. Because
    `c(t)` must be a real function, each returned tuple contains two Qobjs corresponding to the "X"
    (:math:`\propto b^{\dagger} + b`) and "Y" (:math:`\propto i(b^{\dagger} - b)`) parts of the Hamiltonian.
        
    **TODO: Qubit optimization:**
    
    When only considering qubits, anharmonicity terms are dropped, making the formulae above somewhat simpler. In particular,
    when one or more drives are resonant with qubit frequencies, some terms in the interaction and drive Hamiltonians
    will have common frequencies (i.e. :math:`\delta_{jk}` and `\epsilon_{kj}` coincide). We should look into how to detect
    and exploit such cases in the future.
    
    Attributes:
        h_int (list): Interaction term components. List of n_coupling elements. Each element is a three-tuple
            (freq, H_x, H_y), where freq is the frequency for the given term. H_x and H_y are the coupling
            Hamiltonians proportional to the "X" and "Y" parts of the oscillating coefficients.
        h_drive (list): Drive term components. List of n_qubits elements. Each element is a three-tuple
            (expr_gen, H_x, H_y), where expr_gen is an instance of DriveExprGen.
    """
    def __init__(
        self,
        qubits: Sequence[int],
        params: Dict[str, Any],
        num_levels: int = 2
    ) -> None:
        r"""
        Args:
            qubits: List of :math:`n` qudits to include in the Hamiltonian.
            params: Hamiltonian parameters given by IBMQ `backend.configuration().hamiltonian['vars']`, optionally
                augmented with `'crosstalk'`, which should be a `dict` of form `{(i, j): z}` specifying the crosstalk
                factor `z` (complex corresponding to :math:`\alpha_{jk} e^{i\phi_{jk}}`) of drive on qubit `i` seen
                by qubit `j`. `i` and `j` are qubit ids given in `qubits`.
            num_levels: Number of oscillator levels to consider.
        """
        ## Validate and format the input

        if isinstance(qubits, int):
            qubits = (qubits,)

        num_qubits = len(qubits)

        self.qubit_index_mapping = dict((q, iq) for iq, q in enumerate(qubits))
        
        self.num_levels = num_levels

        ## Parse the parameters
        
        self.qubit_frequencies = np.array([params[f'wq{q}'] for q in qubits])
        self.anharmonicities = np.array([params[f'delta{q}'] for q in qubits])
        
        # Element [j,k]: \alpha_{jk} \exp (i \phi_{jk})
        crosstalk_matrix = np.eye(num_qubits, dtype=np.complex128)
        if 'crosstalk' in params:
            for (qsrc, qtrg), factor in params['crosstalk'].items():
                isrc = self.qubit_index_mapping[qsrc]
                itrg = self.qubit_index_mapping[qtrg]
                crosstalk_matrix[isrc, itrg] = factor

        amps = np.array(list(params[f'omegad{ch}'] for ch in qubits)) / 2.
        # Element [j,k]: \alpha_{jk} \frac{\Omega}{2} \exp (i \phi_{jk})
        self.drive_base = amps[:, None] * crosstalk_matrix
        
        self._max_frequency = 0.

        ## Compute the interaction term components

        self._static_term = qtp.Qobj()
        self._hamiltonian = []

        for q1 in qubits:
            iq1 = self.qubit_index_mapping[q1]

            for q2 in qubits[iq1 + 1:]:
                iq2 = self.qubit_index_mapping[q2]

                try:
                    J = params[f'jq{min(q1, q2)}q{max(q1, q2)}']
                except KeyError:
                    # no coupling between the qudits
                    continue

                freq_diff = (self.qubit_frequencies[iq1] - self.qubit_frequencies[iq2])

                D1 = self.anharmonicities[iq1]
                D2 = self.anharmonicities[iq2]
                
                # Define one Hamiltonian term per level pair
                # cf. Hint = sum_{jk} J_{jk} exp(i d_{jk} t) exp(i D_j (N_j - 1) t) exp(-i D_k N_k t) b_j+ b_k + H.C.

                for l1 in range(num_levels - 1):
                    for l2 in range(num_levels - 1):
                        c1 = np.sqrt(l1 + 1) * qtp.basis(num_levels, l1) * qtp.basis(num_levels, l1 + 1).dag()
                        c2 = np.sqrt(l2 + 1) * qtp.basis(num_levels, l2) * qtp.basis(num_levels, l2 + 1).dag()
                        op = [qtp.qeye(num_levels)] * num_qubits
                        op[iq1] = c1
                        op[iq2] = c2.dag()
                        h_x = J * (qtp.tensor(op).dag() + qtp.tensor(op))
                        h_y = J * 1.j * (qtp.tensor(op).dag() - qtp.tensor(op))
                        freq = freq_diff + D1 * l1 - D2 * l2

                        self._hamiltonian.append((h_x, f'cos({freq}*t)'))
                        self._hamiltonian.append((h_y, f'sin({freq}*t)'))
                        
                        self._max_frequency = max(self._max_frequency, abs(freq))
                        
    @property
    def max_frequency(self) -> float:
        return self._max_frequency
    
    @property
    def hamiltonian(self) -> List:
        if self._static_term == qtp.Qobj():
            return list(self._hamiltonian)
        else:
            # The static term does not have to be the first element (nor does it have to be a single term, actually)
            # but qutip recommends following this convention
            return [self._static_term] + self._hamiltonian

    def add_drive(
        self,
        qubit: int,
        drive_def: Dict[str, Any]
    ) -> None:

        ich = self.qubit_index_mapping[qubit]
        num_qubits = len(self.qubit_index_mapping)
        
        try:
            amp_def = drive_def["amplitude"]
        except KeyError:
            drive_amp = ComplexExpression(1., 0.)
        else:
            if isinstance(amp_def, complex):
                drive_amp = ComplexExpression(amp_def.real, amp_def.imag)
            elif isinstance(amp_def, tuple):
                drive_amp = ComplexExpression(amp_def[0], amp_def[1])
            else:
                drive_amp = ComplexExpression(amp_def, 0.)
            
        for iq in range(num_qubits):
            if self.drive_base[ich, iq] == 0.:
                continue

            # Amplitude expressions for X (even) and Y (odd) terms
            # Corresponds to R_j(t) := \alpha_{jk} \frac{\Omega_j}{2} r_j(t) \exp(i \phi_{jk})
            envelope = drive_amp * self.drive_base[ich, iq]
            try:
                radial, phase = envelope.polar()
            except ValueError:
                radial = None
            
            # Define one Hamiltonian term per level
            for level in range(self.num_levels - 1):
                # \omega_k + n \Delta_k
                level_frequency = self.qubit_frequencies[iq] + level * self.anharmonicities[iq]
                # \nu_j - \omega_k - n \Delta_k
                detuning = drive_def['frequency'] - level_frequency
                
                if abs(detuning) < FREQUENCY_EPSILON:
                    amplitude = copy.deepcopy(envelope)
                elif radial is not None:
                    carrier_phase = f'{-detuning}*t{phase:+}'
                    amplitude = ComplexExpression(f'cos({carrier_phase})', f'sin({carrier_phase})')
                    amplitude *= radial
                else:
                    carrier = ComplexExpression(f'cos({-detuning}*t)', f'sin({-detuning}*t)')
                    amplitude = carrier * envelope

                qudit_ops = [qtp.qeye(self.num_levels)] * num_qubits
                qudit_ops[iq] = (np.sqrt(level + 1) * qtp.basis(self.num_levels, level)
                    * qtp.basis(self.num_levels, level + 1).dag())
                annihilator = qtp.tensor(qudit_ops)
                h_x = annihilator.dag() + annihilator
                h_y = 1.j * (annihilator.dag() - annihilator)
                
                for h, amp in zip((h_x, h_y), amplitude):
                    if amp.is_zero():
                        continue
                        
                    if amp.expression is None:
                        self._static_term += h * amp.scale
                    else:
                        self._hamiltonian.append((h * amp.scale, amp.expression))
                        
                self._max_frequency = max(self._max_frequency, abs(detuning))
