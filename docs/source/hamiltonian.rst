=======================================================
Hamiltonian for a statically coupled multi-qudit system
=======================================================

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
===========================

When :math:`|\nu_j + \xi_k^l| \gg |\nu_j - \xi_k^l|` for all :math:`j, k, l`, we can apply the rotating-wave approximation (RWA) to the drive Hamiltonian and ignore
the fast-oscillating terms:

.. math::

    \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{i \rho_{jk}} \sum_l e^{-i (\epsilon_{jk}^l t - \phi_{k}^l)} \sqrt{l+1} |l+1\rangle_k \langle l |_k + \mathrm{h.c.} \right),

where :math:`\epsilon_{jk}^l := \nu_j - \xi_k^l`.

The RWA drive Hamiltonian in the frequency domain is (assuming :math:`\tilde{r}_j` has support only around the frame frequencies)

.. math::

    \bar{H}_{\mathrm{d}} = \sum_{jk} \alpha_{jk} \frac{\Omega_j}{2} \int d\nu \left( \tilde{r}_j(\nu) e^{i \rho_{jk}} \sum_l e^{-i [(\nu - \xi_k^l) t - \phi_{k}^l]} |l+1\rangle_k \langle l |_k + \mathrm{h.c.} \right).

.. _drive-hamiltonian:

Drive Hamiltonian
=================

The drive Hamiltonian for a given channel :math:`j`, qudit :math:`k`, level :math:`l` is

.. math::

    \tilde{H}_{\mathrm{d}}|_{jk}^{l} = \alpha_{jk} \frac{\Omega_j}{2} \left( r_j(t) e^{-i (\nu_j t - \rho_{jk})} + \mathrm{c.c.} \right) \left( e^{i (\xi_k^{l} t + \phi_k^{l})} \sqrt{l+1} |l + 1 \rangle_k \langle l |_k  + \mathrm{h.c.} \right).

Let

.. math::

    R_{jk}(t) = \alpha_{jk} e^{i\rho_{jk}} \frac{\Omega_j}{2} r_j(t)

and

.. math::

    A^{l}_{k} & = e^{-i \phi^{l}_{k}} \sqrt{l + 1} | l \rangle_k \langle l + 1 |_k, \\
    X^{l}_{k} & = A^{l\dagger}_{k} + A^{l}_{k} \\
    Y^{l}_{k} & = i(A^{l\dagger}_{k} - A^{l}_{k}).

Then the drive term above is

.. math::

    \tilde{H}_{\mathrm{d}}|_{jk}^{l} = & R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} A^{l\dagger}_{k} + R^{*}_{jk}(t) e^{i (\nu_j - \xi_k^{l}) t} A^{l}_{k}
                                       + R^{*}_{jk}(t) e^{i (\nu_j + \xi_k^{l}) t} A^{l\dagger}_{k} + R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t} A^{l}_{k} \\
                                     = & \mathrm{Re}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} + R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t}] X^{l}_{k}
                                       + \mathrm{Im}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} - R_{jk}(t) e^{-i (\nu_j + \xi_k^{l}) t}] Y^{l}_{k} \\
                                     = & 2 \mathrm{Re}[R_{jk}(t) e^{-i \nu_j t}] [\cos(\xi_k^{l} t) X^{l}_{k} + \sin(\xi_k^{l} t) Y^{l}_{k}].

With the rotating-wave approximation we instead have

.. math::

    \bar{H}_{\mathrm{d}}|_{jk}^{l} = & R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t} A^{l\dagger}_{k} + R^{*}_{jk}(t) e^{i (\nu_j - \xi_k^{l}) t} A^{l}_{k} \\
                                     = & \mathrm{Re}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t}] X^{l}_{k} + \mathrm{Im}[R_{jk}(t) e^{-i (\nu_j - \xi_k^{l}) t}] Y^{l}_{k}.

The representation in terms of :math:`X^{l}_{k}` and :math:`Y^{l}_{k}` operators has several advantages over using :math:`A^{l}_{k}` and :math:`A^{l\dagger}_{k}`:

- When :math:`r_j(t)` is a callable, QuTiP `sesolve` seems to run slightly faster when :math:`X` and :math:`Y` with real coefficients are passed as Hamiltonian terms.
- The Hamiltonian is manifestly Hermitian.
- For a pure real or imaginary :math:`R_{jk}(t)`, on-resonant (:math:`\nu_j = \xi_k^{l}`) drive, the RWA Hamiltonian reduces to a single term.
