=========================================
Effective Hamiltonian of a constant drive
=========================================

The full time evolution operator :math:`U_{H}(t) = T\left[\exp(-i \int_0^t dt' H(t'))\right]` of a driven qudit
system is time-dependent and highly nontrivial. However, when the drive amplitude is a constant, at a longer time
scale, it should be approximatable with a time evolution by a constant Hamiltonian (= effective Hamiltonian)
:math:`U_{\mathrm{eff}}(t) = \exp(-i H_{\mathrm{eff}} t)`.

Identification of this :math:`H_{\mathrm{eff}}` is essentially a linear fit to the time evolution of Pauli
components of the generator :math:`i \mathrm{log} (U_{H}(t))`. However, it's not possible to actually determine
:math:`H_{\mathrm{eff}}` from the generator because the latter is multi-valued; there are infinitely many Hermitian
matrices whose eigenvalues are separated by multiples of :math:`2 \pi` but exponentiate into the same unitary.
Therefore, we instead compose an :math:`H_{\mathrm{eff}}` ansatz first, multiply it by time and exponentiate
it to form the effective time evolution :math:`U_{\mathrm{eff}}(t)`, and maximize the fidelity

.. math::

  \mathcal{F} = \sum_{i} \big| \mathrm{tr} \left[ U_{\mathrm{eff}}(t)^{\dagger} U_{H}(t_i) \right] \big|^2.

Ring-up
=======

Applying the full-amplitude drive from the first instant may lead to unnatural oscillations of the generator
components and produce spurious terms in :math:`H_{\mathrm{eff}}`. By increasing the amplitude adiabatically
(practically speaking, just slowly), the trajectory of the generator in the space of Hermitians can be placed
in a more stable path. The resulting evolution operators is more reminiscent of the actual pulse gates, which
must also start from zero amplitude. We therefore use a one-sided GaussianSquare pulse with a very slow turn-on,
and perform the :math:`H_{\mathrm{eff}}` fit in the plateau of the pulse.
