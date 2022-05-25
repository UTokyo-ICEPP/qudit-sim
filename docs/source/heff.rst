=========================================
Effective Hamiltonian of a constant drive
=========================================

The full time evolution operator :math:`U_{H}(t) = T\left[\exp(-i \int_0^t dt' H(t'))\right]` of a driven qudit
system is time-dependent and highly nontrivial. However, when the drive amplitude is a constant, at a longer time
scale, it should be approximatable with a time evolution by a constant Hamiltonian (= effective Hamiltonian)
:math:`U_{\mathrm{eff}}(t) = \exp(-i H_{\mathrm{eff}} t)`.

Identification of this :math:`H_{\mathrm{eff}}` is essentially a linear fit to the time evolution of Pauli
components of :math:`i \mathrm{log} (U_{H}(t))`. In qudit-sim we have two implementations of this fit:

- `"fidelity"` finds the effective Pauli components that maximize
  :math:`\sum_{i} \big| \mathrm{tr} \left[ U(t_i)\, \exp \left(i H_{\mathrm{eff}} t_i \right)\right] \big|^2`.
- `"leastsq"` performs a least-squares fit to individual components of :math:`i \mathrm{log} (U_{H}(t))`.

The fidelity method is usually more robust, but the least squares method allows better "fine-tuning". A combined
method is also available.
