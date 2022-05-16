# Pulse simulation for qudits

This repository contains a tool for extracting effective Hamiltonians / gate unitaries of arbitrary microwave pulses applied to a system of coupled d-level quantum oscillators (qudits). Its intended usage is as a base for prototyping new pulse sequences that implement custom quantum gates involving more than two oscillator levels.

Most of the heavy-lifting is done by [QuTiP](https://qutip.org) through its Schrodinger equation solver (`sesolve`). The main functions of this tool are to prepare the Hamiltonian object passed to `sesolve` from the input parameters, and to interpret the result of the simulation.

The tool takes the following input:

* Base (qubit) frequencies and anharmonicities of the oscillators
* Transverse coupling coefficients between the qudits
* (Classical) crosstalk matrix between the qudits
* Carrier frequencies and envelope functions of the drive pulses

The first two items are given in the format used to specify the IBM Quantum machines. The crosstalk matrix determines how a drive pulse applied to a qudit is felt by another qudit.

As the focus of the tool is on prototyping rather than performing an accurate simulation, the tool currently assumes a simple model of the system. In particular, incoherent effects (qudit relaxation, depolarization, etc.) are not considered. The full Hamiltonian can be found in the docstring of `make_hamiltonian_components` in `pulse_sim.py`.

Check out `demo.ipynb` for example usage.
