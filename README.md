qudit-sim: Qudit pulse simulation and effective Hamiltonian analysis
====================================================================

[Y. Iiyama](https://github.com/yiiyama)

Qudit-sim is a tool for extracting effective Hamiltonians / gate unitaries of arbitrary microwave pulses applied to a system of statically coupled d-level quantum oscillators (qudits). Its intended usage is as a base for prototyping new pulse sequences that implement custom quantum gates involving more than two oscillator levels.

Most of the heavy-lifting is done by [QuTiP](https://qutip.org) through its Schrodinger equation solver (`sesolve`). The main functions of this tool are to prepare the Hamiltonian object passed to `sesolve` from the input parameters, and to interpret the result of the simulation.

As the focus of the tool is on prototyping rather than performing an accurate simulation, the tool currently assumes a simple model of the system. In particular, incoherent effects (qudit relaxation, depolarization, etc.) are not considered.

Installation
------------

Most recent tagged versions are available in PyPI.

```
pip install qudit-sim
```

To install from source,

```
git clone https://github.com/UTokyo-ICEPP/qudit-sim
cd qudit-sim
pip install .
```

### Requirements

Exact versions of the required packages have not been checked, but reasonably recent versions should do.

- numpy
- scipy
- qutip
- h5py
- matplotlib
- jax: If using the fidelity maximization (default) method for the effective Hamiltonian extraction
- optax: If using the fidelity maximization (default) method for the effective Hamiltonian extraction
- rqutils

Documentation and examples
--------------------------

The documentation including the mathematical background of the simulation and effective Hamiltonian extraction is available at [Read the Docs](https://qudit-sim.readthedocs.io).

Example analyses using qudit-sim are available as notebooks in the `examples` directory.

Contribute
----------

You are most welcome to contribute to qudit-sim development by either forking this repository and sending pull requests or filing bug reports / suggestions for improvement at the [issues page](https://github.com/UTokyo-ICEPP/qudit-sim/issues).
