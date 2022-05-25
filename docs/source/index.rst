.. qudit-sim documentation master file, created by
   sphinx-quickstart on Tue Apr 12 17:15:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:

   Home <self>
   Qudit Hamiltonian <hamiltonian>
   Effective Hamiltonian <heff>

.. toctree::
   :hidden:
   :caption: API Reference

   apidocs/frontend
   apidocs/heff
   apidocs/backend

=====================================
Welcome to qudit-sim's documentation!
=====================================

``qudit-sim`` is a tool for extracting effective Hamiltonians / gate unitaries of arbitrary microwave pulses applied to a system of statically coupled d-level quantum oscillators (qudits). Its intended usage is as a base for prototyping new pulse sequences that implement custom quantum gates involving more than two oscillator levels.

Most of the heavy-lifting is done by `QuTiP <https://qutip.org>`_ through its Schrodinger equation solver (``qutip.sesolve``). The main functions of this tool are to prepare the Hamiltonian object passed to `sesolve` from the input parameters, and to interpret the result of the simulation.

As the focus of the tool is on prototyping rather than performing an accurate simulation, the tool currently assumes a simple model of the system. In particular, incoherent effects (qudit relaxation, depolarization, etc.) are not considered.

Project status
==============

The package is under a highly active development, with frequent interface-breaking updates.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
