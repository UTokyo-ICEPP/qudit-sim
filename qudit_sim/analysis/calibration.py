from typing import Hashable
import numpy as np

import rqutils.paulis as paulis

from ..hamiltonian import HamiltonianBuilder
from ..pulse import Gaussian, Drag
from ..pulse_sim import pulse_sim
from .decompositions import gate_components

unit_time = 0.2e-9

def find_pi_pulse(
    hgen: HamiltonianBuilder,
    qudit_id: Hashable,
    level: int,
    duration: float = unit_time * 160,
    sigma: int = unit_time * 40
):
    # The "X" Pauli is the third from the last in the list of Paulis for the given level
    resonant_component = (level + 2) ** 2 - 3

    # Index tuple to isolate the single-qudit excitations
    qudit_index = hgen.qudit_index(qudit_id)
    single_qudit_idx = (0,) * qudit_index + (slice(None),) + (0,) * (hgen.num_qudits - qudit_index - 1)

    # Initialize the Hamiltonian
    hgen.clear_drive()
    hgen.set_global_frame('dressed')

    drive_frequency = hgen.frame(qudit_id).frequency[level]

    ## Make the tlist

    hgen.add_drive(qudit_id, frequency=drive_frequency, amplitude=1.)
    tlist = hgen.make_tlist(10, duration=duration)
    hgen.clear_drive()

    ## Set up the amplitude scan

    # Approximate the Gaussian with a triangle -> angle = Hamiltonian * duration / 2.
    # For a resonant drive, the transition Hamiltonian strength is drive_base * amplitude * sqrt(level+1) / 2
    # Therefore amplitude = 2pi / (duration * drive_base * sqrt(level+1))
    drive_amplitude = hgen.qudit_params(qudit_id).drive_amplitude
    rough_amp_estimate = 2. * np.pi / duration / drive_amplitude / np.sqrt(level + 1)

    amplitudes = rough_amp_estimate * np.linspace(0.8, 1.2, 20)
    pulses = list(Gaussian(duration=duration, amp=amp, sigma=sigma) for amp in amplitudes)

    hgens = hgen.make_scan('amplitude', pulses, qudit_id=qudit_id, frequency=drive_frequency)

    ## Run the simulation
    sim_results = pulse_sim(hgens, tlist=tlist, rwa=False)
    components_list = np.array(gate_components(sim_results))
    single_qudit_components = components_list[(slice(None),) + single_qudit_idx]

    ## Find the best amplitude
    halfpi = np.pi / 2.
    idx = np.searchsorted(single_qudit_components[:, resonant_component], halfpi)
    y0, y1 = single_qudit_components[idx - 1:idx + 1, resonant_component]
    x0, x1 = amplitudes[idx - 1:idx + 1]
    best_amplitude = ((halfpi - y0) * x1 - (halfpi - y1) * x0) / (y1 - y0)

    ## Set up the beta scan
    hgen.clear_drive()

    # I have no theoretical backup for this range of values
    betas = np.linspace(0., sigma / 20. * (level + 2), 20)

    pulses = list(Drag(duration=duration, amp=best_amplitude, sigma=sigma, beta=beta) for beta in betas)

    hgens = hgen.make_scan('amplitude', pulses, qudit_id=qudit_id, frequency=drive_frequency)

    ## Run the simulation
    sim_results = pulse_sim(hgens, tlist=tlist, rwa=False)
    components_list = np.array(gate_components(sim_results))
    single_qudit_components = components_list[(slice(None),) + single_qudit_idx]

    ## Find the best beta
    z_diagonal = np.zeros(hgen.num_levels)
    z_diagonal[level] = 1
    z_diagonal[level + 1] = -1
    z_coeff = paulis.components(np.diag(z_diagonal))

    z_components = np.tensordot(single_qudit_components, z_coeff, (1, 0))

    idx = np.searchsorted(z_components, 0.)
    y0, y1 = z_components[idx - 1:idx + 1]
    x0, x1 = betas[idx - 1:idx + 1]
    best_beta = ((0. - y0) * x1 - (0. - y1) * x0) / (y1 - y0)

    return best_amplitude, best_beta
