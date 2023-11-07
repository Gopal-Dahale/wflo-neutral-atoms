# pylint: disable-all

"""Pasqal QAA Samplers"""
from time import perf_counter_ns
from typing import Optional

import dimod
import numpy as np
from numpy.random import randint
from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator, SimConfig

from .pasqal_utils import *


def qaa(h, coupler_starts, coupler_ends, coupler_weights, parameters):
    seed = parameters["seed"]

    Q, scale_factor, detuning_factor = get_qubo_matrix(
        h, coupler_starts, coupler_ends, coupler_weights
    )
    coords = embed_qubo(Q, seed)

    qubits = dict(enumerate(coords))
    reg = Register(qubits)

    # Building the quantum adiabatic algorithm

    # We choose a median value between the min and the max
    Omega = np.median(Q[Q > 0].flatten())
    delta_0 = -detuning_factor  # just has to be negative
    delta_f = -delta_0  # just has to be positive

    # time in ns, we choose a time long enough to ensure the propagation of information in the system
    T = parameters["duration"]

    # similar to
    # Efficient protocol for solving combinatorial graph problems
    # on neutral-atom quantum processors
    # adiabatic_pulse = Pulse(
    #     InterpolatedWaveform(T, [1e-9, Omega, Omega, Omega / 4, 1e-9]),
    #     InterpolatedWaveform(T, [delta_0, 0, delta_f / 3, delta_f, delta_f]),
    #     0,
    # )

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )

    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    timestamp_simul = perf_counter_ns()

    # LOCAL SIMULATOR

    simul = QutipEmulator.from_sequence(
        seq,
        sampling_rate=parameters["sampling_rate"],
        config=parameters["config"],
        evaluation_times=parameters["evaluation_times"],
        with_modulation=parameters["with_modulation"],
    )
    results = simul.run(**parameters["run_options"])

    timestamp_postprocess = perf_counter_ns()

    # np.random.seed(seed)
    count_dict = results.sample_final_state(N_samples=parameters["n_samples"])
    result_dict = dict(
        sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
    )
    counts = np.array(list(result_dict.values()))
    samples = np.array(list(map(lambda x: list(map(int, x[0])), result_dict.items())))
    energies = np.sum((Q @ samples.T) * samples.T, axis=0) / scale_factor

    return samples, energies, counts, timestamp_simul, timestamp_postprocess


class PasqalQAASampler(dimod.Sampler, dimod.Initialized):
    """Pasqal QAA Sampler"""

    parameters = None
    properties = None

    def __init__(
        self,
        duration,
        sampling_rate: float,
        config: Optional[SimConfig] = None,
        evaluation_times="Full",
        with_modulation: bool = False,
        run_options: dict = {},
        n_samples=1000,
        seed: Optional[int] = None,
    ):
        self.parameters = {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "config": config,
            "evaluation_times": evaluation_times,
            "with_modulation": with_modulation,
            "run_options": run_options,
            "n_samples": n_samples,
            "seed": randint(2**31) if seed is None else seed,
        }
        self.properties = {}

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        **kwargs,
    ) -> dimod.SampleSet:
        timestamp_preprocess = perf_counter_ns()
        # get the original vartype so we can return consistently
        original_vartype = bqm.vartype

        # convert to binary (if needed)
        if bqm.vartype is not dimod.BINARY:
            bqm = bqm.change_vartype(dimod.BINARY, inplace=False)

        parsed = self.parse_initial_states(bqm, seed=self.parameters["seed"])
        variable_order = parsed.initial_states.variables

        # read out the BQM
        ldata, (irow, icol, qdata), _ = bqm.to_numpy_vectors(
            variable_order=variable_order
        )

        timestamp_sample = perf_counter_ns()
        samples, energies, counts, timestamp_simul, timestamp_postprocess = qaa(
            ldata, irow, icol, qdata, self.parameters
        )

        response = dimod.SampleSet.from_samples(
            (samples, variable_order),
            energy=energies + bqm.offset,  # add back in the offset
            vartype=dimod.BINARY,
            num_occurrences=counts,
        )

        response.change_vartype(original_vartype, inplace=True)

        response.info.update(
            dict(
                timing=dict(
                    preprocessing_ns=timestamp_sample - timestamp_preprocess,
                    embed_ns=timestamp_simul - timestamp_sample,
                    sampling_ns=timestamp_postprocess - timestamp_simul,
                    # Update timing info last to capture the full postprocessing time
                    postprocessing_ns=perf_counter_ns() - timestamp_postprocess,
                )
            )
        )

        return response
