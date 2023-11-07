# pylint: disable-all

"""Pasqal QAOA Sampler"""
from numbers import Integral
from time import perf_counter_ns
from typing import Optional

import dimod
import numpy as np
from numpy.random import randint
from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator, SimConfig
from scipy.optimize import minimize
from scipy.sparse import coo_array
from scipy.spatial.distance import pdist, squareform

from .pasqal_utils import *


def get_cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    return z.T @ Q @ z


def get_cost(counter, Q):
    cost = sum(counter[key] * get_cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples


def quantum_loop(parameters, seq, n_layers, seq_options, run_options, n_samples):
    params = np.array(parameters)
    t_params, s_params = np.reshape(params.astype(int), (2, n_layers))
    assigned_seq = seq.build(t_list=t_params, s_list=s_params)
    simul = QutipEmulator.from_sequence(assigned_seq, **seq_options)
    results = simul.run(**run_options)

    # sample from the state vector
    count_dict = results.sample_final_state(N_samples=n_samples)
    return count_dict


def func(param, *args):
    Q, seq, n_layers, seq_options, run_options, n_samples = args
    C = quantum_loop(param, seq, n_layers, seq_options, run_options, n_samples)
    cost = get_cost(C, Q)
    return cost


def qaoa(h, coupler_starts, coupler_ends, coupler_weights, parameters):
    seed = parameters["seed"]
    n_layers = parameters["n_layers"]

    seq_options = {
        "sampling_rate": parameters["sampling_rate"],
        "config": parameters["config"],
        "evaluation_times": parameters["evaluation_times"],
        "with_modulation": parameters["with_modulation"],
    }

    Q, scale_factor, _ = get_qubo_matrix(
        h, coupler_starts, coupler_ends, coupler_weights
    )
    coords = embed_qubo(Q, seed)
    qubits = dict(enumerate(coords))
    reg = Register(qubits)

    # Building the quantum approximate optimization algorithm

    # Parametrized sequence
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch0", "rydberg_global")

    t_list = seq.declare_variable("t_list", size=n_layers)
    s_list = seq.declare_variable("s_list", size=n_layers)

    T = parameters["duration"]
    for t, s in zip(t_list, s_list):
        pulse_1 = Pulse.ConstantPulse(T * t, 1.0, 0.0, 0)
        pulse_2 = Pulse.ConstantPulse(T * s, 0.0, 1.0, 0)

        seq.add(pulse_1, "ch0")
        seq.add(pulse_2, "ch0")

    seq.measure("ground-rydberg")

    # np.random.seed(seed)  # ensures reproducibility
    scores = []
    params = []

    n_repetitions = parameters["n_repetitions"]

    for _ in range(n_repetitions):
        guess = {
            "t": np.random.uniform(1, 10, n_layers),
            "s": np.random.uniform(1, 10, n_layers),
        }

        res = minimize(
            func,
            args=(
                Q,
                seq,
                n_layers,
                seq_options,
                parameters["run_options"],
                parameters["n_samples"],
            ),
            x0=np.r_[guess["t"], guess["s"]],
            method="Nelder-Mead",
            tol=1e-5,
            options={"maxiter": parameters["max_iter"]},
        )

        scores.append(res.fun)
        params.append(res.x)

    count_dict = quantum_loop(
        params[np.argmin(scores)],
        seq,
        n_layers,
        seq_options,
        parameters["run_options"],
        parameters["n_samples"],
    )

    result_dict = dict(
        sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
    )
    counts = np.array(list(result_dict.values()))
    samples = np.array(list(map(lambda x: list(map(int, x[0])), result_dict.items())))
    energies = np.sum((Q @ samples.T) * samples.T, axis=0) / scale_factor

    return samples, energies, counts


class PasqalQAOASampler(dimod.Sampler, dimod.Initialized):
    """Pasqal QAOA Sampler"""

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
        n_layers=1,
        max_iter=10,
        n_repetitions=1,
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
            "n_layers": n_layers,
            "max_iter": max_iter,
            "n_repetitions": n_repetitions,
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
        samples, energies, counts = qaoa(ldata, irow, icol, qdata, self.parameters)
        timestamp_postprocess = perf_counter_ns()

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
                    sampling_ns=timestamp_postprocess - timestamp_sample,
                    # Update timing info last to capture the full postprocessing time
                    postprocessing_ns=perf_counter_ns() - timestamp_postprocess,
                )
            )
        )

        return response
