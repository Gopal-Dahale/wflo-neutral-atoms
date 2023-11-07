import threading
from typing import Optional

from hybrid import traits
from hybrid.core import Runnable
from pulser_simulation import SimConfig

from .pasqal_qaa_sampler import PasqalQAASampler
from .pasqal_qaoa_sampler import PasqalQAOASampler

__all__ = [
    "PasqalQAAProblemSampler",
    "PasqalQAASubproblemSampler",
    "PasqalQAOAProblemSampler",
    "PasqalQAOASubproblemSampler",
]

# Subproblem samplers


class PasqalQAASubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
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
        **runopts,
    ):
        super(PasqalQAASubproblemSampler, self).__init__(**runopts)
        self.sampler = PasqalQAASampler(
            duration,
            sampling_rate,
            config,
            evaluation_times,
            with_modulation,
            run_options,
            n_samples,
            seed,
        )
        self._stop_event = threading.Event()

    def next(self, state, **runopts):
        subsamples = self.sampler.sample(state.subproblem)
        return state.updated(subsamples=subsamples)

    def halt(self):
        self._stop_event.set()


class PasqalQAOASubproblemSampler(traits.SubproblemSampler, traits.SISO, Runnable):
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
        **runopts,
    ):
        super(PasqalQAOASubproblemSampler, self).__init__(**runopts)
        self.sampler = PasqalQAOASampler(
            duration,
            sampling_rate,
            config,
            evaluation_times,
            with_modulation,
            run_options,
            n_samples,
            n_layers,
            max_iter,
            n_repetitions,
            seed,
        )
        self._stop_event = threading.Event()

    def next(self, state, **runopts):
        subsamples = self.sampler.sample(state.subproblem)
        return state.updated(subsamples=subsamples)

    def halt(self):
        self._stop_event.set()


# Problem samplers


class PasqalQAAProblemSampler(traits.ProblemSampler, traits.SISO, Runnable):
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
        **runopts,
    ):
        super(PasqalQAAProblemSampler, self).__init__(**runopts)
        self.sampler = PasqalQAASampler(
            duration,
            sampling_rate,
            config,
            evaluation_times,
            with_modulation,
            run_options,
            n_samples,
            seed,
        )
        self._stop_event = threading.Event()

    def next(self, state, **runopts):
        samples = self.sampler.sample(state.problem)
        return state.updated(samples=samples)

    def halt(self):
        self._stop_event.set()


class PasqalQAOAProblemSampler(traits.ProblemSampler, traits.SISO, Runnable):
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
        **runopts,
    ):
        super(PasqalQAOAProblemSampler, self).__init__(**runopts)
        self.sampler = PasqalQAOASampler(
            duration,
            sampling_rate,
            config,
            evaluation_times,
            with_modulation,
            run_options,
            n_samples,
            n_layers,
            max_iter,
            n_repetitions,
            seed,
        )

    def next(self, state, **runopts):
        samples = self.sampler.sample(state.problem)
        return state.updated(samples=samples)

    def halt(self):
        self._stop_event.set()
