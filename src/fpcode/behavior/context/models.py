"""Configured categorical event-context HMM over frozen lower unit evidence."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from fpcode.behavior.compatibility import StreamRequirements, require_compatible
from fpcode.behavior.unit_adapters import occurrence_blocks
from fpcode.contracts.behavior_results import BehaviorResult, StatePosteriorRecord
from fpcode.contracts.behavior_units import UnitOccurrence, UnitStream, UnitType
from fpcode.contracts.timelines import frozen_array
from fpcode.statistics.sequence_inference import (
    forward_backward_hmm,
    validate_log_probabilities,
    viterbi_hmm,
)

IMPLEMENTATION_STATUS = "implemented"


@dataclass(frozen=True)
class EventContextHMM:
    """Event-coordinate inference; persistence does not model intervening seconds.

    Unknown types either fail (default) or are explicitly treated as a missing
    categorical observation with likelihood one, using policy 'uninformative'.
    Parameter learning is outside this configured first-pass model.
    """
    model_id: str
    n_contexts: int
    context_names: tuple[str, ...]
    unit_catalog: tuple[UnitType, ...]
    log_initial: np.ndarray
    log_transition: np.ndarray
    log_emissions: np.ndarray
    unmatched_time_policy: str = "break"
    observation_coordinate: str = "event"
    unknown_type_policy: str = "error"
    memory_budget_bytes: int = 256*1024*1024
    max_work: int = 100000000

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if isinstance(self.n_contexts, bool) or not isinstance(self.n_contexts, Integral) or self.n_contexts < 1:
            raise ValueError("n_contexts must be a positive integer")
        names, catalog = tuple(self.context_names), tuple(self.unit_catalog)
        if len(names) != self.n_contexts or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("context names must be nonempty, unique and match state count")
        if not catalog or any(not isinstance(t, UnitType) for t in catalog) or len({t.type_id for t in catalog}) != len(catalog):
            raise ValueError("unit catalog must contain unique UnitType records")
        if self.unmatched_time_policy not in {"break", "event_steps"} or self.observation_coordinate != "event":
            raise ValueError("event context requires explicit break/event_steps unmatched-time policy")
        if self.unknown_type_policy not in {"error", "uninformative"}:
            raise ValueError("unknown type policy must be error or uninformative")
        for budget in (self.memory_budget_bytes, self.max_work):
            if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
                raise ValueError("inference budgets must be positive integers")
        c, k = self.n_contexts, len(catalog)
        if 16*(c+c*c+c*k) > self.memory_budget_bytes:
            raise ValueError("context parameter memory budget exceeded")
        initial, transition, emission = (frozen_array(a, dtype=float) for a in
                                        (self.log_initial, self.log_transition, self.log_emissions))
        if initial.shape != (c,) or transition.shape != (c, c) or emission.shape != (c, k):
            raise ValueError("context parameter shapes disagree with catalog/state count")
        validate_log_probabilities(initial, axis=0, name="initial distribution")
        validate_log_probabilities(transition, name="transition distribution")
        validate_log_probabilities(emission, name="categorical emissions")
        for name, value in (("context_names", names), ("unit_catalog", catalog), ("log_initial", initial),
                            ("log_transition", transition), ("log_emissions", emission)):
            object.__setattr__(self, name, value)

    def _blocks(self, stream):
        require_compatible(stream, StreamRequirements(levels=("primitive", "episode", "assembly"), exclusive=True))
        return occurrence_blocks(stream, break_gaps=self.unmatched_time_policy == "break")

    def _emissions(self, block):
        if 64*(len(block)+1)*self.n_contexts + 256*len(block) > self.memory_budget_bytes:
            raise ValueError("context observation memory budget exceeded")
        codes = {t.type_id: i for i, t in enumerate(self.unit_catalog)}
        emissions = np.zeros((len(block), self.n_contexts))
        for i, occ in enumerate(block):
            if occ.type_id in codes:
                emissions[i] = self.log_emissions[:, codes[occ.type_id]]
            elif self.unknown_type_policy == "error":
                raise ValueError(f"unknown held-out unit type {occ.type_id}; declare an explicit association/policy")
        return emissions

    def _forward(self, emissions):
        return forward_backward_hmm(self.log_initial, self.log_transition, emissions,
                                    memory_budget_bytes=self.memory_budget_bytes, max_work=self.max_work)

    def score(self, stream: UnitStream) -> float:
        """Marginal log likelihood summed across independent eligible blocks."""
        return float(sum(self._forward(self._emissions(b)).log_likelihood for b in self._blocks(stream)))

    def infer(self, stream: UnitStream) -> BehaviorResult:
        """Retain event posteriors/runs; project only observed physical intervals."""
        posteriors, intervals, occurrences, likelihoods, event_runs = [], [], [], [], []
        catalog = tuple(UnitType(n, f"context state {n}") for n in self.context_names)
        event_offset = 0
        for block_index, block in enumerate(self._blocks(stream)):
            emissions = self._emissions(block)
            fb = self._forward(emissions)
            if not np.isfinite(fb.log_likelihood):
                raise ValueError(f"impossible observation sequence under context model {self.model_id}")
            path, _ = viterbi_hmm(self.log_initial, self.log_transition, emissions,
                                 memory_budget_bytes=self.memory_budget_bytes, max_work=self.max_work)
            likelihoods.append(fb.log_likelihood)
            posteriors.append(fb.posteriors)
            changes = np.flatnonzero(np.r_[True, path[1:] != path[:-1], True])
            for a, b in zip(changes[:-1], changes[1:]):
                event_runs.append((block_index, event_offset+int(a), event_offset+int(b), self.context_names[path[a]]))
            for i, occ in enumerate(block):
                state = int(path[i])
                intervals.append(occ.interval)
                occurrences.append(UnitOccurrence(f"ctx_{len(occurrences)}", self.context_names[state], occ.interval,
                                                  float(fb.posteriors[i, state]), "posterior"))
            event_offset += len(block)
        context_stream = UnitStream(stream.session, "context", self.model_id, stream.clock, stream.fps,
                                    stream.valid_intervals, catalog, tuple(occurrences))
        probs = np.vstack(posteriors) if posteriors else np.empty((0, self.n_contexts))
        posterior = StatePosteriorRecord("event", catalog, probs, np.ones(len(probs), bool), tuple(intervals),
                                         stream.session, self.model_id, clock=stream.clock)
        return BehaviorResult(context_stream, posteriors=posterior, source_streams=(stream,),
                              diagnostics={"log_likelihood": float(sum(likelihoods)),
                                           "block_log_likelihoods": tuple(likelihoods), "event_runs": tuple(event_runs),
                                           "operand_level": stream.level, "source_model_id": stream.model_id,
                                           "unknown_type_policy": self.unknown_type_policy,
                                           "unmatched_time_policy": self.unmatched_time_policy})

    def predict(self, stream: UnitStream) -> BehaviorResult:
        return self.infer(stream)


__all__ = ["IMPLEMENTATION_STATUS", "EventContextHMM"]
