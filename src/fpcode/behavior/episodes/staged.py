"""Probabilistic staged episode HMM/HSMM alternative for behavioural sequences.

Implements categorical and optional diagonal-Gaussian staged episode models with
background, ordered phases, explicit duration PMFs, and segment-based inference.
Preserves segment traceback across immediate same-routine reentries, handles
boundary censoring honestly, and emits native-frame episodes with rich phase
and completion evidence envelopes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np

from fpcode.behavior.compatibility import StreamRequirements, require_compatible
from fpcode.behavior.unit_adapters import occurrence_blocks
from fpcode.contracts.behavior_results import (
    BehaviorResult,
    CompletionEvidence,
    PhaseRecord,
    ScopedOccurrenceReference,
)
from fpcode.contracts.behavior_units import FrameSpan, UnitOccurrence, UnitStream, UnitType
from fpcode.contracts.timelines import frozen_array
from fpcode.statistics.sequence_inference import (
    logsumexp,
    validate_log_probabilities,
    viterbi_hmm,
    viterbi_hsmm,
)

IMPLEMENTATION_STATUS = "implemented"


@dataclass(frozen=True)
class StagedFitResult:
    """Immutable fitted model and data-only effort evidence from an explicit fit."""

    model: StagedEpisodeModel
    diagnostics: Mapping

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class StagedEpisodeModel:
    """Categorical detector with optional Gaussian features and duration PMFs.

    States
    ------
    0 : Background / non-episode state
    1..P : Ordered phases of the target routine (e.g. 1=entry, P=final phase)
    P+1 : Optional second background state, allowing adjacent background segments
    """

    model_id: str
    routine_name: str
    phase_names: tuple[str, ...]
    primitive_catalog: tuple[UnitType, ...]
    log_initial: np.ndarray
    log_transition: np.ndarray
    log_emissions: np.ndarray
    log_duration_pmfs: np.ndarray | None = None
    mode: str = "explicit_duration_hsmm"
    memory_budget_bytes: int = 256*1024*1024
    max_work: int = 100000000
    extra_background: bool = False
    allow_phase_skips: bool = False
    feature_names: tuple[str, ...] = ()
    feature_means: np.ndarray | None = None
    feature_variances: np.ndarray | None = None
    feature_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if not isinstance(self.routine_name, str) or not self.routine_name.strip():
            raise ValueError("routine_name must be a nonempty string")
        phases = tuple(self.phase_names)
        if not phases or len(set(phases)) != len(phases) or any(not isinstance(p, str) or not p for p in phases):
            raise ValueError("routine must define at least one phase")
        object.__setattr__(self, "phase_names", phases)
        object.__setattr__(self, "primitive_catalog", tuple(self.primitive_catalog))

        if not isinstance(self.extra_background, bool) or not isinstance(self.allow_phase_skips, bool):
            raise ValueError("background/skip options must be boolean")
        Q = 1 + len(phases) + int(self.extra_background)
        K = len(self.primitive_catalog)
        if not K or any(not isinstance(t, UnitType) for t in self.primitive_catalog) or len({t.type_id for t in self.primitive_catalog}) != K:
            raise ValueError("primitive catalog must contain unique UnitType records")
        for value in (self.memory_budget_bytes, self.max_work):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("inference resource budgets must be positive integers")
        if 16*(Q+Q*Q+Q*K) > self.memory_budget_bytes:
            raise ValueError("staged parameter memory budget exceeded")

        log_init = frozen_array(self.log_initial, dtype=float)
        log_trans = frozen_array(self.log_transition, dtype=float)
        log_emiss = frozen_array(self.log_emissions, dtype=float)

        if log_init.shape != (Q,):
            raise ValueError(f"log_initial shape must be ({Q},), got {log_init.shape}")
        if log_trans.shape != (Q, Q):
            raise ValueError(f"log_transition shape must be ({Q}, {Q}), got {log_trans.shape}")
        if log_emiss.shape != (Q, K):
            raise ValueError(f"log_emissions shape must be ({Q}, {K}), got {log_emiss.shape}")
        validate_log_probabilities(log_init, axis=0, name="initial distribution")
        validate_log_probabilities(log_trans, name="transition distribution")
        validate_log_probabilities(log_emiss, name="categorical emissions")
        legal = np.zeros((Q, Q), dtype=bool)
        legal[:, 0] = True  # observed early deviation/exit can return to background
        legal[0, 1] = True
        for phase in range(1, len(phases)):
            legal[phase, phase+1] = True
        if len(phases) > 1:
            legal[len(phases), 1] = True
        if self.allow_phase_skips:
            for phase in range(1, len(phases)):
                legal[phase, phase+1:len(phases)+1] = True
        if self.extra_background:
            legal[0, -1] = True
            legal[-1, 1] = True
            legal[len(phases), -1] = True
        if self.mode == "staged_hmm":
            np.fill_diagonal(legal, True)
        if np.any(np.isfinite(log_trans) & ~legal):
            raise ValueError("transition violates declared ordered-stage topology")

        object.__setattr__(self, "log_initial", log_init)
        object.__setattr__(self, "log_transition", log_trans)
        object.__setattr__(self, "log_emissions", log_emiss)
        names = tuple(self.feature_names)
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("feature names must be unique nonempty strings")
        if (isinstance(self.feature_weight, bool) or not isinstance(self.feature_weight, Real)
                or not np.isfinite(self.feature_weight) or self.feature_weight < 0):
            raise ValueError("feature weight must be finite and nonnegative")
        if bool(names) != (self.feature_means is not None and self.feature_variances is not None):
            raise ValueError("Gaussian feature schema and parameters must be supplied together")
        if not names and (self.feature_means is not None or self.feature_variances is not None):
            raise ValueError("Gaussian parameters require feature names")
        if names:
            means = frozen_array(self.feature_means, dtype=float)
            variances = frozen_array(self.feature_variances, dtype=float)
            if (means.shape != (Q, len(names)) or variances.shape != means.shape
                    or not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances <= 0)):
                raise ValueError("Gaussian parameters must be finite Q by feature arrays with positive variances")
            if 32*means.size > self.memory_budget_bytes:
                raise ValueError("Gaussian parameter memory budget exceeded")
            object.__setattr__(self, "feature_means", means)
            object.__setattr__(self, "feature_variances", variances)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_weight", float(self.feature_weight))

        if self.mode not in {"explicit_duration_hsmm", "staged_hmm"}:
            raise ValueError("mode must be explicit_duration_hsmm or staged_hmm")

        if self.mode == "explicit_duration_hsmm":
            if self.log_duration_pmfs is None:
                raise ValueError("explicit_duration_hsmm mode requires log_duration_pmfs")
            if 16*np.asarray(self.log_duration_pmfs).size > self.memory_budget_bytes:
                raise ValueError("duration parameter memory budget exceeded")
            log_dur = frozen_array(self.log_duration_pmfs, dtype=float)
            if log_dur.ndim != 2 or log_dur.shape[0] != Q:
                raise ValueError(f"log_duration_pmfs must have shape ({Q}, D_max)")
            validate_log_probabilities(log_dur, name="duration PMFs")
            if np.any(~np.isneginf(np.diag(log_trans))):
                raise ValueError("HSMM segment self transitions are unsupported")
            object.__setattr__(self, "log_duration_pmfs", log_dur)
        parameter_bytes = 16*(Q+Q*Q+Q*K) + 32*Q*len(names)
        if self.log_duration_pmfs is not None:
            parameter_bytes += 16*np.asarray(self.log_duration_pmfs).size
        if parameter_bytes > self.memory_budget_bytes:
            raise ValueError("combined staged parameter memory budget exceeded")

    @classmethod
    def create_linear_routine(
        cls,
        model_id: str,
        routine_name: str,
        phase_names: tuple[str, ...],
        primitive_catalog: tuple[UnitType, ...],
        phase_emissions: dict[str, dict[str, float]],
        background_emissions: dict[str, float],
        *,
        phase_durations: dict[str, np.ndarray] | None = None,
        background_duration: np.ndarray | None = None,
        allow_immediate_reentry: bool = True,
        mode: str = "explicit_duration_hsmm",
        extra_background: bool = False,
        entry_probability: float = 0.1,
        early_exit_probability: float = 0.0,
        skip_probability: float = 0.0,
    ) -> StagedEpisodeModel:
        """Helper creating a canonical linear phase model (Background -> Phase 1 -> ... -> Phase P -> Exit/Reentry)."""
        if not phase_names or (len(phase_names) == 1 and allow_immediate_reentry):
            raise ValueError("one-stage immediate reentry is ambiguous; explicitly disable it")
        if not isinstance(extra_background, bool):
            raise ValueError("extra_background must be boolean")
        for p in (entry_probability, early_exit_probability, skip_probability):
            if isinstance(p, bool) or not isinstance(p, Real) or not np.isfinite(p) or p < 0 or p >= 1:
                raise ValueError("entry/exit/skip probabilities must be in [0,1)")
        if early_exit_probability + skip_probability >= 1:
            raise ValueError("early exit and skip probability must leave progression mass")
        if extra_background and entry_probability == 0:
            raise ValueError("background entry probability must be positive")
        Q = 1 + len(phase_names) + int(extra_background)
        K = len(primitive_catalog)
        prim_map = {u.type_id: i for i, u in enumerate(primitive_catalog)}
        if set(phase_emissions) != set(phase_names):
            raise ValueError("phase emission mappings must match declared phases")
        for table in [background_emissions, *phase_emissions.values()]:
            if (set(table)-set(prim_map) or any(not np.isfinite(p) or p < 0 or p > 1 for p in table.values())
                    or not np.isclose(sum(table.values()), 1., atol=1e-6, rtol=0)):
                raise ValueError("categorical emission mapping must be normalized over known primitive types")
        if phase_durations is not None and set(phase_durations)-set(phase_names):
            raise ValueError("duration mappings contain unknown phase names")
        for pmf in ([background_duration] if background_duration is not None else []) + list((phase_durations or {}).values()):
            a = np.asarray(pmf, dtype=float)
            if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) or not np.isclose(a.sum(), 1., atol=1e-6, rtol=0):
                raise ValueError("duration PMFs must be finite normalized nonnegative vectors")

        log_init = np.full(Q, float("-inf"), dtype=float)
        log_init[0] = np.log(0.5)  # background
        log_init[1] = np.log(0.5)  # or enter phase 1
        if extra_background:
            log_init[0], log_init[1] = np.log1p(-entry_probability), np.log(entry_probability)

        log_trans = np.full((Q, Q), float("-inf"), dtype=float)
        # Background -> Phase 1 or stay in background
        if mode == "staged_hmm":
            log_trans[0, 0] = np.log(0.9)
            log_trans[0, 1] = np.log(0.1)
        else:
            # HSMM segment transitions: no self-dwell in transition matrix (dwell is in duration PMF)
            log_trans[0, 1] = 0.0  # transitions to phase 1
        if extra_background:
            log_trans[0] = -np.inf
            log_trans[0, 1] = np.log(entry_probability)
            log_trans[0, -1] = np.log1p(-entry_probability)
            log_trans[-1, 0] = np.log1p(-entry_probability)
            log_trans[-1, 1] = np.log(entry_probability)

        # Phase progression
        for p in range(1, len(phase_names)):
            # phase p -> phase p+1
            log_trans[p, p + 1] = 0.0
            exit_mass = early_exit_probability
            skip_mass = skip_probability if p+2 <= len(phase_names) else 0.0
            log_trans[p, p+1] = np.log(1-exit_mass-skip_mass)
            if exit_mass:
                log_trans[p, 0] = np.log(exit_mass)
            if skip_mass:
                log_trans[p, p+2] = np.log(skip_mass)

        # Final phase P
        final_p = len(phase_names)
        if allow_immediate_reentry and len(phase_names) > 1:
            # Can reenter phase 1 (0.5) or exit to background (0.5)
            log_trans[final_p, 1] = np.log(0.5)
            log_trans[final_p, 0] = np.log(0.5)
        else:
            log_trans[final_p, 0] = 0.0

        # Emissions
        log_emiss = np.full((Q, K), float("-inf"), dtype=float)
        # State 0: background
        for prim_id, prob in background_emissions.items():
            if prim_id in prim_map and prob > 0:
                log_emiss[0, prim_map[prim_id]] = np.log(prob)

        # States 1..P
        for p_idx, p_name in enumerate(phase_names, start=1):
            p_emiss = phase_emissions.get(p_name, {})
            for prim_id, prob in p_emiss.items():
                if prim_id in prim_map and prob > 0:
                    log_emiss[p_idx, prim_map[prim_id]] = np.log(prob)
        if extra_background:
            log_emiss[-1] = log_emiss[0]

        log_dur_pmfs = None
        if mode == "explicit_duration_hsmm":
            max_d = 1
            if background_duration is not None:
                max_d = max(max_d, len(background_duration))
            if phase_durations is not None:
                for d_arr in phase_durations.values():
                    max_d = max(max_d, len(d_arr))

            dur_matrix = np.full((Q, max_d), float("-inf"), dtype=float)
            # Background
            if background_duration is not None:
                bg_d = np.asarray(background_duration, dtype=float)
                dur_matrix[0, :len(bg_d)] = np.where(bg_d > 0, np.log(np.where(bg_d > 0, bg_d, 1.)), -np.inf)
            else:
                dur_matrix[0, 0] = 0.0  # d=1 by default

            # Phases
            for p_idx, p_name in enumerate(phase_names, start=1):
                if phase_durations is not None and p_name in phase_durations:
                    p_d = np.asarray(phase_durations[p_name], dtype=float)
                    dur_matrix[p_idx, :len(p_d)] = np.where(p_d > 0, np.log(np.where(p_d > 0, p_d, 1.)), -np.inf)
                else:
                    dur_matrix[p_idx, 0] = 0.0

            log_dur_pmfs = dur_matrix
            if extra_background:
                dur_matrix[-1] = dur_matrix[0]

        return cls(
            model_id=model_id,
            routine_name=routine_name,
            phase_names=phase_names,
            primitive_catalog=primitive_catalog,
            log_initial=log_init,
            log_transition=log_trans,
            log_emissions=log_emiss,
            log_duration_pmfs=log_dur_pmfs,
            mode=mode,
            extra_background=extra_background,
            allow_phase_skips=skip_probability > 0,
        )

    def _feature_rows(self, stream, features):
        if not self.feature_names:
            if features is not None:
                raise ValueError("unused feature input: model has no Gaussian feature schema")
            return None
        values = np.asarray(features, dtype=float)
        if values.shape != (len(stream.occurrences), len(self.feature_names)) or not np.isfinite(values).all():
            raise ValueError("features must be finite and aligned to every supplied occurrence")
        if values.nbytes > self.memory_budget_bytes:
            raise ValueError("feature input exceeds memory budget")
        return {occ.occurrence_id: values[i] for i, occ in enumerate(stream.occurrences)}

    def _observation_emissions(self, block, feature_rows):
        codes = {u.type_id: i for i, u in enumerate(self.primitive_catalog)}
        if any(o.type_id not in codes for o in block):
            raise ValueError("primitive identity is outside the fitted catalogue")
        q, f, t = len(self.log_initial), len(self.feature_names), len(block)
        if 32*t*q + 24*t*f + 32*q*f > self.memory_budget_bytes:
            raise ValueError("staged emission working-array memory budget exceeded")
        result = self.log_emissions[:, [codes[o.type_id] for o in block]].T.copy()
        if feature_rows is not None:
            values = np.asarray([feature_rows[o.occurrence_id] for o in block])
            result += self.feature_weight * gaussian_log_emissions(values, self.feature_means, self.feature_variances)
        return result

    def decode_stream(
        self,
        stream: UnitStream,
        *,
        right_censored: bool = True,
        features: np.ndarray | None = None,
    ) -> BehaviorResult:
        """Decode primitive UnitStream into staged episode occurrences and rich envelope.

        Parameters
        ----------
        stream : UnitStream
            Input primitive stream. Must be chronological and exclusive.
        right_censored : bool
            Whether terminal boundary segments use survival mass.

        Returns
        -------
        BehaviorResult
            Unified envelope with episode UnitStream, PhaseRecord sidecars, and CompletionEvidence.
        """
        require_compatible(stream, StreamRequirements(levels=("primitive",), exclusive=True))

        feature_rows = self._feature_rows(stream, features)
        blocks = occurrence_blocks(stream)

        episode_occurrences: list[UnitOccurrence] = []
        phase_records: list[PhaseRecord] = []
        completion_records: list[CompletionEvidence] = []
        episode_counter = 0
        block_likelihoods = []

        final_phase_idx = len(self.phase_names)

        for block in blocks:
            T_block = len(block)
            if T_block == 0:
                continue

            # Build block emission matrix
            obs_log_emiss = self._observation_emissions(block, feature_rows)

            if self.mode == "explicit_duration_hsmm":
                hsmm_res = viterbi_hsmm(
                    self.log_initial,
                    self.log_transition,
                    obs_log_emiss,
                    self.log_duration_pmfs,
                    right_censored=right_censored,
                    memory_budget_bytes=self.memory_budget_bytes,
                    max_work=self.max_work,
                )
                if np.isneginf(hsmm_res.best_score) or np.isneginf(hsmm_res.log_likelihood):
                    raise ValueError(
                        f"Impossible observation sequence under staged HSMM {self.model_id}: "
                        "model assigns zero probability mass to input observations."
                    )
                segments = hsmm_res.segments
                block_likelihoods.append(hsmm_res.log_likelihood)
            else:
                states, best_sc = viterbi_hmm(self.log_initial, self.log_transition, obs_log_emiss,
                                            memory_budget_bytes=self.memory_budget_bytes, max_work=self.max_work)
                if np.isneginf(best_sc) or (len(states) > 0 and (states == -1).all()):
                    raise ValueError(
                        f"Impossible observation sequence under staged HMM {self.model_id}: "
                        "model assigns zero probability mass to input observations."
                    )
                # Convert states to segments
                segments = []
                from fpcode.statistics.sequence_inference import forward_backward_hmm
                block_likelihoods.append(forward_backward_hmm(
                    self.log_initial, self.log_transition, obs_log_emiss,
                    memory_budget_bytes=self.memory_budget_bytes, max_work=self.max_work).log_likelihood)
                if len(states) > 0:
                    diffs = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
                    for s_idx, e_idx in zip(diffs[:-1], diffs[1:]):
                        from fpcode.statistics.sequence_inference import HSMMSegment
                        state = states[s_idx]
                        weight = float(obs_log_emiss[s_idx:e_idx, state].sum())
                        weight += self.log_initial[state] if s_idx == 0 else self.log_transition[states[s_idx-1], state]
                        if e_idx-s_idx > 1:
                            weight += (e_idx-s_idx-1)*self.log_transition[state, state]
                        segments.append(HSMMSegment(state, s_idx, e_idx, e_idx-s_idx, weight))

            # Group segments into episode invocations
            # Nonzero states (1..P) belong to the routine. State 0 is background.
            # Distinct invocations are bounded by:
            # 1. State 0 (background)
            # 2. Transition from final_phase (P) -> phase 1 (immediate reentry)
            curr_invocation_segs: list = []

            for seg_idx, seg in enumerate(segments):
                if seg.state == 0 or (self.extra_background and seg.state == self.log_emissions.shape[0]-1):
                    if curr_invocation_segs:
                        self._finish_invocation(
                            curr_invocation_segs,
                            block,
                            episode_occurrences,
                            phase_records,
                            completion_records,
                            final_phase_idx,
                            episode_counter,
                            observed_exit=True,
                            stream=stream,
                        )
                        episode_counter += 1
                        curr_invocation_segs = []
                else:
                    # Nonzero phase state
                    if curr_invocation_segs:
                        prev_seg = curr_invocation_segs[-1]
                        # Immediate reentry: final phase -> phase 1 starts a new invocation!
                        if prev_seg.state == final_phase_idx and seg.state == 1:
                            self._finish_invocation(
                                curr_invocation_segs,
                                block,
                                episode_occurrences,
                                phase_records,
                                completion_records,
                                final_phase_idx,
                                episode_counter,
                                observed_exit=True,
                                stream=stream,
                            )
                            episode_counter += 1
                            curr_invocation_segs = []
                    curr_invocation_segs.append(seg)

            # Block ending: if invocation still open, it is right-censored!
            if curr_invocation_segs:
                self._finish_invocation(
                    curr_invocation_segs,
                    block,
                    episode_occurrences,
                    phase_records,
                    completion_records,
                    final_phase_idx,
                    episode_counter,
                    observed_exit=False,  # terminated at block boundary!
                    stream=stream,
                )
                episode_counter += 1

        catalog = (UnitType(self.routine_name, f"staged episode {self.routine_name}"),)
        ep_stream = UnitStream(
            session=stream.session,
            level="episode",
            model_id=self.model_id,
            clock=stream.clock,
            fps=stream.fps,
            valid_intervals=stream.valid_intervals,
            catalog=catalog,
            occurrences=tuple(episode_occurrences),
        )

        return BehaviorResult(
            stream=ep_stream,
            phases=tuple(phase_records),
            completion=tuple(completion_records),
            source_streams=(stream,),
            diagnostics={"log_likelihood": float(sum(block_likelihoods)),
                         "block_log_likelihoods": tuple(block_likelihoods),
                         "observation_coordinate": "primitive_event",
                         "start_assumption": "conditional initial distribution; boundary entries potentially censored",
                         "score_semantics": "selected-path local log weight"},
        )

    def predict(
        self,
        stream: UnitStream,
        *,
        right_censored: bool = True,
        features: np.ndarray | None = None,
    ) -> BehaviorResult:
        """Alias for decode_stream."""
        return self.decode_stream(stream, right_censored=right_censored, features=features)

    def score(
        self,
        stream: UnitStream,
        *,
        right_censored: bool = True,
        features: np.ndarray | None = None,
    ) -> float:
        """Compute the total log likelihood of the stream under this staged model.

        Parameters
        ----------
        stream : UnitStream
            Input primitive stream.
        right_censored : bool
            Whether terminal boundary segments use survival mass.

        Returns
        -------
        float
            Summed log likelihood across independent valid blocks.
        """
        require_compatible(stream, StreamRequirements(levels=("primitive",), exclusive=True))
        feature_rows = self._feature_rows(stream, features)
        blocks = occurrence_blocks(stream)

        total_ll = 0.0
        for block in blocks:
            T_block = len(block)
            if T_block == 0:
                continue
            obs_log_emiss = self._observation_emissions(block, feature_rows)

            if self.mode == "explicit_duration_hsmm":
                from fpcode.statistics.sequence_inference import forward_hsmm
                ll, _ = forward_hsmm(
                    self.log_initial,
                    self.log_transition,
                    obs_log_emiss,
                    self.log_duration_pmfs,
                    right_censored=right_censored,
                    memory_budget_bytes=self.memory_budget_bytes,
                    max_work=self.max_work,
                )
            else:
                from fpcode.statistics.sequence_inference import forward_backward_hmm
                fb = forward_backward_hmm(self.log_initial, self.log_transition, obs_log_emiss,
                                          memory_budget_bytes=self.memory_budget_bytes, max_work=self.max_work)
                ll = fb.log_likelihood

            if np.isneginf(ll):
                return float("-inf")
            total_ll += ll

        return float(total_ll)

    def _finish_invocation(
        self,
        segs: list,
        block: list[UnitOccurrence],
        episode_occurrences: list[UnitOccurrence],
        phase_records: list[PhaseRecord],
        completion_records: list[CompletionEvidence],
        final_phase_idx: int,
        counter: int,
        *,
        observed_exit: bool,
        stream: UnitStream,
    ) -> None:
        first_step = segs[0].start
        last_step = segs[-1].stop
        start_frame = block[first_step].interval.start
        stop_frame = block[last_step - 1].interval.stop

        occ_id = f"ep_{counter}"
        episode_occurrences.append(
            UnitOccurrence(
                occurrence_id=occ_id,
                type_id=self.routine_name,
                interval=FrameSpan(start_frame, stop_frame),
                score=float(sum(seg.score for seg in segs)),
                score_kind="model_score",
            )
        )

        final_reached = bool(any(seg.state == final_phase_idx for seg in segs))
        first_reached = bool(segs[0].state == 1)

        observed_entry = bool(first_reached and first_step > 0)
        left_censored = not observed_entry
        right_censored = bool(not observed_exit)

        if right_censored or left_censored:
            status = "censored"
        elif final_reached and observed_exit:
            status = "completed"
        else:
            status = "deviated"

        completion_records.append(
            CompletionEvidence(
                occurrence_id=occ_id,
                observed_entry=observed_entry,
                observed_exit=observed_exit,
                final_phase_reached=final_reached,
                status=status,
                left_censored=left_censored,
                right_censored=right_censored,
                evidence_kind="selected_model_path",
            )
        )

        for p_order, seg in enumerate(segs):
            phase_name = self.phase_names[seg.state - 1]
            p_start = block[seg.start].interval.start
            p_stop = block[seg.stop - 1].interval.stop
            phase_records.append(
                PhaseRecord(
                    parent_occurrence_id=occ_id,
                    phase_id=phase_name,
                    order=p_order,
                    interval=FrameSpan(p_start, p_stop),
                    source_reference=ScopedOccurrenceReference(
                        session=stream.session,
                        model_id=stream.model_id,
                        occurrence_id=block[seg.start].occurrence_id,
                    ),
                    evidence_kind="hsmm_segment" if self.mode == "explicit_duration_hsmm" else "hmm_state",
                )
            )


def gaussian_log_emissions(values, means, variances):
    """Diagonal Gaussian density with one bounded observation-by-feature scratch.

    State batches avoid both an observations×states×features broadcast and the
    catastrophic cancellation of expanding squared distances around large means.
    """
    values, means, variances = (np.asarray(a, dtype=float) for a in (values, means, variances))
    if (values.ndim != 2 or means.ndim != 2 or variances.shape != means.shape
            or values.shape[1] != means.shape[1] or not all(np.isfinite(a).all() for a in (values, means, variances))
            or np.any(variances <= 0)):
        raise ValueError("Gaussian feature schemas must agree and have finite positive variances")
    result = np.empty((len(values), len(means)))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalizer = np.sum(np.log(variances)+np.log(2*np.pi), axis=1)
            for state in range(len(means)):
                residual = (values-means[state])/np.sqrt(variances[state])
                result[:, state] = -.5*(np.einsum("ij,ij->i", residual, residual)+normalizer[state])
    except FloatingPointError as exc:
        raise ValueError("Gaussian density exceeds supported float64 range") from exc
    if not np.isfinite(result).all():
        raise ValueError("Gaussian density exceeds supported float64 range")
    return result


def fit_staged_model(
    initial_model: StagedEpisodeModel,
    training_streams: Iterable[UnitStream],
    *,
    model_id: str,
    source_model_id: str,
    max_iterations: int = 20,
    seed: int = 0,
    initialization: str = "configured",
    pseudocount: float = 0.5,
    tolerance: float = 1e-6,
    max_observations: int = 100000,
    max_total_work: int = 100000000,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    right_censored: bool = True,
    training_features: Iterable[np.ndarray] | None = None,
    learn_durations: bool = False,
    variance_floor: float = 0.05,
) -> StagedFitResult:
    """Hard-EM reestimation of supported categorical and optional Gaussian evidence.

    Only explicitly supplied primitive streams enter training. Each native valid
    block is an independent sequence; no transitions cross gaps or recordings.
    The initial distribution and topology stay frozen. Duration PMFs stay frozen
    by default; ``learn_durations=True`` updates their existing structural support.
    Duration is measured in primitive events, never inferred physical seconds.
    Boundary segments contribute observations and observed internal transitions.
    The first and right-censored last segment lengths are never counted as exact
    durations. Gaussian means/variances use assigned observations, with the stated
    variance floor; supplied features must align to every occurrence.

    The E-step uses the existing HSMM Viterbi segment decoder. The M-step adds
    ``pseudocount * configured_probability`` to each supported categorical or
    transition count. Structural zeros remain zero and unvisited states retain
    their configured probabilities. This optimizes a regularized selected-path
    objective; marginal likelihood need not increase. It is neither Baum-Welch
    training nor topology/routine discovery. ``initialization='dirichlet'`` mixes
    configured rows equally with seeded Dirichlet(1) rows on the same support.

    Caps apply to all supplied observations and worst-case inference work for
    the requested iteration count; data are never silently truncated. The memory
    estimate budgets fit-owned arrays and inference scratch, not caller-owned
    streams or an OS-enforced RSS limit. All numerics use CPU float64.
    """
    started = perf_counter()
    if not isinstance(initial_model, StagedEpisodeModel) or initial_model.mode != "explicit_duration_hsmm":
        raise ValueError("fitting requires a configured explicit_duration_hsmm model")
    if not isinstance(model_id, str) or not model_id.strip() or model_id == initial_model.model_id:
        raise ValueError("fitted model_id must be nonempty and differ from initial model identity")
    if not isinstance(source_model_id, str) or not source_model_id.strip():
        raise ValueError("source_model_id must explicitly identify the training primitive model")
    for value, name in ((max_iterations, "max_iterations"), (max_observations, "max_observations"),
                        (max_total_work, "max_total_work"), (memory_budget_bytes, "memory_budget_bytes")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if initialization not in {"configured", "dirichlet"}:
        raise ValueError("initialization must be configured or dirichlet")
    if (isinstance(pseudocount, bool) or not isinstance(pseudocount, Real)
            or not np.isfinite(pseudocount) or pseudocount <= 0):
        raise ValueError("pseudocount must be finite and positive")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    if not isinstance(right_censored, bool):
        raise ValueError("right_censored must explicitly be boolean")
    if (not isinstance(learn_durations, bool) or isinstance(variance_floor, bool)
            or not isinstance(variance_floor, Real) or not np.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("duration learning must be boolean and variance floor finite positive")

    streams = tuple(training_streams)
    if not streams:
        raise ValueError("explicit nonempty training streams are required")
    supplied_features = tuple(training_features) if training_features is not None else (None,) * len(streams)
    if len(supplied_features) != len(streams):
        raise ValueError("one aligned feature matrix is required per training stream")
    feature_rows = [initial_model._feature_rows(s, x) for s, x in zip(streams, supplied_features)]
    catalog = {unit.type_id: unit for unit in initial_model.primitive_catalog}
    codes = {unit.type_id: i for i, unit in enumerate(initial_model.primitive_catalog)}
    seen = set()
    clock = None
    observation_count = 0
    block_count = 0
    largest_block = 0
    stream_blocks = []
    for stream in streams:
        require_compatible(stream, StreamRequirements(levels=("primitive",), exclusive=True))
        if stream.model_id != source_model_id:
            raise ValueError("training primitive model identity differs from source_model_id")
        if {unit.type_id: unit for unit in stream.catalog} != catalog:
            raise ValueError("training primitive catalogue differs from configured model catalogue")
        if clock is None:
            clock = stream.clock
        elif stream.clock != clock:
            raise ValueError("training clock identity differs; explicit alignment is required")
        if stream.session in seen:
            raise ValueError("duplicate training recording identity would repeat its weight")
        seen.add(stream.session)
        observation_count += len(stream.occurrences)
        if observation_count > max_observations:
            raise ValueError("staged fit observation budget exceeded; select training spans explicitly")
        blocks = occurrence_blocks(stream)
        block_count += len(blocks)
        largest_block = max(largest_block, max((len(block) for block in blocks), default=0))
        stream_blocks.append(blocks)
    if not observation_count:
        raise ValueError("training streams contain no primitive observations")

    q, k = initial_model.log_emissions.shape
    d = initial_model.log_duration_pmfs.shape[1]
    work_per_pass = 2 * observation_count * q * q * d  # Viterbi plus marginal forward recursion
    required_work = (max_iterations + 1) * work_per_pass
    if required_work > max_total_work:
        raise ValueError("staged fit total work budget exceeded")
    f = len(initial_model.feature_names)
    duration_memory = 16*q*d + 64*min(d, largest_block)*q + 32*min(d, largest_block)
    inference_memory = 96 * (largest_block + 1) * q + 16 * q * q + 256 * largest_block + duration_memory
    # All training block feature copies, per-state sums/squares, Gaussian
    # parameters and the largest block's emission/quadratic scratch coexist.
    owned_memory = (24 * observation_count + 32 * largest_block * q
                    + 64 * (q * k + q * q + q * d) + 16*observation_count*f
                    + 32*largest_block*f + 96*q*f)
    estimated_memory = inference_memory + owned_memory
    if estimated_memory > memory_budget_bytes:
        raise ValueError("staged fit memory budget exceeded")
    feature_blocks = tuple(np.asarray([rows[event.occurrence_id] for event in block]) if rows is not None else None
                           for recording_blocks, rows in zip(stream_blocks, feature_rows) for block in recording_blocks)
    blocks = tuple(np.fromiter((codes[event.type_id] for event in block), dtype=np.int64, count=len(block))
                   for recording_blocks in stream_blocks for block in recording_blocks)
    del stream_blocks

    prior_emissions = pseudocount * np.exp(initial_model.log_emissions)
    prior_transitions = pseudocount * np.exp(initial_model.log_transition)
    prior_durations = pseudocount * np.exp(initial_model.log_duration_pmfs)

    def initialized_rows(log_probabilities):
        if initialization == "configured":
            return log_probabilities
        probabilities = np.exp(log_probabilities)
        result = np.full_like(log_probabilities, -np.inf)
        for row, values in enumerate(probabilities):
            supported = np.isfinite(log_probabilities[row])
            sampled = rng.dirichlet(np.ones(supported.sum()))
            result[row, supported] = np.log(0.5 * values[supported] + 0.5 * sampled)
        return result

    rng = np.random.default_rng(seed)
    model = replace(initial_model, model_id=model_id,
                    log_emissions=initialized_rows(initial_model.log_emissions),
                    log_transition=initialized_rows(initial_model.log_transition),
                    memory_budget_bytes=memory_budget_bytes)

    def evaluate(candidate):
        emissions = np.zeros_like(candidate.log_emissions)
        transitions = np.zeros_like(candidate.log_transition)
        durations = np.zeros_like(candidate.log_duration_pmfs)
        f = len(candidate.feature_names)
        sums, squares, counts = np.zeros((q, f)), np.zeros((q, f)), np.zeros(q)
        likelihood = path_score = 0.0
        terminal_censored = initial_boundary = segments_count = 0
        for observations, values in zip(blocks, feature_blocks):
            observation_emissions = candidate.log_emissions[:, observations].T.copy()
            if values is not None:
                observation_emissions += candidate.feature_weight * gaussian_log_emissions(values, candidate.feature_means, candidate.feature_variances)
            result = viterbi_hsmm(
                candidate.log_initial, candidate.log_transition,
                observation_emissions, candidate.log_duration_pmfs,
                right_censored=right_censored, memory_budget_bytes=memory_budget_bytes,
                max_work=candidate.max_work,
            )
            if not np.isfinite(result.best_score) or not np.isfinite(result.log_likelihood):
                raise ValueError("impossible training block under staged HSMM structural support")
            likelihood += result.log_likelihood
            path_score += result.best_score
            np.add.at(emissions, (result.map_states, observations), 1.0)
            if values is not None:
                # Center around the current state mean before accumulating, so
                # small variances on large coordinate offsets remain measurable.
                residuals = values-candidate.feature_means[result.map_states]
                np.add.at(sums, result.map_states, residuals)
                np.add.at(squares, result.map_states, residuals**2)
                np.add.at(counts, result.map_states, 1)
            previous = None
            for segment in result.segments:
                if segment.start > 0 and not (right_censored and segment.stop == len(observations)):
                    durations[segment.state, segment.duration-1] += 1
                if previous is not None:
                    transitions[previous, segment.state] += 1.0
                previous = segment.state
            initial_boundary += 1
            terminal_censored += int(right_censored)
            segments_count += len(result.segments)
        supported_emissions = prior_emissions > 0
        supported_transitions = prior_transitions > 0
        regularizer = float(np.sum(prior_emissions[supported_emissions]
                                   * candidate.log_emissions[supported_emissions]))
        regularizer += float(np.sum(prior_transitions[supported_transitions]
                                    * candidate.log_transition[supported_transitions]))
        if learn_durations:
            supported = prior_durations > 0
            regularizer += float(np.sum(prior_durations[supported] * candidate.log_duration_pmfs[supported]))
        return likelihood, path_score, path_score + regularizer, emissions, transitions, (
            segments_count, initial_boundary, terminal_censored), (sums, squares, counts), durations

    def reestimated_rows(counts, configured):
        # Log arithmetic preserves finite support even when configured mass is
        # below float64's linear-domain range. Unsupported entries stay -inf.
        log_counts = np.full_like(counts, -np.inf)
        observed = counts > 0
        log_counts[observed] = np.log(counts[observed])
        log_mass = np.logaddexp(log_counts, np.log(pseudocount) + configured)
        return log_mass - logsumexp(log_mass, axis=1, keepdims=True)

    evaluation = evaluate(model)
    likelihood_history = [evaluation[0]]
    path_history = [evaluation[1]]
    objective_history = [evaluation[2]]
    converged = False
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        reestimation_emissions, reestimation_transitions = evaluation[3:5]
        updates = {}
        if model.feature_names:
            sums, squares, counts = evaluation[6]
            visited = counts > 0
            means, variances = model.feature_means.copy(), model.feature_variances.copy()
            residual_means = sums[visited]/counts[visited, None]
            means[visited] += residual_means
            variances[visited] = np.maximum(squares[visited]/counts[visited, None]-residual_means**2, variance_floor)
            updates.update(feature_means=means, feature_variances=variances)
        if learn_durations:
            updates["log_duration_pmfs"] = reestimated_rows(evaluation[7], initial_model.log_duration_pmfs)
        candidate = replace(model,
                            log_emissions=reestimated_rows(evaluation[3], initial_model.log_emissions),
                            log_transition=reestimated_rows(evaluation[4], initial_model.log_transition), **updates)
        updated = evaluate(candidate)
        # Compare the newly decoded, reestimated objective, not a stale E-step.
        improvement = updated[2] - evaluation[2]
        model, evaluation = candidate, updated
        completed_iterations = iteration
        likelihood_history.append(updated[0])
        path_history.append(updated[1])
        objective_history.append(updated[2])
        if abs(improvement) <= tolerance * max(1.0, abs(objective_history[-2])):
            converged = True
            break

    diagnostics = {
        "algorithm": "categorical_viterbi_hard_em_frozen_durations",
        "learned_parameters": ("categorical_emissions", "allowed_segment_transitions"),
        "frozen_parameters": ("initial_distribution", "topology", "duration_pmfs"),
        "observation_coordinate": "primitive_event",
        "duration_coordinate": "primitive_event_count",
        "boundary_duration_policy": "all durations frozen; no observed boundary length counted as exact",
        "start_assumption": "conditional configured initial distribution; first segment may be left censored",
        "right_censored": right_censored,
        "initial_model_id": initial_model.model_id,
        "source_model_id": source_model_id,
        "model_id": model_id,
        "seed": int(seed),
        "initialization": initialization,
        "dirichlet_mix_weight": 0.5 if initialization == "dirichlet" else 0.0,
        "pseudocount": float(pseudocount),
        "prior_semantics": "per-row mass times configured probabilities on structural support",
        "tolerance": float(tolerance),
        "requested_iterations": int(max_iterations),
        "completed_iterations": completed_iterations,
        "converged": converged,
        "stop_reason": "regularized_path_objective_tolerance" if converged else "iteration_limit",
        "initial_log_likelihood": likelihood_history[0],
        "updated_log_likelihood": likelihood_history[-1],
        "initial_selected_path_score": path_history[0],
        "updated_selected_path_score": path_history[-1],
        "log_likelihood_history": tuple(likelihood_history),
        "selected_path_score_history": tuple(path_history),
        "regularized_path_objective_history": tuple(objective_history),
        "objective_semantics": "sum block MAP log weights plus configured-probability log prior",
        "marginal_likelihood_monotonicity": "not guaranteed by hard-EM",
        "training_recordings": tuple((stream.session.recording, stream.session.subject, stream.session.study)
                                     for stream in streams),
        "training_rates": tuple(stream.fps for stream in streams),
        "training_clock": clock,
        "training_observations": observation_count,
        "training_blocks": block_count,
        "selected_segments": evaluation[5][0],
        "initial_boundary_segments": evaluation[5][1],
        "terminal_censored_segments": evaluation[5][2],
        "duration_update_count": 0,
        "emission_observation_counts": tuple(tuple(row) for row in evaluation[3]),
        "segment_transition_counts": tuple(tuple(row) for row in evaluation[4]),
        "reestimation_emission_counts": tuple(tuple(row) for row in reestimation_emissions),
        "reestimation_transition_counts": tuple(tuple(row) for row in reestimation_transitions),
        "count_semantics": "final selected-path counts; reestimation counts identify the last M-step",
        "state_emission_prior_mass": tuple(prior_emissions.sum(axis=1)),
        "state_transition_prior_mass": tuple(prior_transitions.sum(axis=1)),
        "dtype": "float64",
        "max_observations": int(max_observations),
        "max_total_work": int(max_total_work),
        "max_block_work": int(model.max_work),
        "estimated_requested_work": required_work,
        "estimated_executed_work": (completed_iterations + 1) * work_per_pass,
        "memory_budget_bytes": int(memory_budget_bytes),
        "estimated_peak_array_bytes": estimated_memory,
        "elapsed_seconds": perf_counter() - started,
    }
    if model.feature_names or learn_durations:
        diagnostics.update(
            algorithm="mixed_viterbi_hard_em",
            learned_parameters=("categorical_emissions", "allowed_segment_transitions")
                + (("Gaussian_means", "Gaussian_variances") if model.feature_names else ())
                + (("duration_pmfs",) if learn_durations else ()),
            frozen_parameters=("initial_distribution", "topology") + (() if learn_durations else ("duration_pmfs",)),
            feature_names=model.feature_names, feature_weight=model.feature_weight,
            variance_floor=variance_floor, extra_background=model.extra_background,
            allow_phase_skips=model.allow_phase_skips,
            duration_update_count=completed_iterations if learn_durations else 0,
            interior_duration_counts=tuple(tuple(row) for row in evaluation[7]),
            boundary_duration_policy="exclude first and right-censored terminal segments from duration reestimation"
                if learn_durations else "all duration PMFs frozen",
            objective_semantics="MAP path plus categorical/transition/duration priors; Gaussian MLE with variance floor",
            monotonicity="not guaranteed for marginal likelihood or variance-floor updates",
        )
    return StagedFitResult(model, diagnostics)


__all__ = [
    "IMPLEMENTATION_STATUS",
    "StagedEpisodeModel",
    "StagedFitResult",
    "fit_staged_model",
    "gaussian_log_emissions",
]
