"""Probabilistic staged episode HMM/HSMM alternative for behavioural sequences.

Implements configured categorical staged episode models with background, ordered
phase topologies, explicit duration PMFs, and segment-based HSMM inference.
Preserves segment traceback across immediate same-routine reentries, handles
boundary censoring honestly, and emits native-frame episodes with rich phase
and completion evidence envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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
    validate_log_probabilities,
    viterbi_hmm,
    viterbi_hsmm,
)

IMPLEMENTATION_STATUS = "implemented"


@dataclass(frozen=True)
class StagedEpisodeModel:
    """Configured categorical staged episode detector with explicit duration PMF support.

    States
    ------
    0 : Background / non-episode state
    1..P : Ordered phases of the target routine (e.g. 1=entry, P=final phase)
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

        Q = 1 + len(phases)
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
        for phase in range(1, Q-1):
            legal[phase, phase+1] = True
        if Q > 2:
            legal[-1, 1] = True
        if self.mode == "staged_hmm":
            np.fill_diagonal(legal, True)
        if np.any(np.isfinite(log_trans) & ~legal):
            raise ValueError("transition violates declared ordered-stage topology")

        object.__setattr__(self, "log_initial", log_init)
        object.__setattr__(self, "log_transition", log_trans)
        object.__setattr__(self, "log_emissions", log_emiss)

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
    ) -> StagedEpisodeModel:
        """Helper creating a canonical linear phase model (Background -> Phase 1 -> ... -> Phase P -> Exit/Reentry)."""
        if not phase_names or (len(phase_names) == 1 and allow_immediate_reentry):
            raise ValueError("one-stage immediate reentry is ambiguous; explicitly disable it")
        Q = 1 + len(phase_names)
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

        log_trans = np.full((Q, Q), float("-inf"), dtype=float)
        # Background -> Phase 1 or stay in background
        if mode == "staged_hmm":
            log_trans[0, 0] = np.log(0.9)
            log_trans[0, 1] = np.log(0.1)
        else:
            # HSMM segment transitions: no self-dwell in transition matrix (dwell is in duration PMF)
            log_trans[0, 1] = 0.0  # transitions to phase 1

        # Phase progression
        for p in range(1, len(phase_names)):
            # phase p -> phase p+1
            log_trans[p, p + 1] = 0.0

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
                dur_matrix[0, :len(bg_d)] = np.where(bg_d > 0, np.log(bg_d), float("-inf"))
            else:
                dur_matrix[0, 0] = 0.0  # d=1 by default

            # Phases
            for p_idx, p_name in enumerate(phase_names, start=1):
                if phase_durations is not None and p_name in phase_durations:
                    p_d = np.asarray(phase_durations[p_name], dtype=float)
                    dur_matrix[p_idx, :len(p_d)] = np.where(p_d > 0, np.log(p_d), float("-inf"))
                else:
                    dur_matrix[p_idx, 0] = 0.0

            log_dur_pmfs = dur_matrix

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
        )

    def decode_stream(
        self,
        stream: UnitStream,
        *,
        right_censored: bool = True,
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

        prim_map = {u.type_id: i for i, u in enumerate(self.primitive_catalog)}
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
            obs_log_emiss = np.full((T_block, 1 + len(self.phase_names)), float("-inf"), dtype=float)
            for t_idx, occ in enumerate(block):
                if occ.type_id in prim_map:
                    k_idx = prim_map[occ.type_id]
                    obs_log_emiss[t_idx] = self.log_emissions[:, k_idx]

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
                if seg.state == 0:
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
    ) -> BehaviorResult:
        """Alias for decode_stream."""
        return self.decode_stream(stream, right_censored=right_censored)

    def score(
        self,
        stream: UnitStream,
        *,
        right_censored: bool = True,
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
        prim_map = {u.type_id: i for i, u in enumerate(self.primitive_catalog)}
        blocks = occurrence_blocks(stream)

        total_ll = 0.0
        for block in blocks:
            T_block = len(block)
            if T_block == 0:
                continue
            obs_log_emiss = np.full((T_block, 1 + len(self.phase_names)), float("-inf"), dtype=float)
            for t_idx, occ in enumerate(block):
                if occ.type_id in prim_map:
                    k_idx = prim_map[occ.type_id]
                    obs_log_emiss[t_idx] = self.log_emissions[:, k_idx]

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


__all__ = [
    "IMPLEMENTATION_STATUS",
    "StagedEpisodeModel",
]
