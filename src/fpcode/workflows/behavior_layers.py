"""Bounded frozen-method and synthetic behavioural abstraction comparisons.

Composes observation features, primitive segmentation, alternative episode
detectors (staged HSMM, grammar, exact chains, seqNMF), assembly composition,
optional event-context models, and common execution/organisation analyses.
No implicit fitting on import; runs deterministically from declared parameters.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from fpcode.behavior.analysis.execution import (
    aggregate_execution_metrics,
    measure_occurrence_execution,
    measure_onset_profiles,
    measure_phase_profiles,
)
from fpcode.behavior.analysis.unit_views import measure_unit_organisation
from fpcode.behavior.assemblies.grammar import GrammarAssemblyComposer
from fpcode.behavior.context.models import EventContextHMM
from fpcode.behavior.episodes.chains import exact_occurrences
from fpcode.behavior.episodes.grammar import GrammarEpisodeDetector
from fpcode.behavior.episodes.seqnmf import SeqNMF, activation_events
from fpcode.behavior.episodes.staged import StagedEpisodeModel
from fpcode.behavior.observations.representations import extract_associated_trajectory_features
from fpcode.behavior.primitives.resolutions import remap_states
from fpcode.behavior.primitives.segmentation import CentroidPrimitiveSegmenter
from fpcode.behavior.unit_adapters import (
    from_bout_windows,
    from_frame_labels,
    primitive_bout_table,
)
from fpcode.contracts.behavior_results import BehaviorResult, FeatureAssociation
from fpcode.contracts.behavior_units import UnitStream, UnitType
from fpcode.contracts.composition import GrammarRule, GrammarSpecification, SequenceRule, TerminalMatch
from fpcode.contracts.pose import PoseSeries
from fpcode.contracts.sessions import SessionKey
from fpcode.contracts.timelines import TimeAxis
from fpcode.workflows.sequence_atlas import one_hot_segments

SESSION = SessionKey("synthetic_rec", "synthetic_mouse", "behavior_layers_study")


def compare_frozen_episode_methods(
    primitives: UnitStream,
    features: FeatureAssociation,
    *,
    exact_motifs,
    seqnmf_templates,
    seqnmf_type_ids,
    seqnmf_thresholds,
    seqnmf_penalty=.01,
    seqnmf_inference_iterations=35,
    grammar=None,
    staged_models=(),
    max_bouts=5000,
    memory_budget_bytes=256 * 1024 * 1024,
):
    """Compare frozen episode choices on one bounded, recording-scoped stream.

    Exact motif tokens are type identities. seqNMF channel identities must exactly
    match the primitive catalogue order; no vocabulary/order is inferred. Only
    activations are inferred, never templates/thresholds. Features accompany all
    results without influencing the detectors. Overlapping alternatives remain
    in each raw output; the caller explicitly chooses its organisation view.
    """
    for value in (max_bouts, memory_budget_bytes, seqnmf_inference_iterations):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("workflow resource budgets and iterations must be positive integers")
    if not isinstance(primitives, UnitStream) or not isinstance(features, FeatureAssociation):
        raise ValueError("explicit primitive stream and associated features are required")
    if features.session != primitives.session:
        raise ValueError("feature recording does not match primitive recording")
    if len(primitives.occurrences) > max_bouts:
        raise ValueError("pilot bout budget exceeded; select an explicit smaller native interval")
    # The result contract verifies recording and clock before any inference;
    # native feature alignment is checked by the shared measurement consumers.
    lower = BehaviorResult(primitives, features=features)
    bouts, type_ids = primitive_bout_table(primitives)
    if tuple(seqnmf_type_ids) != type_ids:
        raise ValueError("seqNMF vocabulary/order differs from frozen primitive catalogue")
    w = np.asarray(seqnmf_templates)
    thresholds = np.asarray(seqnmf_thresholds, dtype=float)
    if (w.ndim != 3 or w.shape[0] != len(type_ids) or w.shape[1] < 1 or w.shape[2] < 2
            or not np.isfinite(w).all() or np.any(w < 0)
            or np.any(np.sum(w, axis=(0, 2)) <= 0)):
        raise ValueError("seqNMF templates require finite nonnegative nonempty factors and matching vocabulary")
    if thresholds.shape != (w.shape[1],) or not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError("frozen activation thresholds must match template factors")
    blocks = bouts.segment.nunique()
    columns = len(bouts) + blocks * w.shape[2]
    # Conservative working-array bound before the one-hot/inference allocation.
    estimated = 16 * w.size + 64 * columns * (len(type_ids) + w.shape[1] + w.shape[2])
    if estimated > memory_budget_bytes:
        raise ValueError("frozen inference memory budget exceeded")
    code = {type_id: i for i, type_id in enumerate(type_ids)}
    exact_rows = []
    for name, tokens in exact_motifs.items():
        if not isinstance(name, str) or not name or any(token not in code for token in tokens):
            raise ValueError("exact motifs require named known primitive type identities")
        motif = tuple(code[token] for token in tokens)
        for start, stop in exact_occurrences(bouts, motif):
            exact_rows.append({'chain_id': name, 'start': start, 'end': stop, 'selection_score': float(len(motif))})
    exact = pd.DataFrame(exact_rows, columns=['chain_id', 'start', 'end', 'selection_score'])
    nmf_rows = []
    if len(bouts):
        x, valid, original = one_hot_segments(bouts, len(type_ids), w.shape[2])
        model = SeqNMF(w.shape[1], w.shape[2], penalty=seqnmf_penalty)
        model.w = w
        h = model.infer(x, valid_starts=valid, iterations=seqnmf_inference_iterations)
        for event in activation_events(h, w, x, thresholds=thresholds, valid_starts=valid):
            start, stop = int(original[event['start']]), int(original[event['end'] - 1]) + 1
            if start < 0 or stop <= start:
                raise ValueError("activation crosses a padding barrier")
            nmf_rows.append({**event, 'start': start, 'end': stop,
                             'chain_id': f"seqnmf_{event['factor']:02d}",
                             'selection_score': 1. - event['template_error']})
    nmf = pd.DataFrame(nmf_rows, columns=['chain_id', 'start', 'end', 'selection_score',
                                        'factor', 'activation', 'template_error'])
    results = {'primitives': lower}
    for name, table, catalog in (
        ('exact', exact, tuple(UnitType(key) for key in exact_motifs)),
        ('seqnmf', nmf, tuple(UnitType(f'seqnmf_{i:02d}') for i in range(w.shape[1]))),
    ):
        stream = from_bout_windows(table, bouts, session=primitives.session, level='episode',
                                   model_id=f'{primitives.model_id}/{name}', clock=primitives.clock,
                                   fps=primitives.fps, valid_intervals=primitives.valid_intervals,
                                   catalog=catalog, score_column='selection_score', score_kind='model_score')
        results[name] = BehaviorResult(stream, features=features, source_streams=(primitives,))
    if grammar is not None:
        results['grammar'] = replace(grammar.detect(primitives, memory_budget_bytes=memory_budget_bytes), features=features)
    for model in staged_models:
        key = f'staged/{model.model_id}'
        if key in results:
            raise ValueError("duplicate staged model identity")
        bounded = replace(model, memory_budget_bytes=min(model.memory_budget_bytes, memory_budget_bytes))
        results[key] = replace(bounded.decode_stream(primitives), features=features)
    return {'results': results, 'bouts': bouts, 'seqnmf_events': nmf,
            'resource_estimate_bytes': estimated}


def generate_synthetic_pose(
    n_frames: int = 120,
    fps: float = 10.0,
    seed: int = 42,
) -> PoseSeries:
    """Generate a synthetic 2D pose trajectory with movement bouts, turns, and gaps."""
    t_frames = np.arange(n_frames)
    time = TimeAxis(t_frames, unit="frame", clock="video")

    coords = np.zeros((n_frames, 2, 2))  # (T, K, 2): 0=nose, 1=tail_base
    conf = np.ones((n_frames, 2))

    pos_x, pos_y = 50.0, 50.0
    theta = 0.0

    for t in range(n_frames):
        # Two adjacent forward/turn cycles, then stationary and forward motion.
        if t < 20 or 40 <= t < 60 or t >= 100:
            speed = 3.0  # forward 3 px/frame
            d_theta = 0.0
        elif 20 <= t < 40 or 60 <= t < 80:
            speed = 1.0
            d_theta = 0.15  # turn
        else:
            speed = 0.0  # rest
            d_theta = 0.0

        theta += d_theta
        pos_x += speed * np.cos(theta)
        pos_y += speed * np.sin(theta)

        # Body axis length = 10 px
        coords[t, 0] = [pos_x + 5.0 * np.cos(theta), pos_y + 5.0 * np.sin(theta)]
        coords[t, 1] = [pos_x - 5.0 * np.cos(theta), pos_y - 5.0 * np.sin(theta)]

    # Introduce a gap at frames 70..74 (unobserved / low confidence)
    conf[70:75] = 0.0

    return PoseSeries(
        time=time,
        coordinates=coords,
        confidence=conf,
        bodyparts=("nose", "tail_base"),
        unit="px",
        coordinate_frame="arena",
    )


def run_synthetic_behavior_comparison(
    seed: int = 42,
    include_context: bool = True,
) -> dict[str, Any]:
    """Execute full end-to-end synthetic comparison workflow across layers and methods."""
    fps = 10.0
    pose = generate_synthetic_pose(n_frames=120, fps=fps, seed=seed)

    # 1. Continuous features
    association = extract_associated_trajectory_features(
        pose,
        session=SESSION,
        fps=fps,
        confidence_threshold=0.5,
    )
    features = association.series

    # 2. Primitive segmentation
    segmenter = CentroidPrimitiveSegmenter.fit(
        features,
        k=3,
        model_id="vq_segmenter",
        seed=seed,
        selected_features=("speed", "angular_velocity"),
    )
    primitives = segmenter.segment_stream(features, session=SESSION, fps=fps)

    # Coarse resolution remapping
    # Merge primitive 0 and 1 into cluster 0
    raw_labels = segmenter.predict_labels(features)
    coarse_labels = remap_states(raw_labels, [0, 0, 1])
    primitives_coarse = from_frame_labels(
        np.asarray(features.time.values, dtype=int),
        coarse_labels,
        session=SESSION,
        model_id="coarse_primitives",
        clock="video",
        fps=fps,
    )

    # 3. Alternative Episode Detectors
    # A. Staged HSMM detector
    staged_model = StagedEpisodeModel.create_linear_routine(
        model_id="staged_hsmm",
        routine_name="routine_01",
        phase_names=("p0", "p1"),
        primitive_catalog=segmenter.catalog,
        phase_emissions={
            "p0": {"0": 0.8, "1": 0.2},
            "p1": {"1": 0.8, "2": 0.2},
        },
        background_emissions={"0": 0.1, "1": 0.1, "2": 0.8},
        allow_immediate_reentry=True,
        mode="explicit_duration_hsmm",
        background_duration=np.array([.25, .25, .25, .25]),
    )
    staged_res = staged_model.decode_stream(primitives)

    # B. Finite Grammar Episode detector
    ep_rule = GrammarRule("grammar_ep", SequenceRule((TerminalMatch("0"), TerminalMatch("1"))))
    ep_spec = GrammarSpecification("ep_grammar_spec", (ep_rule,))
    grammar_detector = GrammarEpisodeDetector(ep_spec, model_id="grammar_detector")
    grammar_res = grammar_detector.detect(primitives)

    # C. Exact chains detector
    bouts, crosswalk = primitive_bout_table(primitives)
    exact_wins = exact_occurrences(bouts, [0, 1])
    exact_df = pd.DataFrame([{"chain_id": "exact_01", "start": a, "end": b} for a, b in exact_wins])
    exact_stream = from_bout_windows(
        exact_df,
        bouts,
        session=SESSION,
        level="episode",
        model_id="exact_chains",
        clock="video",
        fps=fps,
        valid_intervals=primitives.valid_intervals,
        catalog=(UnitType("exact_01"),),
    )

    # D. seqNMF activation detector
    # Synthetic factor templates
    n_bouts = len(bouts)
    h = np.zeros((1, n_bouts), dtype=np.float32)
    for start, _ in exact_wins:
        h[0, start] = 1.0
    x = np.eye(3, dtype=np.float32)[:, bouts.syllable_id.to_numpy(dtype=int)]
    # L = 2 template length
    w = np.zeros((3, 1, 2), dtype=np.float32)
    w[0, 0, 0] = 1.0
    w[1, 0, 1] = 1.0

    L = w.shape[2]
    valid_starts = np.zeros(n_bouts, dtype=bool)
    for b_idx in range(n_bouts - L + 1):
        if bouts.iloc[b_idx]["segment"] == bouts.iloc[b_idx + L - 1]["segment"]:
            valid_starts[b_idx] = True
    nmf_events = pd.DataFrame(activation_events(h, w, x, thresholds=[0.5], valid_starts=valid_starts, max_error=0.1))
    if not nmf_events.empty:
        nmf_events["chain_id"] = "nmf_0"
    nmf_stream = from_bout_windows(
        nmf_events,
        bouts,
        session=SESSION,
        level="episode",
        model_id="seqnmf",
        clock="video",
        fps=fps,
        valid_intervals=primitives.valid_intervals,
        catalog=(UnitType("nmf_0"),),
        score_column="activation" if not nmf_events.empty else None,
        score_kind="model_score" if not nmf_events.empty else None,
    )

    # 4. Assembly Composition
    # Compose grammar episodes into higher-order assemblies
    as_rule = GrammarRule("assembly_motif", SequenceRule((TerminalMatch("grammar_ep"), TerminalMatch("grammar_ep"))))
    as_spec = GrammarSpecification("as_grammar_spec", (as_rule,))
    as_composer = GrammarAssemblyComposer(as_spec, model_id="assembly_composer")
    assembly_res = as_composer.compose(grammar_res.stream)

    # 5. Optional Context Model
    context_res = None
    if include_context:
        ctx_model = EventContextHMM(
            model_id="context_hmm",
            n_contexts=2,
            context_names=("CtxA", "CtxB"),
            unit_catalog=segmenter.catalog,
            log_initial=np.log([0.5, 0.5]),
            log_transition=np.log([[0.8, 0.2], [0.2, 0.8]]),
            log_emissions=np.log([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]),
            unmatched_time_policy="break",
        )
        context_res = ctx_model.infer(primitives)

    # 6. Common Consumers: Organisation and Execution Analytics
    # Applied identically across all episode choices without detector-class branching!
    exec_tables = {}
    org_tables = {}
    onset_tables = {}

    for name, stream_or_res in [
        ("primitives", primitives),
        ("primitives_coarse", primitives_coarse),
        ("staged", staged_res),
        ("grammar", grammar_res),
        ("exact", exact_stream),
        ("seqnmf", nmf_stream),
        ("assembly", assembly_res),
    ]:
        s = stream_or_res.stream if hasattr(stream_or_res, "stream") else stream_or_res
        exec_df = measure_occurrence_execution(stream_or_res, features=association, feature_name="speed")
        agg_df = aggregate_execution_metrics(exec_df)
        org = measure_unit_organisation(s, gap_seconds=0.0)
        exec_tables[name] = agg_df
        org_tables[name] = org
        onset_tables[name] = measure_onset_profiles(stream_or_res, association)

    return {
        "features": features,
        "feature_association": association,
        "primitives": primitives,
        "primitives_coarse": primitives_coarse,
        "staged_result": staged_res,
        "grammar_result": grammar_res,
        "exact_stream": exact_stream,
        "seqnmf_stream": nmf_stream,
        "assembly_result": assembly_res,
        "context_result": context_res,
        "execution_metrics": exec_tables,
        "organisation_metrics": org_tables,
        "onset_profiles": onset_tables,
        "phase_profiles": measure_phase_profiles(staged_res, association),
    }


__all__ = [
    "compare_frozen_episode_methods",
    "generate_synthetic_pose",
    "run_synthetic_behavior_comparison",
]
