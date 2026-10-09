#!/usr/bin/env python3
"""
syllable_chains/intra_chain.py
=============================================================================
Part 2: Intra-Chain Execution & Intra-Syllable Analysis
Quantifies:
1. Consistency: Biomechanical/kinematic execution similarity via DTW and
   velocity profile correlation across occurrences.
2. Completion: Completion rates, stage-by-stage hazard of early exit, and
   destination exit syllable distributions.
3. Variations: Frequency and pattern of insertions, omissions, and repetitions.
4. Pacing: Stage-by-stage duration fractions and temporal elasticity.
5. Context-Dependent Movement: Kinematics of identical syllables inside vs.
   outside canonical chains.
6. Intra-Syllable Micro-Movement: Frame-by-frame velocity pacing within syllables.
=============================================================================
"""

import math
import itertools
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from .core import ChainDefinition, ChainOccurrence, ChainBout, ChainFinder


def fast_dtw_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes exact Dynamic Time Warping (DTW) distance between two 1D signals
    using dynamic programming. Time complexity O(N*M), negligible for resampled N=50.
    """
    N = len(x)
    M = len(y)
    if N == 0 or M == 0:
        return np.nan

    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        xi = x[i - 1]
        for j in range(1, M + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[N, M] / (N + M))


class IntraChainAnalyzer:
    """
    Analyzes intra-chain execution fidelity, biomechanical consistency,
    completion hazard, structural variations, and context-dependent syllable movement.
    """

    def __init__(
        self,
        chain_defs: List[ChainDefinition],
        occurrences: List[ChainOccurrence],
        session_dict: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.chain_defs = {cd.chain_id: cd for cd in chain_defs}
        self.chain_ids = [cd.chain_id for cd in chain_defs]
        self.occurrences = occurrences
        self.session_dict = session_dict

    # -------------------------------------------------------------------------
    # 1. CONSISTENCY ANALYSIS (DTW & PROFILE CORRELATION)
    # -------------------------------------------------------------------------
    def compute_consistency(self, sample_limit: int = 40) -> pd.DataFrame:
        """
        Quantifies execution consistency within each chain across genotypes.
        Averages across animals first to prevent hyperactive animal dominance.
        Measures:
        - Mean pairwise Dynamic Time Warping (DTW) distance of velocity profiles (lower = more stereotyped)
        - Mean pairwise Pearson correlation r of velocity profiles (higher = more stereotyped)
        - Coefficient of variation (CV) of execution durations
        """
        records = []
        groups = sorted(list(set(o.genotype for o in self.occurrences)))

        for cid in self.chain_ids:
            c_occs = [o for o in self.occurrences if o.chain_id == cid and o.is_complete]
            
            for g in groups:
                g_occs = [o for o in c_occs if o.genotype == g]
                if len(g_occs) < 3:
                    continue

                mouse_ids = sorted(list(set(o.animal_id for o in g_occs)))
                mouse_dtws = []
                mouse_rs = []
                durations = [o.duration_s for o in g_occs]
                cv_dur = float(np.std(durations) / np.mean(durations)) if np.mean(durations) > 0 else np.nan

                # Compute per-animal consistency
                for mid in mouse_ids:
                    m_occs = [o for o in g_occs if o.animal_id == mid]
                    profiles = [o.speed_profile for o in m_occs if o.speed_profile is not None and not np.isnan(o.speed_profile).any()]
                    if len(profiles) >= 2:
                        if len(profiles) > sample_limit:
                            rng = np.random.default_rng(42)
                            sub_idx = rng.choice(len(profiles), size=sample_limit, replace=False)
                            profiles = [profiles[idx] for idx in sub_idx]
                        nP = len(profiles)
                        dtw_dists = []
                        p_rs = []
                        for i in range(nP):
                            for j in range(i + 1, nP):
                                dtw_dists.append(fast_dtw_1d(profiles[i], profiles[j]))
                                r, _ = stats.pearsonr(profiles[i], profiles[j])
                                if not np.isnan(r):
                                    p_rs.append(r)
                        if dtw_dists:
                            mouse_dtws.append(np.mean(dtw_dists))
                        if p_rs:
                            mouse_rs.append(np.mean(p_rs))

                # Pooled fallback if per-mouse occurrences are sparse
                if not mouse_dtws:
                    profiles = [o.speed_profile for o in g_occs if o.speed_profile is not None and not np.isnan(o.speed_profile).any()]
                    if len(profiles) >= 3:
                        dtw_dists = [fast_dtw_1d(profiles[i], profiles[j]) for i in range(len(profiles)) for j in range(i + 1, len(profiles))]
                        mouse_dtws = [np.mean(dtw_dists)]

                mean_dtw = float(np.mean(mouse_dtws)) if mouse_dtws else np.nan
                mean_r = float(np.mean(mouse_rs)) if mouse_rs else np.nan

                records.append({
                    "chain_id": cid,
                    "genotype": g,
                    "n_complete_occurrences": len(g_occs),
                    "n_animals_evaluated": len(mouse_dtws),
                    "mean_pairwise_dtw_dist": mean_dtw,
                    "mean_velocity_profile_correlation": mean_r,
                    "duration_cv": cv_dur,
                    "mean_duration_s": float(np.mean(durations)),
                    "std_duration_s": float(np.std(durations))
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 2. COMPLETION & EARLY ABORT / HAZARD ANALYSIS
    # -------------------------------------------------------------------------
    def compute_completion_and_aborts(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes:
        1. Completion rate per chain across genotypes.
        2. Stage-by-stage abort hazard: probability of ending early at stage k
           given that stage k was reached.
        3. Exit syllable distribution: which syllable did the animal divert to?
        """
        completion_records = []
        hazard_records = []
        groups = sorted(list(set(o.genotype for o in self.occurrences)))

        for cid in self.chain_ids:
            c_def = self.chain_defs[cid]
            L = c_def.length
            c_occs = [o for o in self.occurrences if o.chain_id == cid]

            for g in groups:
                g_occs = [o for o in c_occs if o.genotype == g]
                total = len(g_occs)
                if total == 0:
                    continue

                n_complete = sum(1 for o in g_occs if o.is_complete)
                n_abort = total - n_complete
                comp_rate = (n_complete / total) * 100.0

                completion_records.append({
                    "chain_id": cid,
                    "genotype": g,
                    "total_attempts": total,
                    "completed_count": n_complete,
                    "aborted_count": n_abort,
                    "completion_rate_pct": comp_rate
                })

                # Stage-by-stage hazard
                # A chain of length L has stages 0, 1, ..., L-1
                # Abort stage = k means it aborted while or after executing target stage k
                for stage_idx in range(L - 1):
                    # Count how many reached at least stage_idx
                    reached = sum(1 for o in g_occs if (o.is_complete or (o.abort_stage is not None and o.abort_stage >= stage_idx)))
                    stopped_here = sum(1 for o in g_occs if not o.is_complete and o.abort_stage == stage_idx)
                    hazard = (stopped_here / reached) if reached > 0 else 0.0

                    hazard_records.append({
                        "chain_id": cid,
                        "genotype": g,
                        "stage_index": stage_idx,
                        "stage_token": c_def.tokens[stage_idx],
                        "reached_count": reached,
                        "abort_count": stopped_here,
                        "abort_hazard_rate": hazard
                    })

        return pd.DataFrame(completion_records), pd.DataFrame(hazard_records)

    # -------------------------------------------------------------------------
    # 3. STRUCTURAL VARIATIONS (INSERTIONS, DELETIONS, REPETITIONS)
    # -------------------------------------------------------------------------
    def compute_variations(self) -> pd.DataFrame:
        """
        Quantifies structural variations during chain execution:
        - Insertion rate: extraneous syllables injected into the sequence.
        - Deletion / omission rate: canonical syllables omitted.
        - Repetition / stutter rate: syllables immediately re-executed within the chain.
        - Canonical fidelity: fraction of executions with zero edit distance.
        """
        records = []
        groups = sorted(list(set(o.genotype for o in self.occurrences)))

        for cid in self.chain_ids:
            c_occs = [o for o in self.occurrences if o.chain_id == cid]

            for g in groups:
                g_occs = [o for o in c_occs if o.genotype == g]
                total = len(g_occs)
                if total == 0:
                    continue

                n_exact = sum(1 for o in g_occs if o.edit_distance == 0)
                mean_edits = float(np.mean([o.edit_distance for o in g_occs]))
                mean_insertions = float(np.mean([len(o.insertions) for o in g_occs]))
                mean_deletions = float(np.mean([len(o.deletions) for o in g_occs]))
                mean_repetitions = float(np.mean([len(o.repetitions) for o in g_occs]))

                records.append({
                    "chain_id": cid,
                    "genotype": g,
                    "total_attempts": total,
                    "exact_execution_pct": (n_exact / total) * 100.0,
                    "mean_edit_distance": mean_edits,
                    "mean_insertions": mean_insertions,
                    "mean_deletions": mean_deletions,
                    "mean_repetitions_stutter": mean_repetitions
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 4. TEMPORAL PACING & ELASTICITY
    # -------------------------------------------------------------------------
    def compute_pacing_profiles(self) -> pd.DataFrame:
        """
        Quantifies temporal pacing across stages of each chain:
        - Absolute duration (s) of each stage.
        - Relative duration fraction (fraction of total chain spent in each stage).
        - Allows detecting whether a genotype prolongs a specific action (e.g. rearing or wall-sniffing).
        """
        records = []
        groups = sorted(list(set(o.genotype for o in self.occurrences)))

        for cid in self.chain_ids:
            c_def = self.chain_defs[cid]
            L = c_def.length
            c_occs = [o for o in self.occurrences if o.chain_id == cid and o.is_complete]

            for g in groups:
                g_occs = [o for o in c_occs if o.genotype == g]
                if len(g_occs) < 2:
                    continue

                # Collect stage durations for each execution
                stage_durs = np.array([o.stage_durations_s[:L] for o in g_occs if len(o.stage_durations_s) >= L])
                if len(stage_durs) == 0:
                    continue

                mean_durs = np.mean(stage_durs, axis=0)
                std_durs = np.std(stage_durs, axis=0)
                total_dur = np.sum(mean_durs)

                for stage_idx in range(L):
                    stage_frac = (mean_durs[stage_idx] / total_dur) if total_dur > 0 else 0.0

                    records.append({
                        "chain_id": cid,
                        "genotype": g,
                        "stage_index": stage_idx,
                        "stage_token": c_def.tokens[stage_idx],
                        "stage_mean_duration_s": float(mean_durs[stage_idx]),
                        "stage_std_duration_s": float(std_durs[stage_idx]),
                        "stage_relative_fraction": float(stage_frac)
                    })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 5. CONTEXT-DEPENDENT SYLLABLE MOVEMENT
    # -------------------------------------------------------------------------
    def compute_context_dependent_movement(
        self,
        target_tokens: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compares execution kinematics (speed, duration) of the SAME syllable
        when performed inside a canonical chain vs. outside (standalone/other).
        Uses syllable-specific stage slicing and outside-chain baseline comparison.
        """
        records = []
        if target_tokens is None:
            # Collect all tokens present across canonical chains
            all_tokens = set()
            for cd in self.chain_defs.values():
                all_tokens.update(cd.tokens)
            target_tokens = sorted(list(all_tokens))

        groups = sorted(list(set(o.genotype for o in self.occurrences)))

        for tok in target_tokens:
            for g in groups:
                # Find occurrences where token was part of a completed chain
                in_chain_speeds = []
                in_chain_durs = []

                for o in self.occurrences:
                    if o.genotype == g and o.is_complete and tok in o.observed_tokens:
                        c_def = self.chain_defs.get(o.chain_id)
                        if c_def:
                            # Iterate over ALL matching stage positions (e.g. pos 0 and pos 2 in A -> B -> A)
                            indices = [idx for idx, t in enumerate(c_def.tokens) if t == tok]
                            for s_idx in indices:
                                if s_idx < len(o.stage_durations_s):
                                    in_chain_durs.append(o.stage_durations_s[s_idx])
                                if hasattr(o, "stage_mean_speeds") and s_idx < len(o.stage_mean_speeds):
                                    sp = o.stage_mean_speeds[s_idx]
                                    if not np.isnan(sp):
                                        in_chain_speeds.append(sp)

                if len(in_chain_durs) < 3:
                    continue

                # Compute outside-chain baseline if session_dict is available
                outside_speeds = []
                outside_durs = []
                if self.session_dict is not None:
                    for sid, sdata in self.session_dict.items():
                        if sdata.get("genotype") == g:
                            sylls = sdata["syllables"]
                            vel = sdata["velocity"]
                            fps = sdata["fps"]
                            in_chain_mask = np.zeros(len(sylls), dtype=bool)
                            for occ in [o for o in self.occurrences if o.session_id == sid and o.is_complete]:
                                in_chain_mask[occ.start_frame:occ.end_frame] = True

                            toks, starts, ends = ChainFinder.rle_bouts(sylls)
                            for t, s, e in zip(toks, starts, ends):
                                if t == tok and not in_chain_mask[s:e].any():
                                    outside_durs.append((e - s) / fps)
                                    if len(vel) >= e:
                                        outside_speeds.append(float(np.mean(vel[s:e])))

                mean_in_spd = float(np.mean(in_chain_speeds)) if in_chain_speeds else np.nan
                mean_out_spd = float(np.mean(outside_speeds)) if outside_speeds else np.nan
                mean_in_dur = float(np.mean(in_chain_durs)) if in_chain_durs else np.nan
                mean_out_dur = float(np.mean(outside_durs)) if outside_durs else np.nan

                spd_ratio = (mean_in_spd / mean_out_spd) if (mean_in_spd and mean_out_spd and mean_out_spd > 0) else np.nan
                dur_ratio = (mean_in_dur / mean_out_dur) if (mean_in_dur and mean_out_dur and mean_out_dur > 0) else np.nan

                records.append({
                    "syllable_token": tok,
                    "genotype": g,
                    "n_in_chain_samples": len(in_chain_durs),
                    "n_outside_chain_samples": len(outside_durs),
                    "in_chain_mean_duration_s": mean_in_dur,
                    "outside_chain_mean_duration_s": mean_out_dur,
                    "duration_modulation_ratio": dur_ratio,
                    "in_chain_mean_speed": mean_in_spd,
                    "outside_chain_mean_speed": mean_out_spd,
                    "speed_modulation_ratio": spd_ratio,
                    "in_chain_speed_std": float(np.std(in_chain_speeds)) if in_chain_speeds else np.nan
                })

        return pd.DataFrame(records)

    def compute_intra_syllable_micro_kinematics(
        self,
        n_bins: int = 30,
        min_bout_frames: int = 5
    ) -> pd.DataFrame:
        """
        6. Intra-Syllable Micro-Movement Analysis (Intra-Frame Kinematics):
        Extracts frame-by-frame velocity trajectories within individual syllables,
        interpolates them across normalized time tau in [0.0, 1.0], and compares
        micro-pacing across contexts (inside chain vs outside baseline) and genotypes.
        """
        records = []
        if self.session_dict is None:
            return pd.DataFrame(records)

        target_tokens = sorted(list(set(itertools.chain.from_iterable([c.tokens for c in self.chain_defs.values()]))))
        genotypes = sorted(list(set(o.genotype for o in self.occurrences if o.genotype)))
        
        session_in_chain_spans = {sid: [] for sid in self.session_dict.keys()}
        for occ in self.occurrences:
            if occ.is_complete and occ.session_id in session_in_chain_spans:
                session_in_chain_spans[occ.session_id].append((occ.start_frame, occ.end_frame))

        for tok in target_tokens:
            for g in genotypes:
                in_curves = []
                out_curves = []
                
                for sid, sdata in self.session_dict.items():
                    if sdata.get("genotype") != g:
                        continue
                    sylls = sdata["syllables"]
                    vel = sdata.get("velocity")
                    if vel is None or len(vel) < len(sylls):
                        continue
                    
                    in_spans = session_in_chain_spans.get(sid, [])
                    toks, starts, ends = ChainFinder.rle_bouts(sylls)
                    
                    for t, sf, ef in zip(toks, starts, ends):
                        if t == tok and (ef - sf) >= min_bout_frames:
                            snippet = vel[sf:ef]
                            if np.all(np.isnan(snippet)):
                                continue
                            orig_x = np.linspace(0, 1, len(snippet))
                            new_x = np.linspace(0, 1, n_bins)
                            resampled = np.interp(new_x, orig_x, snippet)
                            
                            is_in_chain = any(s_sf <= sf and ef <= s_ef for s_sf, s_ef in in_spans)
                            if is_in_chain:
                                in_curves.append(resampled)
                            else:
                                out_curves.append(resampled)

                for context_name, curves in [("in_chain", in_curves), ("outside_chain", out_curves)]:
                    if len(curves) > 0:
                        mean_curve = np.nanmean(curves, axis=0)
                        sem_curve = np.nanstd(curves, axis=0) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean_curve)
                        peak_idx = int(np.nanargmax(mean_curve))
                        peak_tau = float(peak_idx / (n_bins - 1))
                        onset_speed = float(mean_curve[0])
                        peak_speed = float(mean_curve[peak_idx])
                        offset_speed = float(mean_curve[-1])
                        early_accel = float((mean_curve[int(n_bins * 0.3)] - mean_curve[0]) / 0.3)
                        
                        for b_idx in range(n_bins):
                            tau = b_idx / (n_bins - 1)
                            records.append({
                                "syllable_token": tok,
                                "genotype": g,
                                "context": context_name,
                                "n_bouts": len(curves),
                                "tau": tau,
                                "speed_mean": float(mean_curve[b_idx]),
                                "speed_sem": float(sem_curve[b_idx]),
                                "onset_speed": onset_speed,
                                "peak_speed": peak_speed,
                                "offset_speed": offset_speed,
                                "peak_tau": peak_tau,
                                "early_accel": early_accel
                            })

        return pd.DataFrame(records)

