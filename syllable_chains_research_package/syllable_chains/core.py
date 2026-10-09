#!/usr/bin/env python3
"""
syllable_chains/core.py
=============================================================================
Core Data Structures, Dataset Adapters, and Chain Identification Engines.
Decoupled and cohort-agnostic: operates on session pickles, KPMS results.h5,
or pandas DataFrames.
=============================================================================
"""

import os
import sys
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist


@dataclass
class ChainDefinition:
    """Definition of a canonical behavioral sequence (chain/routine)."""
    chain_id: str
    name: str
    tokens: List[int]
    category: str = "Uncategorized"
    description: str = ""
    target_cluster: Optional[List[int]] = None

    @property
    def length(self) -> int:
        return len(self.tokens)

    def to_string(self) -> str:
        return " -> ".join([f"syll_{s:02d}" for s in self.tokens])


@dataclass
class ChainOccurrence:
    """An individual empirical execution of a behavioral chain in a session."""
    occurrence_id: str
    chain_id: str
    session_id: str
    animal_id: str
    genotype: str
    condition: str
    
    # Timing
    start_frame: int
    end_frame: int
    duration_s: float
    fps: float
    
    # Token execution details
    observed_tokens: List[int]
    stage_frame_durations: List[int]
    stage_durations_s: List[float]
    is_complete: bool
    
    stage_frame_intervals: List[Tuple[int, int]] = field(default_factory=list)
    stage_mean_speeds: List[float] = field(default_factory=list)
    stage_peak_speeds: List[float] = field(default_factory=list)
    
    # Execution status
    abort_stage: Optional[int] = None  # None if complete; else index of stage reached before exit
    exit_syllable: Optional[int] = None # Syllable transitioned to after exit/abort
    
    # Structural variations vs canonical template
    edit_distance: int = 0
    insertions: List[int] = field(default_factory=list)
    deletions: List[int] = field(default_factory=list)
    repetitions: List[int] = field(default_factory=list)
    
    # Continuous Kinematics (if available)
    mean_speed: float = np.nan
    peak_speed: float = np.nan
    mean_ang_vel: float = np.nan
    trajectory_x: Optional[np.ndarray] = None
    trajectory_y: Optional[np.ndarray] = None
    speed_profile: Optional[np.ndarray] = None
    
    # Inter-chain context
    is_lead: bool = True               # True if first in a repeated burst; False if repetition
    bout_cycle_index: int = 1          # 1 for first execution, 2 for second in burst, etc.
    prev_chain_id: Optional[str] = None
    prev_chain_gap_s: Optional[float] = None


@dataclass
class ChainBout:
    """A sustained behavioral burst of one or more consecutive chain cycles."""
    bout_id: str
    chain_id: str
    session_id: str
    animal_id: str
    genotype: str
    condition: str
    start_frame: int
    end_frame: int
    duration_s: float
    cycle_count: int
    occurrences: List[ChainOccurrence]


class CohortDataAdapter:
    """
    Cohort-agnostic data loader.
    Translates various recording formats into unified session records:
    Dict[session_id, {
        'syllables': np.ndarray,
        'velocity': np.ndarray,
        'centroid': np.ndarray,
        'heading': Optional[np.ndarray],
        'fps': float,
        'animal_id': str,
        'genotype': str,
        'condition': str,
        'metadata': dict
    }]
    """

    @staticmethod
    def load_from_pickle(
        pkl_path: Union[str, Path],
        fps_default: float = 60.0,
        syllable_col: str = "syllable"
    ) -> Dict[str, Dict[str, Any]]:
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

        print(f"[CohortDataAdapter] Loading sessions from pickle: {pkl_path}...", flush=True)
        with open(pkl_path, "rb") as f:
            sessions = pickle.load(f)

        records = {}
        for s in sessions:
            meta = getattr(s, "metadata", {})
            sid = str(getattr(s, "session_id", meta.get("session_id", "unknown")))
            aid = str(meta.get("animal_id", "unknown"))
            g = str(meta.get("genotype", "Unknown"))
            cond = str(meta.get("condition", "Unknown"))

            if not hasattr(s, "all_data_df") or s.all_data_df is None:
                continue

            target_col = syllable_col if (syllable_col in s.all_data_df.columns and s.all_data_df[syllable_col].notna().sum() > 0) else "syllable"
            if target_col not in s.all_data_df.columns:
                continue

            df = s.all_data_df.copy()
            df = df.dropna(subset=[target_col])
            if len(df) < 100:
                continue

            sylls = df[target_col].to_numpy(dtype=int, na_value=-1)
            valid_mask = sylls >= 0
            df = df.iloc[valid_mask]
            sylls = sylls[valid_mask]

            if "velocity_px_s" in df.columns:
                vel = df["velocity_px_s"].values
            elif "velocity" in df.columns:
                vel = df["velocity"].values
            else:
                vel = np.zeros(len(sylls))
            
            # Centroid
            if "centroid_x" in df.columns and "centroid_y" in df.columns:
                centroid = np.column_stack([df["centroid_x"].values, df["centroid_y"].values])
            elif "x" in df.columns and "y" in df.columns:
                centroid = np.column_stack([df["x"].values, df["y"].values])
            else:
                centroid = np.zeros((len(sylls), 2))

            heading = df["heading"].values if "heading" in df.columns else None

            # Framerate
            fps = float(meta.get("fps", fps_default))
            if "SystemTimestamp" in df.columns:
                ts = df["SystemTimestamp"].values
                dt = np.diff(ts)
                dt_pos = dt[dt > 0]
                if len(dt_pos) > 10:
                    med_dt = np.median(dt_pos)
                    if 0.005 < med_dt < 0.1:
                        fps = float(np.round(1.0 / med_dt))

            records[sid] = {
                "syllables": sylls,
                "velocity": vel,
                "centroid": centroid,
                "heading": heading,
                "fps": fps,
                "animal_id": aid,
                "genotype": g,
                "condition": cond,
                "metadata": meta
            }

        print(f"[CohortDataAdapter] Successfully loaded {len(records)} valid sessions.", flush=True)
        return records

    @staticmethod
    def load_from_kpms_h5(
        h5_path: Union[str, Path],
        meta_csv: Optional[Union[str, Path]] = None,
        fps_default: float = 40.0
    ) -> Dict[str, Dict[str, Any]]:
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"KPMS results.h5 not found: {h5_path}")

        print(f"[CohortDataAdapter] Loading KPMS results.h5: {h5_path}...", flush=True)
        meta_df = None
        if meta_csv and Path(meta_csv).exists():
            meta_df = pd.read_csv(meta_csv)
            print(f"[CohortDataAdapter] Loaded metadata CSV with {len(meta_df)} rows from {meta_csv}")

        records = {}
        with h5py.File(h5_path, "r") as f:
            for k in f.keys():
                sylls = f[k]["syllable"][:]
                centroid = f[k]["centroid"][:] if "centroid" in f[k] else np.zeros((len(sylls), 2))
                heading = f[k]["heading"][:] if "heading" in f[k] else None

                # Compute speed from centroid diff
                fps = fps_default
                speed = np.zeros(len(sylls))
                if len(centroid) > 1:
                    d_pos = np.linalg.norm(np.diff(centroid, axis=0), axis=1) * fps
                    speed = np.concatenate(([0.0], d_pos))

                aid, g, cond = "unknown", "Unknown", "Unknown"
                meta_dict = {}

                if meta_df is not None:
                    # Match by key or session_id
                    match = meta_df[meta_df["results_dict_key"] == k]
                    if len(match) == 0 and "session_id" in meta_df.columns:
                        match = meta_df[meta_df["session_id"] == k]
                    if len(match) > 0:
                        row = match.iloc[0]
                        aid = str(row.get("animal_id", "unknown"))
                        g = str(row.get("genotype", "Unknown"))
                        cond = str(row.get("drug", row.get("condition", "Unknown")))
                        meta_dict = row.to_dict()

                records[k] = {
                    "syllables": sylls,
                    "velocity": speed,
                    "centroid": centroid,
                    "heading": heading,
                    "fps": fps,
                    "animal_id": aid,
                    "genotype": g,
                    "condition": cond,
                    "metadata": meta_dict
                }

        print(f"[CohortDataAdapter] Successfully loaded {len(records)} segments from H5.", flush=True)
        return records


class ChainFinder:
    """
    Finds and extracts chain occurrences from syllable streams.
    Handles both exact matches and structural variations/completions.
    """

    def __init__(self, chain_defs: Optional[List[ChainDefinition]] = None):
        self.chain_defs = chain_defs or []

    def extract_chain_occurrences(
        self,
        session_id: str,
        syllables: np.ndarray,
        velocity: np.ndarray,
        centroid: np.ndarray,
        animal_id: str,
        genotype: str,
        condition: str,
        fps: float = 40.0
    ) -> List[ChainOccurrence]:
        sdata = {
            "syllables": syllables,
            "velocity": velocity,
            "centroid": centroid,
            "animal_id": animal_id,
            "genotype": genotype,
            "condition": condition,
            "fps": fps
        }
        all_occs = []
        for c_def in self.chain_defs:
            occs = self.find_chain_occurrences(c_def, sdata, session_id)
            all_occs.extend(occs)
        return all_occs

    @staticmethod
    def rle_bouts(sylls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run-length encode syllable array into tokens, starts, and ends."""
        if len(sylls) == 0:
            return np.array([]), np.array([]), np.array([])
        changes = np.concatenate(([True], sylls[1:] != sylls[:-1]))
        starts = np.where(changes)[0]
        ends = np.concatenate((starts[1:], [len(sylls)]))
        tokens = sylls[starts]
        return tokens, starts, ends

    @classmethod
    def find_chain_occurrences(
        cls,
        chain_def: ChainDefinition,
        session_data: Dict[str, Any],
        session_id: str,
        max_inter_bout_gap_bouts: int = 1,
        max_inter_bout_gap_seconds: float = 0.5
    ) -> List[ChainOccurrence]:
        """
        Extracts all occurrences (both complete and aborted) of a chain in a session.
        Allows for flexible execution: checks if chain was completed, truncated early,
        or had internal insertions.
        """
        sylls = session_data["syllables"]
        vel = session_data["velocity"]
        centroid = session_data["centroid"]
        fps = session_data["fps"]
        aid = session_data["animal_id"]
        g = session_data["genotype"]
        cond = session_data["condition"]

        tokens, starts, ends = cls.rle_bouts(sylls)
        target = chain_def.tokens
        L = len(target)
        occurrences = []

        # Find occurrences starting at target[0]
        match_starts = np.where(tokens == target[0])[0]

        for s_idx in match_starts:
            # Greedily search forward up to 2*L bouts for the canonical progression
            curr_target_stage = 1
            observed = [target[0]]
            stage_starts = [starts[s_idx]]
            stage_ends = [ends[s_idx]]
            
            insertions = []
            repetitions = []
            
            b_idx = s_idx + 1
            aborted = False
            abort_stage = None
            exit_syll = None

            while curr_target_stage < L and b_idx < len(tokens):
                tok = tokens[b_idx]
                gap_bouts = (starts[b_idx] - stage_ends[-1])  # in frames, continuous
                
                if tok == target[curr_target_stage]:
                    # Target advanced!
                    observed.append(tok)
                    stage_starts.append(starts[b_idx])
                    stage_ends.append(ends[b_idx])
                    curr_target_stage += 1
                elif tok == observed[-1]:
                    # Syllable repeated / stutter
                    repetitions.append(tok)
                    observed.append(tok)
                    stage_ends[-1] = ends[b_idx]  # extend current stage
                else:
                    # Potential insertion or abort
                    # If within 1 insertion allowance and matches NEXT stage
                    if (curr_target_stage + 1 < L) and (b_idx + 1 < len(tokens)) and (tokens[b_idx + 1] == target[curr_target_stage]):
                        insertions.append(tok)
                        observed.append(tok)
                    else:
                        # Aborted execution
                        aborted = True
                        abort_stage = curr_target_stage - 1
                        exit_syll = tok
                        break
                b_idx += 1

            is_complete = (curr_target_stage == L)
            if not is_complete and not aborted:
                aborted = True
                abort_stage = curr_target_stage - 1
                exit_syll = tokens[b_idx] if b_idx < len(tokens) else None

            # Only record if at least step 1 was performed
            start_f = stage_starts[0]
            end_f = stage_ends[-1]
            dur_s = (end_f - start_f) / fps

            # Compute Levenshtein edit distance against canonical
            edit_dist = len(insertions) + len(repetitions) + (L - curr_target_stage)

            # Continuous kinematics slice
            occ_speed = vel[start_f:end_f] if len(vel) >= end_f else np.array([])
            mean_v = float(np.mean(occ_speed)) if len(occ_speed) > 0 else np.nan
            peak_v = float(np.percentile(occ_speed, 95)) if len(occ_speed) > 0 else np.nan

            traj_x = centroid[start_f:end_f, 0] if len(centroid) >= end_f else None
            traj_y = centroid[start_f:end_f, 1] if len(centroid) >= end_f else None

            # Resampled speed profile (50 points)
            if len(occ_speed) >= 5:
                x_old = np.linspace(0, 1, len(occ_speed))
                x_new = np.linspace(0, 1, 50)
                speed_prof = np.interp(x_new, x_old, occ_speed)
            else:
                speed_prof = None

            # Stage durations and stage-specific kinematics
            stage_f_durs = [int(e - s) for s, e in zip(stage_starts, stage_ends)]
            stage_s_durs = [float(d / fps) for d in stage_f_durs]
            stage_intervals = list(zip(stage_starts, stage_ends))
            stage_mean_v = [float(np.mean(vel[s:e])) if (e > s and len(vel) >= e) else np.nan for s, e in stage_intervals]
            stage_peak_v = [float(np.percentile(vel[s:e], 95)) if (e > s and len(vel) >= e) else np.nan for s, e in stage_intervals]

            occ = ChainOccurrence(
                occurrence_id=f"{session_id}_{chain_def.chain_id}_{start_f}",
                chain_id=chain_def.chain_id,
                session_id=session_id,
                animal_id=aid,
                genotype=g,
                condition=cond,
                start_frame=int(start_f),
                end_frame=int(end_f),
                duration_s=float(dur_s),
                fps=float(fps),
                observed_tokens=observed,
                stage_frame_durations=stage_f_durs,
                stage_durations_s=stage_s_durs,
                stage_frame_intervals=stage_intervals,
                stage_mean_speeds=stage_mean_v,
                stage_peak_speeds=stage_peak_v,
                is_complete=is_complete,
                abort_stage=abort_stage if not is_complete else None,
                exit_syllable=int(exit_syll) if exit_syll is not None else None,
                edit_distance=int(edit_dist),
                insertions=insertions,
                deletions=list(range(curr_target_stage, L)) if not is_complete else [],
                repetitions=repetitions,
                mean_speed=mean_v,
                peak_speed=peak_v,
                trajectory_x=traj_x,
                trajectory_y=traj_y,
                speed_profile=speed_prof
            )
            occurrences.append(occ)

        # Post-process: mark bursts / continuations vs lead executions
        # A valid continuation repetition MUST be a complete execution with non-negative gap (0 <= gap_s <= 1.0s)
        occurrences.sort(key=lambda o: o.start_frame)
        for i in range(len(occurrences)):
            if i == 0:
                occurrences[i].is_lead = True
                occurrences[i].bout_cycle_index = 1
                occurrences[i].prev_chain_id = None
                occurrences[i].prev_chain_gap_s = None
            else:
                prev = occurrences[i - 1]
                gap_s = (occurrences[i].start_frame - prev.end_frame) / fps
                occurrences[i].prev_chain_id = prev.chain_id
                occurrences[i].prev_chain_gap_s = float(gap_s)
                
                # Enforce strictly non-negative gap and complete executions to prevent phantom suffix bouts
                if prev.chain_id == occurrences[i].chain_id and (0.0 <= gap_s <= 1.0) and prev.is_complete and occurrences[i].is_complete:
                    occurrences[i].is_lead = False
                    occurrences[i].bout_cycle_index = prev.bout_cycle_index + 1
                else:
                    occurrences[i].is_lead = True
                    occurrences[i].bout_cycle_index = 1

        return occurrences

    @staticmethod
    def resolve_non_overlapping_episodes(
        occurrences: List[ChainOccurrence],
        require_complete: bool = True
    ) -> List[ChainOccurrence]:
        """
        Resolves simultaneous and overlapping candidate occurrences into
        distinct, non-overlapping behavioral routine episodes.
        Eliminates sorting artifacts where simultaneous starts generate fake transitions.
        """
        if not occurrences:
            return []

        cands = [o for o in occurrences if o.is_complete] if require_complete else list(occurrences)
        if not cands:
            return []

        # Sort by start_frame, then longer duration, then lower edit distance
        cands.sort(key=lambda o: (o.start_frame, -o.duration_s, o.edit_distance))

        resolved = [cands[0]]
        for i in range(1, len(cands)):
            curr = cands[i]
            prev = resolved[-1]
            # Require next episode to start after previous episode finishes
            if curr.start_frame >= prev.end_frame:
                resolved.append(curr)

        return resolved


def auto_discover_chains(
    session_dict: Dict[str, Dict[str, Any]],
    top_n: int = 6,
    chain_length: int = 3
) -> List[ChainDefinition]:
    """
    Automatically discovers high-frequency recurring syllable chains (n-grams)
    from session data streams. Ensures discovered chains are non-trivial and
    representative of the cohort's natural behavioral repertoire.
    """
    from collections import Counter

    ngram_counts = Counter()
    for sdata in session_dict.values():
        sylls = sdata["syllables"]
        if len(sylls) < chain_length:
            continue
        tokens, _, _ = ChainFinder.rle_bouts(sylls)
        for i in range(len(tokens) - chain_length + 1):
            gram = tuple(int(x) for x in tokens[i:i + chain_length])
            ngram_counts[gram] += 1

    discovered = []
    for idx, (gram, count) in enumerate(ngram_counts.most_common(top_n), 1):
        chain_id = f"AutoChain_{idx}_{'_'.join(str(s) for s in gram)}"
        name = f"Discovered Motif {idx}"
        desc = f"Empirical {chain_length}-stage sequence occurring {count:,} times"
        discovered.append(ChainDefinition(
            chain_id=chain_id,
            name=name,
            tokens=list(gram),
            category="AutoDiscovered",
            description=desc
        ))
    return discovered
