#!/usr/bin/env python3
"""
syllable_chains/inter_chain.py
=============================================================================
Part 1: Inter-Chain Organisation Analysis
Quantifies:
1. Repeats (Immediate re-execution probability & perseveration index)
2. Continuations (Sustained cycle run lengths & gap-based continuation hazard)
3. Transitions (Inter-chain Markov transition matrix & syntax entropy)
4. Initiations (Initiation event rate & lead vs repeat kinematic contrasts)
=============================================================================
"""

import itertools
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .core import ChainDefinition, ChainOccurrence, ChainBout, ChainFinder


class InterChainAnalyzer:
    """
    Analyzes higher-order organization between behavioral chains:
    how chains repeat, transition, continue, and initiate across animal groups.
    """

    def __init__(
        self,
        chain_defs: List[ChainDefinition],
        occurrences: List[ChainOccurrence],
        continuation_gap_threshold_s: float = 1.0
    ):
        self.chain_defs = {cd.chain_id: cd for cd in chain_defs}
        self.chain_ids = [cd.chain_id for cd in chain_defs]
        self.occurrences = sorted(occurrences, key=lambda o: (o.session_id, o.start_frame))
        self.continuation_gap = continuation_gap_threshold_s

    # -------------------------------------------------------------------------
    # 1. REPEATS & PERSEVERATION INDEX
    # -------------------------------------------------------------------------
    def compute_repeat_statistics(self) -> pd.DataFrame:
        """
        Computes immediate repeat probability P(C_i -> C_i) and perseveration index
        for each session and animal.
        Uses resolved non-overlapping episodes and correct outgoing-transition denominators.
        """
        records = []
        sessions = sorted(list(set(o.session_id for o in self.occurrences)))

        for sid in sessions:
            raw_occs = [o for o in self.occurrences if o.session_id == sid]
            s_occs = ChainFinder.resolve_non_overlapping_episodes(raw_occs, require_complete=True)
            if len(s_occs) < 2:
                continue

            aid = s_occs[0].animal_id
            g = s_occs[0].genotype
            cond = s_occs[0].condition

            n_trans = len(s_occs) - 1
            n_loops = sum(1 for i in range(n_trans) if s_occs[i].chain_id == s_occs[i + 1].chain_id)
            perseveration_index = n_loops / n_trans if n_trans > 0 else 0.0

            # Count outgoing transitions originating from each chain
            outgoing_counts = {cid: 0 for cid in self.chain_ids}
            chain_repeats = {cid: 0 for cid in self.chain_ids}

            for i in range(n_trans):
                cid = s_occs[i].chain_id
                next_cid = s_occs[i + 1].chain_id
                if cid in outgoing_counts:
                    outgoing_counts[cid] += 1
                if cid == next_cid and cid in chain_repeats:
                    chain_repeats[cid] += 1

            rec = {
                "session_id": sid,
                "animal_id": aid,
                "genotype": g,
                "condition": cond,
                "total_chain_occurrences": len(s_occs),
                "total_transitions": n_trans,
                "total_self_loops": n_loops,
                "perseveration_index": perseveration_index
            }

            for cid in self.chain_ids:
                tot_out = outgoing_counts[cid]
                reps = chain_repeats[cid]
                p_rep = (reps / tot_out) if tot_out > 0 else 0.0
                rec[f"{cid}_count"] = tot_out
                rec[f"{cid}_repeats"] = reps
                rec[f"{cid}_p_repeat"] = p_rep

            records.append(rec)

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 2. CONTINUATIONS & SUSTAINED BOUT DYNAMICS
    # -------------------------------------------------------------------------
    def extract_chain_bouts(self) -> List[ChainBout]:
        """
        Groups consecutive completed occurrences of the same chain within continuation_gap
        (enforcing 0.0 <= gap_s <= continuation_gap) into sustained multi-cycle chain bouts.
        Prevents overlapping occurrences or aborted suffixes from inflating cycle counts.
        """
        bouts = []
        sessions = sorted(list(set(o.session_id for o in self.occurrences)))

        for sid in sessions:
            raw_occs = [o for o in self.occurrences if o.session_id == sid]
            # Restrict to completed executions
            s_occs = [o for o in raw_occs if o.is_complete]
            if not s_occs:
                continue

            curr_bout_occs = [s_occs[0]]
            for i in range(1, len(s_occs)):
                prev = s_occs[i - 1]
                curr = s_occs[i]
                gap_s = (curr.start_frame - prev.end_frame) / curr.fps

                # Enforce strictly non-negative gap to eliminate phantom overlapping repetitions
                if curr.chain_id == prev.chain_id and (0.0 <= gap_s <= self.continuation_gap):
                    curr_bout_occs.append(curr)
                else:
                    # Finalize previous bout
                    start_f = curr_bout_occs[0].start_frame
                    end_f = curr_bout_occs[-1].end_frame
                    fps = curr_bout_occs[0].fps
                    bouts.append(ChainBout(
                        bout_id=f"{sid}_{curr_bout_occs[0].chain_id}_{start_f}",
                        chain_id=curr_bout_occs[0].chain_id,
                        session_id=sid,
                        animal_id=curr_bout_occs[0].animal_id,
                        genotype=curr_bout_occs[0].genotype,
                        condition=curr_bout_occs[0].condition,
                        start_frame=start_f,
                        end_frame=end_f,
                        duration_s=(end_f - start_f) / fps,
                        cycle_count=len(curr_bout_occs),
                        occurrences=list(curr_bout_occs)
                    ))
                    curr_bout_occs = [curr]

            # Final bout in session
            if curr_bout_occs:
                start_f = curr_bout_occs[0].start_frame
                end_f = curr_bout_occs[-1].end_frame
                fps = curr_bout_occs[0].fps
                bouts.append(ChainBout(
                    bout_id=f"{sid}_{curr_bout_occs[0].chain_id}_{start_f}",
                    chain_id=curr_bout_occs[0].chain_id,
                    session_id=sid,
                    animal_id=curr_bout_occs[0].animal_id,
                    genotype=curr_bout_occs[0].genotype,
                    condition=curr_bout_occs[0].condition,
                    start_frame=start_f,
                    end_frame=end_f,
                    duration_s=(end_f - start_f) / fps,
                    cycle_count=len(curr_bout_occs),
                    occurrences=list(curr_bout_occs)
                ))

        return bouts

    def compute_continuation_hazard(self, max_cycles: int = 5) -> pd.DataFrame:
        """
        Computes continuation hazard: probability of continuing to cycle k+1
        given that the mouse completed cycle k.
        """
        bouts = self.extract_chain_bouts()
        records = []

        groups = sorted(list(set(b.genotype for b in bouts)))
        for g in groups:
            g_bouts = [b for b in bouts if b.genotype == g]
            for cid in self.chain_ids:
                c_bouts = [b for b in g_bouts if b.chain_id == cid]
                total = len(c_bouts)
                if total == 0:
                    continue

                counts_reached = [sum(1 for b in c_bouts if b.cycle_count >= k) for k in range(1, max_cycles + 1)]
                hazard_rates = []
                for k in range(len(counts_reached) - 1):
                    at_k = counts_reached[k]
                    at_next = counts_reached[k + 1]
                    p_cont = (at_next / at_k) if at_k > 0 else 0.0
                    hazard_rates.append(p_cont)

                records.append({
                    "genotype": g,
                    "chain_id": cid,
                    "total_bouts": total,
                    "mean_cycles_per_bout": np.mean([b.cycle_count for b in c_bouts]),
                    "median_bout_duration_s": np.median([b.duration_s for b in c_bouts]),
                    "p_continue_1_to_2": hazard_rates[0] if len(hazard_rates) > 0 else 0.0,
                    "p_continue_2_to_3": hazard_rates[1] if len(hazard_rates) > 1 else 0.0,
                    "p_continue_3_to_4": hazard_rates[2] if len(hazard_rates) > 2 else 0.0,
                })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 3. INTER-CHAIN TRANSITIONS & SYNTAX ENTROPY
    # -------------------------------------------------------------------------
    def compute_transition_matrices(self) -> Dict[str, np.ndarray]:
        """
        Computes the empirical transition probability matrix between chains
        for each animal genotype.
        """
        matrices = {}
        groups = sorted(list(set(o.genotype for o in self.occurrences)))
        K = len(self.chain_ids)
        cid_to_idx = {cid: i for i, cid in enumerate(self.chain_ids)}

        for g in groups:
            T_mat = np.zeros((K, K), dtype=float)
            g_occs = [o for o in self.occurrences if o.genotype == g]
            sessions = sorted(list(set(o.session_id for o in g_occs)))

            for sid in sessions:
                raw_occs = [o for o in g_occs if o.session_id == sid]
                # Resolve to non-overlapping completed episodes
                s_occs = ChainFinder.resolve_non_overlapping_episodes(raw_occs, require_complete=True)
                for i in range(len(s_occs) - 1):
                    c1 = s_occs[i].chain_id
                    c2 = s_occs[i + 1].chain_id
                    if c1 in cid_to_idx and c2 in cid_to_idx:
                        T_mat[cid_to_idx[c1], cid_to_idx[c2]] += 1.0

            # Normalize rows to probabilities
            row_sums = T_mat.sum(axis=1, keepdims=True)
            T_prob = np.divide(T_mat, row_sums, out=np.zeros_like(T_mat), where=row_sums > 0)
            matrices[g] = T_prob

        return matrices

    def compute_syntax_entropy(self) -> pd.DataFrame:
        """
        Computes transition syntax entropy rate H(C_{t+1} | C_t) per session
        using resolved non-overlapping episodes.
        """
        records = []
        sessions = sorted(list(set(o.session_id for o in self.occurrences)))
        K = len(self.chain_ids)
        cid_to_idx = {cid: i for i, cid in enumerate(self.chain_ids)}

        for sid in sessions:
            raw_occs = [o for o in self.occurrences if o.session_id == sid]
            s_occs = ChainFinder.resolve_non_overlapping_episodes(raw_occs, require_complete=True)
            if len(s_occs) < 5:
                continue

            aid = s_occs[0].animal_id
            g = s_occs[0].genotype
            cond = s_occs[0].condition

            pairs = [(s_occs[i].chain_id, s_occs[i + 1].chain_id) for i in range(len(s_occs) - 1)]
            if not pairs:
                continue

            srcs = [p[0] for p in pairs]
            p_src = pd.Series(srcs).value_counts(normalize=True)
            p_pairs = pd.Series(pairs).value_counts(normalize=True)

            h_cond = 0.0
            for (i, j), pij in p_pairs.items():
                pi = p_src.get(i, 1e-12)
                h_cond -= pij * np.log2((pij / pi) + 1e-12)

            records.append({
                "session_id": sid,
                "animal_id": aid,
                "genotype": g,
                "condition": cond,
                "chain_transition_entropy": h_cond
            })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # 4. INITIATIONS & LEAD VS REPETITION KINEMATICS
    # -------------------------------------------------------------------------
    def compare_lead_vs_repeat_kinematics(self) -> pd.DataFrame:
        """
        Compares execution kinematics (speed, duration) between Lead (Cycle 1)
        and Follow-on Repetitions (Cycle >= 2) within sustained bouts.
        Computes animal-level paired inference to avoid pseudoreplication.
        """
        bouts = self.extract_chain_bouts()
        records = []

        # Filter to multi-cycle bouts
        multi_bouts = [b for b in bouts if b.cycle_count >= 2]

        for cid in self.chain_ids:
            c_bouts = [b for b in multi_bouts if b.chain_id == cid]
            if len(c_bouts) < 3:
                continue

            lead_speeds = [b.occurrences[0].mean_speed for b in c_bouts if not np.isnan(b.occurrences[0].mean_speed)]
            repeat_speeds = []
            for b in c_bouts:
                for o in b.occurrences[1:]:
                    if not np.isnan(o.mean_speed):
                        repeat_speeds.append(o.mean_speed)

            lead_durs = [b.occurrences[0].duration_s for b in c_bouts]
            repeat_durs = [o.duration_s for b in c_bouts for o in b.occurrences[1:]]

            # Animal-level paired statistics to protect against pseudoreplication
            mouse_ids = sorted(list(set(b.animal_id for b in c_bouts)))
            mouse_leads = []
            mouse_repeats = []
            for mid in mouse_ids:
                m_bouts = [b for b in c_bouts if b.animal_id == mid]
                m_lead = [b.occurrences[0].mean_speed for b in m_bouts if not np.isnan(b.occurrences[0].mean_speed)]
                m_rep = [o.mean_speed for b in m_bouts for o in b.occurrences[1:] if not np.isnan(o.mean_speed)]
                if m_lead and m_rep:
                    mouse_leads.append(float(np.mean(m_lead)))
                    mouse_repeats.append(float(np.mean(m_rep)))

            p_paired = np.nan
            if len(mouse_leads) >= 3:
                _, p_paired = stats.ttest_rel(mouse_leads, mouse_repeats)

            records.append({
                "chain_id": cid,
                "n_multi_bouts": len(c_bouts),
                "n_animals_tested": len(mouse_leads),
                "lead_mean_speed": np.mean(lead_speeds) if lead_speeds else np.nan,
                "repeat_mean_speed": np.mean(repeat_speeds) if repeat_speeds else np.nan,
                "speed_ratio_lead_to_repeat": (np.mean(lead_speeds) / np.mean(repeat_speeds)) if (lead_speeds and repeat_speeds and np.mean(repeat_speeds) > 0) else np.nan,
                "p_speed_diff_paired_mouse": p_paired,
                "lead_mean_duration_s": np.mean(lead_durs) if lead_durs else np.nan,
                "repeat_mean_duration_s": np.mean(repeat_durs) if repeat_durs else np.nan,
                "duration_ratio_lead_to_repeat": (np.mean(lead_durs) / np.mean(repeat_durs)) if (lead_durs and repeat_durs and np.mean(repeat_durs) > 0) else np.nan
            })

        return pd.DataFrame(records)
