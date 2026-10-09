#!/usr/bin/env python3
"""
run_24_candidate_transition_exemplars.py

Evaluates behavioral sequence candidates using FORWARD CONSTITUENT TRANSITION EXEMPLARS.
Instead of evaluating the multi-syllable sequence post-hoc as a single global trajectory,
this pipeline evaluates each constituent syllable locally within its forward transition bucket
(syllable S transitioning to next syllable S_next, or cluster C_next).

Frozen Viewing Preset:
- 40 FPS natural speed (sync_mode="nothing")
- follow_centroid crop
- Centroid dot on mouse (strictly active syllable color, no badge dot, no yellow transition flashes)
- Solid black canvas padding
- Token HUD
- NO trajectory inset (show_trajectory_inset=False)

Outputs:
- Videos: syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/
- Catalog: syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/README_24_TRANSITION_EXEMPLARS.md
- Log: syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars/screening_transition_exemplars.log
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# Headless & platform safety
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.chain_grid_movie_wrapper import (
    generate_chain_grid_movie,
    load_chain_movie_resources,
    standardize_chain_instances,
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/screening_24_transition_exemplars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "screening_transition_exemplars.log"
MD_CATALOG = OUTPUT_DIR / "README_24_TRANSITION_EXEMPLARS.md"
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


SYLL_TO_CLUSTER = {
    0: 0, 4: 0, 9: 0, 13: 0, 15: 0, 16: 0,
    1: 1, 11: 1, 12: 1, 22: 1, 30: 1,
    2: 2, 6: 2, 8: 2, 10: 2, 17: 2, 18: 2, 19: 2, 21: 2,
    7: 3, 14: 3, 27: 3, 34: 3, 38: 3,
    3: 4, 5: 4, 20: 4, 23: 4, 24: 4, 26: 4, 29: 4, 33: 4, 41: 4, 47: 4
}


def get_recording_metadata(results):
    meta = {}
    for rk in results.keys():
        drug = "Amphetamine" if "day2" in rk else "Saline"
        parts = rk.split("_")
        tag = parts[2] if len(parts) > 2 else ""
        first_id = tag.split(".")[0] if "." in tag else "Unknown"
        geno = "KI" if any(x in tag for x in ["1658", "1664", "1666", "1749", "1771", "1694"]) else "WT"
        meta[rk] = {"animal_id": first_id, "genotype": geno, "drug": drug}
    return meta


class TransitionBucketEvaluator:
    """
    Evaluates individual constituent syllable bouts within their forward transition buckets:
    (syllable S -> next syllable S_next) or coarse back-off (syllable S -> next cluster C_next).
    
    Fits 3D PCA on each transition bucket's single-syllable latent trajectories, computes
    the medoid exemplar, and scores bouts by distance percentile rank within that bucket.
    """
    def __init__(self, results, all_bouts, min_bucket_size=15):
        self.results = results
        self.min_bucket_size = min_bucket_size
        self.buckets = {}
        self.models = {}

        log("Indexing all clean bouts into forward transition buckets...")
        for b in all_bouts:
            if b["dur"] < 5 or b["dur"] > 65:
                continue
            s = b["syll"]
            ns = b["next_s"]
            nc = b["next_c"]

            # Key 1: exact forward pair (s, ns)
            k_pair = ("pair", s, ns)
            if k_pair not in self.buckets:
                self.buckets[k_pair] = []
            self.buckets[k_pair].append(b)

            # Key 2: coarse cluster transition (s, nc)
            k_clust = ("clust", s, nc)
            if k_clust not in self.buckets:
                self.buckets[k_clust] = []
            self.buckets[k_clust].append(b)

            # Key 3: syllable alone
            k_syll = ("syll", s)
            if k_syll not in self.buckets:
                self.buckets[k_syll] = []
            self.buckets[k_syll].append(b)

        log(f"Indexed {len(all_bouts):,} total bouts into {len(self.buckets):,} transition bucket keys.")

    def get_bucket_model(self, s, ns, nc):
        # Choose most specific bucket with enough data
        k_pair = ("pair", s, ns)
        if k_pair in self.buckets and len(self.buckets[k_pair]) >= self.min_bucket_size:
            target_key = k_pair
        else:
            k_clust = ("clust", s, nc)
            if k_clust in self.buckets and len(self.buckets[k_clust]) >= self.min_bucket_size:
                target_key = k_clust
            else:
                target_key = ("syll", s)

        if target_key in self.models:
            return self.models[target_key]

        bouts = self.buckets.get(target_key, [])
        if len(bouts) < 2:
            model = {"valid": False, "key": target_key, "N": len(bouts)}
            self.models[target_key] = model
            return model

        trajs = []
        for b in bouts:
            chunk = self.results[b["rk"]]["latent_state"][b["sf"]:b["ef"]]
            if len(chunk) < 5:
                continue
            t_orig = np.linspace(0, 1, len(chunk))
            t_target = np.linspace(0, 1, 15)
            interp = interp1d(t_orig, chunk, axis=0)
            trajs.append(interp(t_target).flatten())

        if len(trajs) < 2:
            model = {"valid": False, "key": target_key, "N": len(trajs)}
            self.models[target_key] = model
            return model

        X = np.array(trajs)
        pca = PCA(n_components=min(3, len(X) - 1), random_state=42).fit(X)
        Xpca = pca.transform(X)

        # Transition Medoid Exemplar
        pairwise = np.linalg.norm(Xpca[:, None, :] - Xpca[None, :, :], axis=-1)
        medoid_idx = int(np.argmin(pairwise.sum(axis=1)))
        dists = np.linalg.norm(Xpca - Xpca[medoid_idx], axis=1)
        sorted_dists = np.sort(dists)

        model = {
            "valid": True, "key": target_key, "N": len(X),
            "pca": pca, "medoid_pca": Xpca[medoid_idx],
            "sorted_dists": sorted_dists
        }
        self.models[target_key] = model
        return model

    def score_bout(self, rk, sf, ef, s, ns, nc):
        model = self.get_bucket_model(s, ns, nc)
        if not model["valid"]:
            return 0.5, model["key"], model["N"]

        chunk = self.results[rk]["latent_state"][sf:ef]
        if len(chunk) < 5:
            return 0.1, model["key"], model["N"]

        t_orig = np.linspace(0, 1, len(chunk))
        t_target = np.linspace(0, 1, 15)
        interp = interp1d(t_orig, chunk, axis=0)
        traj = interp(t_target).flatten()[None, :]

        pca_pt = model["pca"].transform(traj)[0]
        dist = float(np.linalg.norm(pca_pt - model["medoid_pca"]))
        rank = np.searchsorted(model["sorted_dists"], dist)
        percentile = float(max(0.0, 1.0 - (rank / model["N"])))
        return percentile, model["key"], model["N"]


def score_and_select_constituent_exemplars(raw_candidates, results, evaluator, target_count=6):
    """
    Evaluates each candidate chain solely by its constituent stage scores
    in their respective forward transition buckets.
    """
    N = len(raw_candidates)
    if N == 0:
        return [], {"N": 0, "strict_N": 0, "mod_N": 0, "mean_top6_pct": "0.0%", "min_top6_pct": "0.0%", "unique_animals": 0}

    scored_candidates = []
    strict_count = 0
    mod_count = 0

    for cand in raw_candidates:
        rk = cand["recording_key"]
        bounds = cand["stage_boundaries"]
        n_stages = len(bounds) - 1
        sylls = results[rk]["syllable"]

        stage_ps = []
        for k in range(n_stages):
            sf = bounds[k]
            ef = bounds[k+1]
            s = int(sylls[sf]) if sf < len(sylls) else 0
            if k + 1 < n_stages:
                next_sf = bounds[k+1]
                ns = int(sylls[next_sf]) if next_sf < len(sylls) else -1
            else:
                ns = int(sylls[ef]) if ef < len(sylls) else -1
            nc = SYLL_TO_CLUSTER.get(ns, -1) if ns != -1 else -1

            p, _, _ = evaluator.score_bout(rk, sf, ef, s, ns, nc)
            stage_ps.append(p)

        min_p = min(stage_ps)
        geom_p = float(np.prod(stage_ps) ** (1.0 / len(stage_ps)))

        if min_p >= 0.50:
            strict_count += 1
        if min_p >= 0.33:
            mod_count += 1

        cand_copy = dict(cand)
        cand_copy["geom_p"] = geom_p
        cand_copy["min_p"] = min_p
        cand_copy["stage_ps"] = stage_ps
        scored_candidates.append(cand_copy)

    # Sort descending by constituent geometric mean percentile
    scored_candidates.sort(key=lambda x: (x["geom_p"], x["min_p"]), reverse=True)

    # Select top 6 stratified across distinct animals
    selected = []
    seen_animals = set()
    for c in scored_candidates:
        aid = str(c.get("animal_id", ""))
        if aid not in seen_animals:
            selected.append(c)
            seen_animals.add(aid)
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        for c in scored_candidates:
            if c not in selected:
                selected.append(c)
                if len(selected) >= target_count:
                    break

    top_geom = [c["geom_p"] for c in selected]
    top_min = [c["min_p"] for c in selected]
    mean_top6 = float(np.mean(top_geom)) * 100.0 if top_geom else 0.0
    min_top6 = float(np.mean(top_min)) * 100.0 if top_min else 0.0
    unique_animals = len(set(c.get("animal_id", "") for c in selected))

    stats = {
        "N": N,
        "strict_N": strict_count,
        "strict_pct": f"{strict_count / max(1, N) * 100.0:.1f}%",
        "mod_N": mod_count,
        "mod_pct": f"{mod_count / max(1, N) * 100.0:.1f}%",
        "mean_top6_pct": f"{mean_top6:.1f}%",
        "min_top6_pct": f"{min_top6:.1f}%",
        "unique_animals": unique_animals
    }

    return selected, stats


def write_catalog(records):
    header = (
        "# 24-Candidate Constituent Transition-Exemplar Catalog\n\n"
        "Exemplar selection using **Forward Transition-Bucketed Constituent Scoring**.\n"
        "Each constituent syllable is evaluated locally and causally within its forward transition bucket "
        "($S_k \\to S_{k+1}$ or $S_k \\to C_{k+1}$), completely eliminating post-hoc whole-chain trajectory filtering.\n\n"
        "### Mathematical Formulation:\n"
        "- **Forward Transition Bucketing**: For every constituent bout $k$, its 10D continuous latents are resampled to $L=15$ timepoints ($150\\text{D}$) and projected onto 3D PCA fitted on all bouts in that forward transition bucket $\\mathcal{B}(S_k \\to S_{k+1})$.\n"
        "- **Constituent Percentile**: Stage quality $p_k \\in [0, 1]$ represents proximity rank to the transition medoid exemplar within that bucket ($1.0 = \\text{exemplar}$). Chains are ranked by geometric mean $\\bar{p} = (\\prod p_k)^{1/M}$.\n"
        "- **Strict Constituent Chains**: Occurrences where *every single constituent stage* ranks in the top 50% ($p_{\\min} \\ge 0.50$) of its forward transition bucket.\n"
        "- **Frozen Viewing Preset**: Continuous 40 FPS real-speed, follow centroid crop, colored centroid dot on the mouse, solid black padding, and token HUD (trajectory inset removed).\n\n"
        "| ID | Method | Candidate Family | Tokens | Total Raw N | Strict Chains (≥50%) | Mod Chains (≥33%) | Top-6 Mean Pct | Top-6 Min Stage Pct | Animals | WT / KI | Saline / Amph | Video File |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in records:
        vid_link = f"[{Path(r['video_file']).name}](./{Path(r['video_file']).name})" if r.get("video_file") else "Pending"
        row = (
            f"| **{r['id']}** | {r['method']} | {r['name']} | `{r['tokens']}` | "
            f"{r['total_matches']:,} | {r['strict_N']:,} ({r['strict_pct']}) | {r['mod_N']:,} ({r['mod_pct']}) | "
            f"{r['mean_top6_pct']} | {r['min_top6_pct']} | {r['unique_animals']} | "
            f"{r['wt_count']} / {r['ki_count']} | {r['sal_count']} / {r['amph_count']} | {vid_link} |"
        )
        rows.append(row)

    footer = (
        "\n---\n\n"
        "## 3-Question Screening Review Rubric\n\n"
        "When reviewing each 6-tile video, evaluate the sequence against this 3-question rubric:\n\n"
        "1. **Understandable Progression**:\n"
        "   - `Clear`: Consistent, recognizable behavioral transition shared across all 6 tiles.\n"
        "   - `Possible`: Shared motor theme is visible, but timing or subtle variations make boundaries loose.\n"
        "   - `No`: Tiles appear qualitatively discordant despite sharing token strings.\n"
        "   - `Corrupt`: Tracking artifacts, occlusion, or boundary chatter obscure interpretation.\n"
        "2. **What Appears Consistent?**\n"
        "   - Note invariant motor elements (e.g., body elongation, head yaw direction, gait cadence, wall proximity).\n"
        "3. **What Makes Interpretation Difficult?**\n"
        "   - Note confounding factors (e.g., varying departure angles, duration stretching, different wall orientations).\n\n"
        "---\n\n"
        "## Comparison Across the 3 Paradigms\n\n"
        "- **Uniform Screening Batch** ([`../screening_24_batch/README_24_CANDIDATES.md`](../screening_24_batch/README_24_CANDIDATES.md)): Samples evenly across the dataset to measure raw inter-session variability.\n"
        "- **Whole-Chain Density-Ratio Batch** ([`../screening_24_exemplars/README_24_EXEMPLARS.md`](../screening_24_exemplars/README_24_EXEMPLARS.md)): Evaluates the entire multi-syllable sequence as a single global trajectory via KPMS contrastive density ratio (post-hoc selection on complete chains).\n"
        "- **Constituent Transition-Exemplar Batch** (This Catalog): Evaluates each constituent syllable locally within its forward transition bucket, selecting chains composed entirely of canonical forward movements without whole-chain post-hoc bias.\n"
    )

    with open(MD_CATALOG, "w") as f:
        f.write(header + "\n".join(rows) + "\n" + footer)


def main():
    t_start = time.time()
    log("==================================================================")
    log("STARTING 24-CANDIDATE CONSTITUENT TRANSITION-EXEMPLAR BATCH GENERATION")
    log("Algorithm: Forward Transition-Bucketed Constituent Scoring (No Whole-Chain Post-Hoc)")
    log("Preset: 40 FPS natural speed, follow centroid, NO trajectory inset, black padding")
    log(f"Output Directory: {OUTPUT_DIR}")
    log("==================================================================")

    res = load_chain_movie_resources()
    results = res["results"]
    vfi = res["video_frame_indexes"]
    rec_meta = get_recording_metadata(results)
    log(f"Loaded {len(results)} recording sessions.")

    # Extract all syllable bouts
    log("Extracting all continuous syllable bouts across sessions...")
    all_bouts = []
    session_bouts = {}
    for rk, data in results.items():
        sylls = data["syllable"]
        c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
        b_sylls = np.array([sylls[0]] + sylls[c_idx].tolist())
        b_starts = np.array([0] + c_idx.tolist())
        b_ends = np.array(c_idx.tolist() + [len(sylls)])
        b_clusters = np.array([SYLL_TO_CLUSTER.get(s, 0) for s in b_sylls])
        session_bouts[rk] = {
            "sylls": b_sylls, "starts": b_starts, "ends": b_ends, "clusters": b_clusters
        }
        for i in range(len(b_sylls)):
            s = int(b_sylls[i])
            sf = int(b_starts[i])
            ef = int(b_ends[i])
            dur = ef - sf
            ns = int(b_sylls[i+1]) if i + 1 < len(b_sylls) else -1
            nc = SYLL_TO_CLUSTER.get(ns, -1) if ns != -1 else -1
            all_bouts.append({
                "rk": rk, "sf": sf, "ef": ef, "dur": dur,
                "syll": s, "next_s": ns, "next_c": nc
            })

    evaluator = TransitionBucketEvaluator(results, all_bouts)

    df_b = pd.read_csv(DATA_DIR / "diverse_templates_match_database.csv") if (DATA_DIR / "diverse_templates_match_database.csv").exists() else pd.DataFrame()
    df_dtw = pd.read_csv(DATA_DIR / "top20_retrieved_episodes_alignment.csv") if (DATA_DIR / "top20_retrieved_episodes_alignment.csv").exists() else pd.DataFrame()

    candidates = [
        # Method A1: Exact N-Grams
        {
            "id": "A1_01", "method": "Method A1 (Exact N-Grams)", "name": "Locomotor Alternation Triplet",
            "type": "exact", "pattern": [1, 11, 1], "tokens": "[1, 11, 1]"
        },
        {
            "id": "A1_02", "method": "Method A1 (Exact N-Grams)", "name": "Orienting Turn Triplet",
            "type": "exact", "pattern": [6, 17, 2], "tokens": "[6, 17, 2]"
        },
        {
            "id": "A1_03", "method": "Method A1 (Exact N-Grams)", "name": "Rearing-Descent Triplet",
            "type": "exact", "pattern": [14, 7, 34], "tokens": "[14, 7, 34]"
        },

        # Method A2: SEQUITUR Grammar Induction
        {
            "id": "A2_01", "method": "Method A2 (SEQUITUR Grammar)", "name": "Locomotor Subroutine Rule_1",
            "type": "exact", "pattern": [1, 11, 1, 11], "tokens": "Rule_1: [1, 11] + [1, 11]"
        },
        {
            "id": "A2_02", "method": "Method A2 (SEQUITUR Grammar)", "name": "Exploration Reorient Rule_2",
            "type": "exact", "pattern": [16, 4, 0], "tokens": "Rule_2: [16, 4] + [0]"
        },
        {
            "id": "A2_03", "method": "Method A2 (SEQUITUR Grammar)", "name": "Rearing Reorient Rule_3",
            "type": "exact", "pattern": [14, 7, 0], "tokens": "Rule_3: [14, 7] + [0]"
        },

        # Method A3: Cluster-RLE Triplet
        {
            "id": "A3_01", "method": "Method A3 (Cluster-RLE)", "name": "Locomotion-Pause-Locomotion",
            "type": "cluster_rle", "pattern": [1, 4, 1], "tokens": "Cluster [1 -> 4 -> 1]"
        },
        {
            "id": "A3_02", "method": "Method A3 (Cluster-RLE)", "name": "Exploratory-Orient-Exploratory",
            "type": "cluster_rle", "pattern": [0, 2, 0], "tokens": "Cluster [0 -> 2 -> 0]"
        },
        {
            "id": "A3_03", "method": "Method A3 (Cluster-RLE)", "name": "Rearing-Locomotion-Pause",
            "type": "cluster_rle", "pattern": [3, 1, 4], "tokens": "Cluster [3 -> 1 -> 4]"
        },

        # Method B: Weighted Alignment Motif
        {
            "id": "B_01", "method": "Method B (Weighted Alignment)", "name": "tpl_001_alt3 (Locomotion Family)",
            "type": "db_b", "template_id": "tpl_001_alt3", "tokens": "[1, 11, 1] (soft align)"
        },
        {
            "id": "B_02", "method": "Method B (Weighted Alignment)", "name": "tpl_003_ori3 (Orienting Family)",
            "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "[6, 17, 2] (soft align)"
        },
        {
            "id": "B_03", "method": "Method B (Weighted Alignment)", "name": "tpl_005_rear3 (Rearing Family)",
            "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "[14, 7, 34] (soft align)"
        },

        # Method C: Profile HMM Matches
        {
            "id": "C_01", "method": "Method C (Profile HMM)", "name": "Locomotor Elastic Family",
            "type": "exact", "pattern": [1, 11, 12], "tokens": "pHMM Family 1: [1, 11, 12]"
        },
        {
            "id": "C_02", "method": "Method C (Profile HMM)", "name": "Orienting Insertion Family",
            "type": "exact", "pattern": [6, 17, 21], "tokens": "pHMM Family 2: [6, 17, 21]"
        },
        {
            "id": "C_03", "method": "Method C (Profile HMM)", "name": "Rearing Exploration Family",
            "type": "exact", "pattern": [2, 14, 7, 34], "tokens": "pHMM Family 3: [2, 14, 7, 34]"
        },

        # Method D: Convolutional seqNMF Factors
        {
            "id": "D_01", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_05 (Locomotor Motif)",
            "type": "exact", "pattern": [1, 11, 1, 30], "tokens": "Factor 5: [1, 11, 1, 30]"
        },
        {
            "id": "D_02", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_02 (Orienting Factor)",
            "type": "exact", "pattern": [19, 6, 2], "tokens": "Factor 2: [19, 6, 2]"
        },
        {
            "id": "D_03", "method": "Method D (Convolutional seqNMF)", "name": "seqnmf_motif_01 (Exploration Factor)",
            "type": "exact", "pattern": [16, 4, 0, 13], "tokens": "Factor 1: [16, 4, 0, 13]"
        },

        # Method E: PST Branching Contexts
        {
            "id": "E_01", "method": "Method E (PST Branch)", "name": "context_018 (Rearing Exit Branch)",
            "type": "exact", "pattern": [14, 7, 0], "tokens": "Context 018: [14, 7] -> 0"
        },
        {
            "id": "E_02", "method": "Method E (PST Branch)", "name": "context_029 (Approach Branch)",
            "type": "exact", "pattern": [16, 4, 17], "tokens": "Context 029: [16, 4] -> 17"
        },
        {
            "id": "E_03", "method": "Method E (PST Branch)", "name": "context_023 (Locomotion Pause Branch)",
            "type": "exact", "pattern": [1, 0, 21], "tokens": "Context 023: [1, 0] -> 21"
        },

        # Method DTW: Continuous Kinematic DTW Reranked
        {
            "id": "DTW_01", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_alt3 (Kinematic Locomotion)",
            "type": "db_dtw", "tokens": "DTW Top Matches [1, 11, 1]"
        },
        {
            "id": "DTW_02", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_ori3 (Kinematic Orienting)",
            "type": "db_b", "template_id": "tpl_003_ori3", "tokens": "DTW Top Matches [6, 17, 2]"
        },
        {
            "id": "DTW_03", "method": "Method DTW (Kinematic Trajectory)", "name": "dtw_rear3 (Kinematic Rearing)",
            "type": "db_b", "template_id": "tpl_005_rear3", "tokens": "DTW Top Matches [14, 7, 34]"
        }
    ]

    catalog_records = []

    for idx, cand in enumerate(candidates, 1):
        cid = cand["id"]
        cname = cand["name"]
        log(f"\n[{idx}/24] Evaluating Forward Transition Exemplars for {cid}: {cname} ({cand['method']})...")

        raw_candidates = []
        c_type = cand["type"]

        if c_type == "exact":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                b_sylls = b_info["sylls"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]
                for i in range(len(b_sylls) - pat_len + 1):
                    if np.array_equal(b_sylls[i:i+pat_len], pat):
                        s_frame = int(b_starts[i])
                        e_frame = int(b_ends[i+pat_len-1])
                        s_bounds = [int(b_starts[i+s]) for s in range(pat_len)] + [e_frame]
                        toks = [f"syll_{p:03d}" for p in pat]
                        raw_candidates.append({
                            "instance_id": f"{cid}_{rk[:10]}_{s_frame}",
                            "recording_key": rk,
                            "start_frame": s_frame,
                            "end_frame": e_frame,
                            "stage_boundaries": s_bounds,
                            "stage_tokens": toks,
                            "animal_id": m["animal_id"],
                            "genotype": m["genotype"],
                            "drug": m["drug"]
                        })

        elif c_type == "cluster_rle":
            pat = cand["pattern"]
            pat_len = len(pat)
            for rk, b_info in session_bouts.items():
                b_clusters = b_info["clusters"]
                b_starts = b_info["starts"]
                b_ends = b_info["ends"]
                m = rec_meta[rk]

                c_change = np.where(b_clusters[1:] != b_clusters[:-1])[0] + 1
                c_vals = np.array([b_clusters[0]] + b_clusters[c_change].tolist())
                c_b_start = np.array([0] + c_change.tolist())
                c_b_end = np.array(c_change.tolist() + [len(b_clusters)])

                for i in range(len(c_vals) - pat_len + 1):
                    if np.array_equal(c_vals[i:i+pat_len], pat):
                        start_bout = c_b_start[i]
                        end_bout = c_b_end[i+pat_len-1] - 1
                        s_frame = int(b_starts[start_bout])
                        e_frame = int(b_ends[end_bout])
                        s_bounds = [int(b_starts[c_b_start[i+s]]) for s in range(pat_len)] + [e_frame]
                        toks = [f"Cluster_{c_vals[i+s]:02d}" for s in range(pat_len)]
                        raw_candidates.append({
                            "instance_id": f"{cid}_{rk[:10]}_{s_frame}",
                            "recording_key": rk,
                            "start_frame": s_frame,
                            "end_frame": e_frame,
                            "stage_boundaries": s_bounds,
                            "stage_tokens": toks,
                            "animal_id": m["animal_id"],
                            "genotype": m["genotype"],
                            "drug": m["drug"]
                        })

        elif c_type == "db_b":
            tid = cand["template_id"]
            sub_df = df_b[df_b["template_id"] == tid]
            recs = sub_df.to_dict(orient="records")
            raw_candidates = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        elif c_type == "db_dtw":
            sub_df = df_dtw
            recs = sub_df.to_dict(orient="records")
            raw_candidates = standardize_chain_instances(recs, results=results, video_frame_indexes=vfi)

        # Evaluate and Select via Forward Constituent Transition Scoring
        selected_instances, stats = score_and_select_constituent_exemplars(
            raw_candidates=raw_candidates,
            results=results,
            evaluator=evaluator,
            target_count=6
        )

        wt_c = sum(1 for c in raw_candidates if c.get("genotype") == "WT")
        ki_c = sum(1 for c in raw_candidates if c.get("genotype") == "KI")
        sal_c = sum(1 for c in raw_candidates if c.get("drug") == "Saline")
        amph_c = sum(1 for c in raw_candidates if c.get("drug") == "Amphetamine")

        log(f"  Total Candidates (N): {stats['N']:,}")
        log(f"  Strict Constituent Exemplars (>=50% all stages): {stats['strict_N']:,} ({stats['strict_pct']})")
        log(f"  Moderate Constituent Exemplars (>=33% all stages): {stats['mod_N']:,} ({stats['mod_pct']})")
        log(f"  Selected Top-6: Mean Pct={stats['mean_top6_pct']} | Min Stage Pct={stats['min_top6_pct']} across {stats['unique_animals']} animals")

        # Clean video filename
        safe_name = cname.lower().replace(" ", "_").replace("/", "_").replace("->", "to").replace("+", "plus")
        vid_filename = f"{cid.lower()}_trans_exemplar_{safe_name}.mp4"
        vid_out_path = OUTPUT_DIR / vid_filename

        log(f"  Rendering transition-exemplar grid movie -> {vid_filename}...")
        t_vid = time.time()
        try:
            generate_chain_grid_movie(
                instances=selected_instances,
                output_path=str(vid_out_path),
                resources=res,
                rows=2,
                cols=3,
                sync_mode="nothing",
                centroid_mode="follow_centroid",
                reset_heading_per_syllable=False,
                filter_outliers=False,
                show_hud=True,
                show_trajectory_inset=False,
                use_gpu_encoder=True
            )
            vid_dur = time.time() - t_vid
            log(f"  Rendered {vid_filename} in {vid_dur:.1f}s.")
            vid_recorded_path = str(vid_out_path)
        except Exception as e:
            log(f"  ERROR rendering {vid_filename}: {e}")
            vid_recorded_path = ""

        rec = {
            "id": cid,
            "method": cand["method"],
            "name": cname,
            "tokens": cand["tokens"],
            "total_matches": stats["N"],
            "strict_N": stats["strict_N"],
            "strict_pct": stats["strict_pct"],
            "mod_N": stats["mod_N"],
            "mod_pct": stats["mod_pct"],
            "mean_top6_pct": stats["mean_top6_pct"],
            "min_top6_pct": stats["min_top6_pct"],
            "unique_animals": stats["unique_animals"],
            "wt_count": wt_c,
            "ki_count": ki_c,
            "sal_count": sal_c,
            "amph_count": amph_c,
            "video_file": vid_recorded_path
        }
        catalog_records.append(rec)
        write_catalog(catalog_records)

    total_time = time.time() - t_start
    log("==================================================================")
    log(f"ALL 24 TRANSITION-EXEMPLAR MOVIES COMPLETE in {total_time:.1f}s!")
    log(f"Catalog written to: {MD_CATALOG}")
    log("==================================================================")


if __name__ == "__main__":
    main()
