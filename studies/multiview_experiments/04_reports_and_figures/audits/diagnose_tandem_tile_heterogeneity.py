#!/usr/bin/env python3
"""
diagnose_tandem_tile_heterogeneity.py
=====================================
Test 1: Do tandem (Seed 404) syllables look less alike than 1-cam (k240k) syllables
        when judged ONLY on dorsal (top-view) egocentric pose snippets?
Test 3: Is the large variance of bot_nose_tip / bot_tail_base signal or noise?

Both read the continuous-calibrated fused h5 files (60 fps raw top-frame index space)
and the results.h5 + video_frame_indexes.npz of each model.
"""
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.cluster import MiniBatchKMeans

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
FUSED = resolve_data_path('data/fused/multiview/fused_h5_continuous_calibrated')
T_DIR = resolve_data_path('data/models/kpms_tandem_cohort_continuous_calibrated')
T_RES = T_DIR / "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073/results.h5"
T_VFI = T_DIR / "video_frame_indexes.npz"
K_DIR = REPO / 'vendor/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k'
OUT = EXP / "04_reports_and_figures/audits"

DORSAL = ["nose", "left_ear_tip", "right_ear_tip", "neck", "mid_back", "mouse_center", "mid_backend",
          "mid_backend2", "mid_backend3", "tail_base", "left_shoulder", "left_midside", "left_hip",
          "right_shoulder", "right_midside", "right_hip", "head_midpoint"]
SNIP_LEN, SNIP_STEP, MAX_PER_SYL, CONF = 30, 5, 300, 0.5
rng = np.random.default_rng(0)


def label_arrays(res, vfi, key_to_base, n_by_base, upsample):
    """-> {base: int16 array (n_frames_raw) of syllable, -1 where unlabeled}"""
    out = {}
    for k in res.keys():
        b = key_to_base(k)
        if b not in n_by_base:
            continue
        lab = out.setdefault(b, np.full(n_by_base[b], -1, np.int16))
        idx = vfi[k][:]
        syl = res[k]["syllable"][:]
        n = min(len(idx), len(syl))
        idx, syl = idx[:n], syl[:n]
        ok = idx < len(lab)
        lab[idx[ok]] = syl[ok]
        if upsample:  # 1-cam is downsampled x2: copy to the skipped odd frame
            idx2 = idx[ok] + 1
            ok2 = idx2 < len(lab)
            lab[idx2[ok2]] = syl[ok][ok2]
    return out


def ego_dorsal(df):
    sc = df.columns.levels[0][0]
    P = np.stack([df[(sc, b, "x")].to_numpy() for b in DORSAL], 1), np.stack([df[(sc, b, "y")].to_numpy() for b in DORSAL], 1)
    L = np.stack([df[(sc, b, "likelihood")].to_numpy() for b in DORSAL], 1)
    XY = np.stack(P, -1).astype(np.float32)  # n,17,2
    ok = np.isfinite(XY).all((1, 2)) & (L >= CONF).all(1)
    c = XY.mean(1)
    v = XY[:, DORSAL.index("nose")] - XY[:, DORSAL.index("tail_base")]
    th = np.arctan2(v[:, 1], v[:, 0])
    cs, sn = np.cos(-th), np.sin(-th)
    R = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)  # n,2,2
    E = np.einsum("nij,nkj->nki", R, XY - c[:, None])
    E[~ok] = np.nan
    return E.reshape(len(E), -1)


def runs(lab):
    ch = np.flatnonzero(np.diff(lab) != 0) + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(lab)]
    return [(s, e, lab[s]) for s, e in zip(st, en) if lab[s] >= 0]


def r2_and_ceiling(S, y):
    """S: snippets (n,d); y: labels. returns (R2 by labels, R2 of kmeans with same K, K)."""
    tot = ((S - S.mean(0)) ** 2).sum()
    within = 0.0
    for s in np.unique(y):
        g = S[y == s]
        within += ((g - g.mean(0)) ** 2).sum()
    K = len(np.unique(y))
    km = MiniBatchKMeans(K, n_init=3, random_state=0, batch_size=4096).fit(S)
    return 1 - within / tot, 1 - km.inertia_ / tot, K


def main():
    files = sorted(FUSED.glob("*_tandem_23kp.h5"))
    base = lambda p: p.name.replace("_tandem_23kp.h5", "")
    n_by_base = {}
    ego, dfs_t3 = {}, {}
    t3 = dict(dn=[], dt=[], bc_n=[], bc_t=[], toplen=[], sess=[], hf_n=[], hf_t=[], hf_topn=[], hf_topt=[])
    ego_var = []
    for f in files:
        df = pd.read_hdf(f)
        b = base(f); n_by_base[b] = len(df); ego[b] = ego_dorsal(df)
        sc = df.columns.levels[0][0]
        g = lambda bp, c: df[(sc, bp, c)].to_numpy()
        # ---- Test 3 ----
        xy = lambda bp: np.stack([g(bp, "x"), g(bp, "y")], 1)
        tn, tt, bn, bt = xy("nose"), xy("tail_base"), xy("bot_nose_tip"), xy("bot_tail_base")
        pn, pt = g("bot_nose_tip", "likelihood"), g("bot_tail_base", "likelihood")
        ok = np.isfinite(tn).all(1) & np.isfinite(tt).all(1) & np.isfinite(bn).all(1) & np.isfinite(bt).all(1) & (g("nose", "likelihood") > .8) & (g("tail_base", "likelihood") > .8)
        sel = np.flatnonzero(ok)[::20]
        d_n, d_t = np.linalg.norm(bn - tn, axis=1), np.linalg.norm(bt - tt, axis=1)
        def hf(x):  # high-frequency residual std (px) vs 15-frame median
            xs = x[ok][:20000]
            return np.nanstd(xs - median_filter(xs, size=(15, 1), mode="nearest"), axis=0).mean()
        t3["dn"].append(d_n[sel]); t3["dt"].append(d_t[sel]); t3["bc_n"].append(pn[sel]); t3["bc_t"].append(pt[sel])
        t3["toplen"].append(np.linalg.norm(tn - tt, axis=1)[sel]); t3["sess"].append(np.full(len(sel), len(t3["sess"])))
        t3["hf_n"].append(hf(bn)); t3["hf_t"].append(hf(bt)); t3["hf_topn"].append(hf(tn)); t3["hf_topt"].append(hf(tt))
    print(f"loaded {len(files)} fused sessions", flush=True)

    # ---------------- Test 3 summary ----------------
    cat = {k: np.concatenate(v) for k, v in t3.items() if k in ("dn", "dt", "bc_n", "bc_t", "toplen", "sess")}
    s3 = {}
    for nm, d, p in (("nose", cat["dn"], cat["bc_n"]), ("tail", cat["dt"], cat["bc_t"])):
        s3[nm] = dict(
            median_px=float(np.nanmedian(d)), p90_px=float(np.nanpercentile(d, 90)), p99_px=float(np.nanpercentile(d, 99)),
            frac_gt_15px=float((d > 15).mean()), frac_gt_40px=float((d > 40).mean()),
            median_conf=float(np.nanmedian(p)),
            median_px_conf_ge_0p8=float(np.nanmedian(d[p >= .8])), median_px_conf_lt_0p5=float(np.nanmedian(d[p < .5])) if (p < .5).any() else None,
            corr_with_top_body_length=float(np.corrcoef(d[np.isfinite(d)], cat["toplen"][np.isfinite(d)])[0, 1]))
        # fraction of between-session variance (is it a per-session offset rather than behaviour?)
        sess = cat["sess"]; ms = pd.Series(d).groupby(sess).mean()
        s3[nm]["frac_var_between_sessions"] = float(ms.var() / (np.nanvar(d)))
    s3["hf_resid_std_px"] = dict(bot_nose=float(np.mean(t3["hf_n"])), top_nose=float(np.mean(t3["hf_topn"])),
                                 bot_tail=float(np.mean(t3["hf_t"])), top_tail=float(np.mean(t3["hf_topt"])))
    print("TEST3", json.dumps(s3, indent=1), flush=True)

    # ---------------- Test 1 ----------------
    t_res = h5py.File(T_RES); t_vfi = np.load(T_VFI)
    k_res = h5py.File(K_DIR / "results.h5"); k_vfi = np.load(K_DIR / "video_frame_indexes.npz")
    tb = lambda k: re.sub(r"_seg\d+$", "", k)
    kb = lambda k: re.search(r"CenterTopCamVid_(.+?)DLC", k).group(1)
    lab_t = label_arrays(t_res, t_vfi, tb, n_by_base, False)
    lab_k = label_arrays(k_res, k_vfi, kb, n_by_base, True)
    common = sorted(set(lab_t) & set(lab_k))
    print("common sessions", len(common), flush=True)

    summary = {}
    for name, labs in (("tandem_seed404", lab_t), ("onecam_k240k", lab_k)):
        snips, ys, durs_all = [], [], []
        for b in common:
            lab = labs[b].copy()
            both = (lab_t[b] >= 0) & (lab_k[b] >= 0)
            lab[~both] = -1  # identical frame support for both models
            E = ego[b]
            for s, e, y in runs(lab):
                durs_all.append(e - s)
                if e - s < SNIP_LEN:
                    continue
                w = E[s:s + SNIP_LEN:SNIP_STEP]
                if np.isnan(w).any():
                    continue
                snips.append(w.ravel()); ys.append(y)
        S, y = np.array(snips, np.float32), np.array(ys)
        keep = np.concatenate([rng.permutation(np.flatnonzero(y == s))[:MAX_PER_SYL] for s in np.unique(y)])
        S, y = S[keep], y[keep]
        # drop tiny syllables
        vals, cnt = np.unique(y, return_counts=True)
        good = vals[cnt >= 10]; m = np.isin(y, good); S, y = S[m], y[m]
        r2, r2km, K = r2_and_ceiling(S, y)
        summary[name] = dict(n_snippets=int(len(y)), n_syllables=int(K), R2_labels=float(r2), R2_kmeans_same_K=float(r2km),
                             R2_ratio=float(r2 / r2km), median_bout_raw60fps_frames=float(np.median(durs_all)))
        print(name, summary[name], flush=True)
    (OUT / "tile_heterogeneity_diagnosis.json").write_text(json.dumps(dict(test1=summary, test3=s3), indent=1))


if __name__ == "__main__":
    main()
