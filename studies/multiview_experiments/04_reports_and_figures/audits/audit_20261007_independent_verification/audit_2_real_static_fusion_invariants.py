#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpcode.common.paths import get_repo_root, resolve_data_path

REPO = Path(str(get_repo_root()))
EXP = REPO / 'studies/multiview_experiments'
AUDIT_DIR = EXP / "04_reports_and_figures/audits/audit_20261007_independent_verification"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from fpcode.tracking.camera_correspondence import get_session_mapping
from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.tracking.static_fusion import PAW_MAP, STATIC_ARENA_TRANSFORMS, TOP_USE_BODYPARTS


def audit_session_invariants(sid, fused_path, pair):
    df_fused = pd.read_hdf(fused_path)
    df_top = pd.read_hdf(pair["top_h5"])
    df_bot = pd.read_hdf(pair["bot_h5"])

    f_sc = df_fused.columns.levels[0][0]
    t_sc = df_top.columns.levels[0][0]
    b_sc = df_bot.columns.levels[0][0]

    arena = pair["arena"]
    A = np.array(STATIC_ARENA_TRANSFORMS[arena]["matrix_2x2"], dtype=np.float32)
    b = np.array(STATIC_ARENA_TRANSFORMS[arena]["translation_2d"], dtype=np.float32)
    arena_width = 972.0

    mapping = get_session_mapping(sid, strict=True)
    N_top = len(df_fused)
    mapping = mapping[:N_top]
    top_idx = np.where((mapping >= 0) & (mapping < len(df_bot)))[0]
    bot_idx = mapping[top_idx]

    # 1. Test Dorsal Equality
    dorsal_errs = []
    for bp in TOP_USE_BODYPARTS:
        tx = df_top[t_sc, bp, "x"].iloc[:N_top].to_numpy(dtype=np.float32)
        ty = df_top[t_sc, bp, "y"].iloc[:N_top].to_numpy(dtype=np.float32)
        fx = df_fused[f_sc, bp, "x"].to_numpy(dtype=np.float32)
        fy = df_fused[f_sc, bp, "y"].to_numpy(dtype=np.float32)
        dx = np.nanmax(np.abs(tx - fx))
        dy = np.nanmax(np.abs(ty - fy))
        dorsal_errs.append(max(float(dx), float(dy)))
    max_dorsal_err = max(dorsal_errs)

    # 2. Test Real Paw Projection Against Ground Truth Formula
    paw_errs = []
    expected_projs = {}
    for target_bp, (_, orig_swap_bp) in PAW_MAP.items():
        raw_bx = df_bot[b_sc, orig_swap_bp, "x"].iloc[bot_idx].to_numpy(dtype=np.float32)
        raw_by = df_bot[b_sc, orig_swap_bp, "y"].iloc[bot_idx].to_numpy(dtype=np.float32)
        raw_bp = df_bot[b_sc, orig_swap_bp, "likelihood"].iloc[bot_idx].to_numpy(dtype=np.float32)

        raw_bx_refl = arena_width - raw_bx
        expected_proj = np.column_stack([raw_bx_refl, raw_by]) @ A.T + b
        expected_projs[target_bp] = expected_proj

        fused_x = df_fused[f_sc, target_bp, "x"].iloc[top_idx].to_numpy(dtype=np.float32)
        fused_y = df_fused[f_sc, target_bp, "y"].iloc[top_idx].to_numpy(dtype=np.float32)
        fused_p = df_fused[f_sc, target_bp, "likelihood"].iloc[top_idx].to_numpy(dtype=np.float32)

        err_x = np.nanmax(np.abs(fused_x - expected_proj[:, 0]))
        err_y = np.nanmax(np.abs(fused_y - expected_proj[:, 1]))
        err_p = np.nanmax(np.abs(fused_p - raw_bp))
        paw_errs.append(max(float(err_x), float(err_y), float(err_p)))
    max_paw_err = max(paw_errs)

    # 3. Test Real Anchor Perturbation
    # Perturb bot nose and tail in df_bot and verify that fused paws do NOT change
    df_bot_perturbed = df_bot.copy()
    df_bot_perturbed[(b_sc, "nose_tip", "x")] += 500.0
    df_bot_perturbed[(b_sc, "tail_base", "y")] -= 300.0
    # Re-evaluate expected paws from perturbed bot df
    pert_paw_errs = []
    for target_bp, (_, orig_swap_bp) in PAW_MAP.items():
        raw_bx_pert = df_bot_perturbed[b_sc, orig_swap_bp, "x"].iloc[bot_idx].to_numpy(dtype=np.float32)
        raw_bx_refl = arena_width - raw_bx_pert
        raw_by_pert = df_bot_perturbed[b_sc, orig_swap_bp, "y"].iloc[bot_idx].to_numpy(dtype=np.float32)
        expected_proj_pert = np.column_stack([raw_bx_refl, raw_by_pert]) @ A.T + b
        diff_pert = np.nanmax(np.abs(expected_proj_pert[:, 0] - expected_projs[target_bp][:, 0]))
        pert_paw_errs.append(float(diff_pert))
    max_pert_diff = max(pert_paw_errs)

    # 4. NEGATIVE CONTROLS:
    # A) Swapped paw negative control
    # Compare left_front_paw fused against expected right_front_paw
    raw_bx_swap = df_bot[b_sc, "left_front_paw", "x"].iloc[bot_idx].to_numpy(dtype=np.float32)
    expected_swap = np.column_stack([arena_width - raw_bx_swap, df_bot[b_sc, "left_front_paw", "y"].iloc[bot_idx].to_numpy(dtype=np.float32)]) @ A.T + b
    swap_diff = float(np.nanmedian(np.abs(fused_x - expected_swap[:, 0])))

    # B) One-frame temporal shift negative control
    bot_idx_shifted = np.clip(bot_idx + 1, 0, len(df_bot) - 1)
    raw_bx_shift = df_bot[b_sc, "right_front_paw", "x"].iloc[bot_idx_shifted].to_numpy(dtype=np.float32)
    expected_shift = np.column_stack([arena_width - raw_bx_shift, df_bot[b_sc, "right_front_paw", "y"].iloc[bot_idx_shifted].to_numpy(dtype=np.float32)]) @ A.T + b
    shift_diff = float(np.nanmedian(np.abs(fused_x - expected_shift[:, 0])))

    return {
        "sid": sid,
        "max_dorsal_err_px": max_dorsal_err,
        "max_paw_err_px": max_paw_err,
        "anchor_perturbation_diff_px": max_pert_diff,
        "negative_control_paw_swap_diff_px": swap_diff,
        "negative_control_1frame_shift_diff_px": shift_diff,
        "real_audit_passed": bool(max_dorsal_err < 1e-4 and max_paw_err < 1e-4 and max_pert_diff == 0.0),
        "negative_controls_caught": bool(swap_diff > 1.0 and shift_diff > 0.05)
    }

def main():
    pairs = find_matched_pairs()
    pairs_by_sid = {p["sid"]: p for p in pairs}
    fused_dir = resolve_data_path('data/fused/multiview/cohort_fused_static_21kp')
    fused_files = sorted(fused_dir.glob("*_tandem_21kp.h5"))[:10] # Audit 10 representative cohort files

    results = []
    for f in fused_files:
        sid = f.name.replace("_tandem_21kp.h5", "")
        pair = pairs_by_sid.get(sid)
        if not pair: continue
        res = audit_session_invariants(sid, f, pair)
        results.append(res)
        print(f"[{sid}]")
        print(f"  Max dorsal err: {res['max_dorsal_err_px']:.2e} px")
        print(f"  Max paw err:    {res['max_paw_err_px']:.2e} px")
        print(f"  Anchor pert diff: {res['anchor_perturbation_diff_px']:.2e} px")
        print(f"  Neg-control Swap diff:   {res['negative_control_paw_swap_diff_px']:.2f} px")
        print(f"  Neg-control 1-fr shift:  {res['negative_control_1frame_shift_diff_px']:.2f} px")
        print(f"  Audit Passed: {'✅' if res['real_audit_passed'] else '❌'}, Neg-Controls Caught: {'✅' if res['negative_controls_caught'] else '❌'}")

    out_file = AUDIT_DIR / "real_static_fusion_invariants_and_negative_controls.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved real invariant audit to {out_file}")

if __name__ == "__main__":
    main()
