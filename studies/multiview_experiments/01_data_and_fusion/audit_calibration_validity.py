#!/usr/bin/env python3
"""
audit_calibration_validity.py
=============================================================================
Read-only audit: for every session in the TTL cache, compute the camera
correspondence with strict=False and report calibration provenance
(status / is_valid_crystal / slope / ppm drift / anchors / residuals / mapped
fraction), plus whether strict=True would accept the session.

Reads timestamp CSVs only. Writes a single new CSV report (never overwrites
unless --overwrite) and prints a summary. CPU-only; does not touch datasets,
checkpoints or running jobs.
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from fpcode.tracking.camera_correspondence import compute_session_camera_mapping
from fpcode.tracking.clock_sync import load_ttl_cache

DEFAULT_OUT = EXP_DIR / "04_reports_and_figures/audits/calibration_validity_audit.csv"


def audit_sessions(max_res_ms_limit: float = 2.0):
    cache = load_ttl_cache()
    rows = []
    for sid, info in sorted(cache.items()):
        top_csv = Path(info.get("top_csv", ""))
        bot_csv = Path(info.get("bot_csv", ""))
        row = {"sid": sid, "top_csv": str(top_csv), "bot_csv": str(bot_csv)}
        if not top_csv.is_file() or not bot_csv.is_file():
            row.update(status="MISSING_TIMESTAMP_FILES", strict_ok=False, error="csv not found")
            rows.append(row)
            continue
        try:
            mapping, dt_ms, calib = compute_session_camera_mapping(top_csv, bot_csv, strict=False)
            valid = mapping >= 0
            row.update(
                status=calib["status"],
                mode=calib["mode"],
                strict_ok=bool(calib["is_valid_crystal"]),
                slope=calib["slope"],
                ppm_drift=calib["ppm_drift"],
                n_anchors=calib["n_anchors_retained"],
                n_rejected=calib["n_anchors_rejected"],
                max_res_ms=calib["anchor_max_res_ms"],
                mean_res_ms=calib["anchor_mean_res_ms"],
                n_top=int(len(mapping)),
                mapped_frac=float(valid.mean()) if len(mapping) else 0.0,
                n_dropped=int((~valid).sum()),
                error="",
            )
            # Fit-quality gate (reported only here; strict mode in the library
            # rejects on anchor count).
            if row["strict_ok"] and calib["anchor_max_res_ms"] > max_res_ms_limit:
                row["strict_ok"] = False
                row["status"] = "POOR_FIT_QUALITY"
        except Exception as e:  # malformed timestamps etc.
            row.update(status="ERROR", strict_ok=False, error=f"{type(e).__name__}: {e}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Read-only calibration validity audit")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-res-ms", type=float, default=2.0,
                    help="Max anchor residual (ms) tolerated as a good fit")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and not args.overwrite:
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d_%H%M%S')}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = audit_sessions(args.max_res_ms)
    df.to_csv(out, index=False)

    print("=" * 80)
    print(f"CALIBRATION VALIDITY AUDIT  ({len(df)} sessions, {time.time() - t0:.1f}s)")
    print("=" * 80)
    print(df["status"].value_counts().to_string())
    print(f"\nStrict-calibration accepted: {int(df['strict_ok'].sum())} / {len(df)}")
    bad = df[~df["strict_ok"]]
    if len(bad):
        print("\nSessions rejected under strict mode:")
        cols = [c for c in ["sid", "status", "n_anchors", "max_res_ms", "error"] if c in bad]
        print(bad[cols].to_string(index=False))
    ok = df[df["strict_ok"]]
    if len(ok):
        print(f"\nAccepted sessions: ppm drift [{ok.ppm_drift.min():.1f}, {ok.ppm_drift.max():.1f}], "
              f"max anchor residual <= {ok.max_res_ms.max():.3f} ms, "
              f"mapped fraction min {ok.mapped_frac.min():.4f}")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
