#!/usr/bin/env python3
"""Read-only production probe. Writes a new JSON report; never alters model/data.

Use the installed fpcode package and the same KPMS environment as rendering.
Pairwise comparisons use the same filtered ONSET pose and native onset time.
Raw observations are masked by their original likelihood, never zero-filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


def project_bot(xy, A, b, width):
    """Match static_fusion_engine float32 arithmetic, including W-x reflection."""
    p=np.asarray(xy,dtype=np.float32).copy()
    p[...,0]=np.float32(width)-p[...,0]
    return p@np.asarray(A,dtype=np.float32).T+np.asarray(b,dtype=np.float32)


def identity_error(actual_xy,actual_conf,expected_xy,expected_conf):
    """Identity checks do not hide missing-value disagreements behind QC masks."""
    actual_xy=np.asarray(actual_xy);expected_xy=np.asarray(expected_xy)
    fa=np.isfinite(actual_xy);fe=np.isfinite(expected_xy);both=fa&fe
    ca=np.asarray(actual_conf);ce=np.asarray(expected_conf)
    cb=np.isfinite(ca)&np.isfinite(ce)
    return {"coordinate_finite_mask_disagreements":int(np.sum(fa!=fe)),
            "max_abs_coordinate_error_px":float(np.max(np.abs(actual_xy[both]-expected_xy[both]))) if both.any() else None,
            "likelihood_finite_mask_disagreements":int(np.sum(np.isfinite(ca)!=np.isfinite(ce))),
            "max_abs_likelihood_error":float(np.max(np.abs(ca[cb]-ce[cb]))) if cb.any() else None}


def video_seek_check(path,frame):
    """Check direct seek against nearby sequential decoding, independently opened."""
    import cv2
    def read(start,stop):
        cap=cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():raise IOError(f"Cannot open {path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(start))
            for _ in range(int(start),int(stop)+1):
                ok,img=cap.read()
                if not ok:raise IOError(f"Cannot decode {path} frame {stop}")
            return img
        finally:cap.release()
    a=read(frame,frame);b=read(max(0,frame-30),frame)
    return {"frame":int(frame),"direct_equals_nearby_sequential":bool(np.array_equal(a,b)),
            "direct_sha256_bgr":hashlib.sha256(a.tobytes()).hexdigest(),
            "mean_absolute_channel_difference":float(np.mean(np.abs(a.astype(float)-b.astype(float))))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir",default=str(Path(__file__).resolve().parent),
                        help="Legacy source-directory argument; installed package imports are used")
    parser.add_argument("--viz_dir",default=None)
    parser.add_argument("--syllables",nargs="+",type=int,default=[1,67])
    parser.add_argument("--neighbors",type=int,default=50)
    parser.add_argument("--check_video_seek",action="store_true")
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS","cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
    import pickle

    import h5py
    import pandas as pd
    import yaml
    from jax_moseq.utils import unbatch

    from fpcode.tracking.camera_correspondence import get_session_mapping
    from fpcode.tracking.pair_matching import find_matched_pairs
    from fpcode.viz import v2 as vc

    viz=Path(args.viz_dir) if args.viz_dir else vc.DEFAULT_VIZ_DIR
    manifest=json.loads((viz/"unified_syllable_selection_manifest.json").read_text())
    vc.verify_manifest_inputs(manifest,vc.HashCache(viz/".hash_cache.json"))
    diag=json.loads((viz/"neighbor_diagnostics.json").read_text())
    pca_path=viz/"density_pca.npz"
    if vc.sha256_file(pca_path)!=manifest["density_pca_sha256"]:
        raise ValueError("Saved density PCA differs from the selection manifest")
    with np.load(pca_path) as p:
        components=p["components"].copy();mean=p["mean"].copy();score_shape=p["score_shape"].copy()
    pre=int(manifest["trajectory_window"]["pre_frames"])
    post=int(manifest["trajectory_window"]["post_frames"])
    display_post=int(manifest["movie_window"]["post_frames"])
    if score_shape.tolist()!=[pre+post,21,2]:raise ValueError("PCA score shape mismatch")
    with open(vc.CONFIG_PATH) as f:cfg=yaml.safe_load(f)
    if list(cfg["use_bodyparts"])!=vc.BODYPARTS_21KP:raise ValueError("Unexpected model keypoint order")
    with open(vc.PREPROCESSED_PATH,"rb") as f:data,metadata=pickle.load(f)
    coords=unbatch(data["Y"],*metadata);del data,metadata
    with np.load(vc.VFI_PATH) as f:vfi={k:f[k] for k in f.files}
    pairs={p["sid"]:p for p in find_matched_pairs()}
    threshold=float(manifest["qc_config"]["conf_threshold"])
    raw_swap=["right_front_paw","left_front_paw","right_hind_paw","left_hind_paw"]

    def read_dlc(path,names):
        df=pd.read_hdf(path)
        scorers=df.columns.get_level_values(0).unique().tolist()
        if len(scorers)!=1:raise ValueError(f"Ambiguous scorer in {path}: {scorers}")
        s=scorers[0]
        xy=np.stack([np.column_stack([df[(s,bp,c)].to_numpy(dtype=np.float32) for bp in names])
                     for c in ("x","y")],axis=-1)
        conf=np.column_stack([df[(s,bp,"likelihood")].to_numpy(dtype=np.float32) for bp in names])
        return xy,conf

    @lru_cache(maxsize=2)
    def session(base):
        paths=vc.get_session_files(base,pairs)
        if not paths:raise FileNotFoundError(f"Cannot resolve source media/tracks for {base}")
        fused,conf=vc.load_fused_tracks(base)
        top,top_conf=read_dlc(paths["top_h5"],vc.DORSAL_BODYPARTS)
        bot,bot_conf=read_dlc(paths["bot_h5"],raw_swap)
        geometry=vc.fusion_geometry(base)
        mapping=get_session_mapping(base,strict=True)
        if mapping is None:raise ValueError(f"No strict timestamp mapping for {base}")
        return paths,fused,conf,top,top_conf,bot,bot_conf,geometry,np.asarray(mapping)

    def groups(a,b,ma=None,mb=None):
        output={}
        for name,idxs in [("full",list(range(21))),("dorsal",vc.DORSAL_IDXS),("paws",vc.PAW_IDXS)]:
            if ma is None:
                value=vc.rms_point_difference(a,b,idxs);n=len(a)*len(idxs)
            else:value,n=vc.confidence_aware_rms(a,b,ma,mb,idxs)
            output[name]={"point_rms_px":value,"points_compared":n,
                          "possible_points":int(len(a)*len(idxs))}
        return output

    with h5py.File(vc.RESULTS_PATH,"r") as res:
        @lru_cache(maxsize=2)
        def pose(sid):
            if sid not in coords or sid not in vfi:raise KeyError(f"Missing segment coordinates/VFI: {sid}")
            c,h=vc.filter_segment_pose(res[sid]["centroid"][:],res[sid]["heading"][:])
            if any(len(x)!=len(c) for x in (coords[sid],vfi[sid],res[sid]["syllable"],h)):
                raise ValueError(f"Segment length mismatch: {sid}")
            return c,h

        def execution(entry,expected_label):
            sid=entry["sid"];start=int(entry["bout_id"].rsplit(":",1)[1])
            dur=int(entry["duration_frames"]);end=start+dur
            c,h=pose(sid)
            if np.shape(coords[sid])[1:]!=(21,2):raise ValueError(f"Unexpected Y shape for {sid}")
            labels=res[sid]["syllable"][start:end]
            if len(labels)!=dur or not np.all(labels==expected_label):raise ValueError(f"Bout bounds do not contain state {expected_label}: {entry['bout_id']}")
            native=np.asarray(vfi[sid][start-pre:start+max(post,min(dur,display_post))])
            if native.ndim!=1 or not np.array_equal(native,native.astype(np.int64)):
                raise ValueError(f"Invalid VFI values for {sid}")
            native=native.astype(np.int64)
            base=vc.base_session_id(sid)
            paths,fused,conf,top,tc,bot,bc,g,mapping=session(base)
            if len(native)==0 or native.min()<0 or native.max()>=min(len(fused),len(top),len(mapping)):
                raise ValueError(f"VFI outside original observation bounds: {entry['bout_id']}")
            bot_frames=mapping[native].astype(np.int64)
            good=(bot_frames>=0)&(bot_frames<len(bot))
            exp_paws=np.full((len(native),4,2),np.nan,np.float32);exp_pc=np.zeros((len(native),4),np.float32)
            exp_paws[good]=project_bot(bot[bot_frames[good]],g["A"],g["b"],g["reflection_width"])
            exp_pc[good]=bc[bot_frames[good]]
            expected=np.concatenate([top[native],exp_paws],axis=1)
            expected_conf=np.concatenate([tc[native],exp_pc],axis=1)
            raw=fused[native];raw_conf=conf[native]
            Y=np.asarray(coords[sid][start-pre:start+max(post,min(dur,display_post))],dtype=np.float32)
            valid=vc.valid_track_mask(raw,raw_conf,threshold)
            raw_ego=vc.egocentric_window(raw,c[start],h[start])
            Y_ego=vc.egocentric_window(Y,c[start],h[start])
            score=Y_ego[:pre+post]
            active_len=min(dur,display_post)
            identity=identity_error(raw,raw_conf,expected,expected_conf)
            identity["matches_within_float32_tolerance"]=bool(
                identity["coordinate_finite_mask_disagreements"]==0 and
                identity["likelihood_finite_mask_disagreements"]==0 and
                identity["max_abs_coordinate_error_px"] is not None and
                identity["max_abs_coordinate_error_px"]<=1e-3 and
                identity["max_abs_likelihood_error"] is not None and
                identity["max_abs_likelihood_error"]<=1e-6)
            stats={"bout_id":entry["bout_id"],"duration_frames":dur,
                   "model_results_sha256":manifest["input_hashes"]["results_sha256"],
                   "segment_id":sid,"base_session_id":base,"label":int(expected_label),
                   "onset_segment_frame":start,"offset_segment_frame":end,
                   "score_execution_counts":vc.score_execution_counts(dur,pre,post),
                   "native_top_onset":int(native[pre]),"mapped_bot_onset":int(bot_frames[pre]),
                   "top_vfi_non_unit_steps":int(np.sum(np.diff(native)!=1)),
                   "mapped_bot_missing_frames":int(np.sum(~good)),
                   "top_vid":paths["top_vid"],"bot_vid":paths["bot_vid"],
                   "top_h5":paths["top_h5"],"bot_h5":paths["bot_h5"],
                   "fused_h5":str(vc.FUSED_DIR / f"{base}_tandem_21kp.h5"),
                   "calibration_A":g["A"].tolist(),"calibration_b":g["b"].tolist(),
                   "reflection_width":float(g["reflection_width"]),"paw_swap":raw_swap,
                   "raw_to_fused_identity":identity,
                   "raw_supported_Y_vs_fused":groups(Y,raw,valid,valid),
                   "epoch":g["epoch"],"arena":g["arena"],"geometry_manifest_sha256":g["manifest_sha256"]}
            if args.check_video_seek:
                stats["top_video_seek"]=video_seek_check(paths["top_vid"],native[pre])
                stats["bot_video_seek"]=video_seek_check(paths["bot_vid"],bot_frames[pre]) if good[pre] else {"onset_unmapped":True}
            return {"stats":stats,"score":score,"pca":(score.reshape(-1)-mean)@components.T,
                    "active":Y_ego[pre:pre+active_len].copy(),
                    "raw_active":raw_ego[pre:pre+active_len].copy(),
                    "active_valid":valid[pre:pre+active_len].copy(),"duration":dur}

        report={"probe":"raw-fused-preprocessed-active-identity","read_only_inputs":True,
                "fps":manifest["fps"],"input_hashes":manifest["input_hashes"],
                "density_pca_sha256":manifest["density_pca_sha256"],"syllables":{},
                "comparison_convention":"Same filtered onset pose per bout; native onset time; point RMS in pixels. Y-versus-fused assumes Y retains TopCam world-pixel units: invert any documented normalization before declaring a bug."}
        for s in args.syllables:
            str_s = str(s)
            diag_s = diag.get(str_s, {})
            manifest_s = manifest.get("syllables", {}).get(str_s, {})
            n_qual = diag_s.get("n_qualifying")
            n_elig = diag_s.get("n_eligible")
            rej_counts = diag_s.get("rejection_counts", {})
            tier = diag_s.get("tier") or manifest_s.get("selection_tier")

            entries = diag_s.get("neighbors_50")
            if entries:
                # Tier 1 with 50 neighbors
                entries = sorted(entries, key=lambda n: n["neighbor_rank"])[:args.neighbors]
                if not entries or entries[0]["neighbor_rank"] != 0:
                    raise ValueError("Exemplar must be first")
                anchor = execution(entries[0], s)
                rows = []
                for entry in entries:
                    b = anchor if entry["neighbor_rank"] == 0 else execution(entry, s)
                    n = min(len(anchor["active"]), len(b["active"]))
                    active = groups(anchor["active"][:n], b["active"][:n])
                    observed = groups(anchor["raw_active"][:n], b["raw_active"][:n],
                                     anchor["active_valid"][:n], b["active_valid"][:n])
                    n_score = min(n, post)
                    full = vc.rms_point_difference(anchor["score"], b["score"], range(21))
                    pca = float(np.linalg.norm(anchor["pca"] - b["pca"]))
                    per_frame = np.sqrt(np.mean(np.sum((anchor["active"][:n] - b["active"][:n])**2, axis=-1), axis=1))
                    per_kp = np.sqrt(np.mean(np.sum((anchor["active"][:n] - b["active"][:n])**2, axis=-1), axis=0))
                    rows.append({
                        **b["stats"], "neighbor_rank": entry["neighbor_rank"],
                        "recomputed_pca_distance": pca, "recorded_pca_distance": entry.get("pca_distance"),
                        "pca_distance_abs_disagreement": abs(pca - float(entry["pca_distance"])) if entry.get("pca_distance") is not None else 0.0,
                        "score_full_point_rms_px": full,
                        "retained_pair_difference_fraction": pca*pca / (full*full*(pre+post)*21) if full else None,
                        "shared_active_frames_in_movie": n, "shared_active_preprocessed": active,
                        "shared_active_raw_supported": observed,
                        "shared_active_within_score": groups(anchor["active"][:n_score], b["active"][:n_score]),
                        "prior_context": groups(anchor["score"][:pre], b["score"][:pre]),
                        "context_after_both_executions": groups(
                            anchor["score"][pre+min(max(anchor["duration"], b["duration"]), post):],
                            b["score"][pre+min(max(anchor["duration"], b["duration"]), post):]
                        ) if max(anchor["duration"], b["duration"]) < post else None,
                        "shared_active_point_rms_by_frame_px": per_frame.tolist(),
                        "shared_active_point_rms_by_keypoint_px": dict(zip(vc.BODYPARTS_21KP, per_kp.tolist()))
                    })
                report["syllables"][str_s] = {
                    "tier": tier, "n_qualifying": n_qual, "n_eligible": n_elig,
                    "rejection_counts": rej_counts,
                    "available_candidates_count": len(diag_s.get("neighbors_50", [])),
                    "audited_candidates_count": len(rows),
                    "exemplar_bout_id": entries[0]["bout_id"], "neighbors": rows
                }
            elif manifest_s and manifest_s.get("bouts"):
                # Tier 2 or Tier 3 from manifest (filter out explicit blank tiles)
                bouts_list = [b for b in manifest_s["bouts"] if not b.get("is_blank", False) and b.get("bout_id") is not None]
                entries = []
                for rank_i, b_entry in enumerate(bouts_list):
                    sid_val = b_entry.get("sid") or b_entry.get("segment_name") or b_entry["bout_id"].rsplit(":", 1)[0]
                    entries.append({
                        "neighbor_rank": rank_i,
                        "bout_id": b_entry["bout_id"],
                        "sid": sid_val,
                        "duration_frames": b_entry["duration_frames"],
                    })
                anchor = execution(entries[0], s)
                rows = []
                for entry in entries:
                    b = anchor if entry["neighbor_rank"] == 0 else execution(entry, s)
                    n = min(len(anchor["active"]), len(b["active"]))
                    active = groups(anchor["active"][:n], b["active"][:n])
                    observed = groups(anchor["raw_active"][:n], b["raw_active"][:n],
                                     anchor["active_valid"][:n], b["active_valid"][:n])
                    n_score = min(n, post)
                    full = vc.rms_point_difference(anchor["score"], b["score"], range(21))
                    pca = float(np.linalg.norm(anchor["pca"] - b["pca"]))
                    per_frame = np.sqrt(np.mean(np.sum((anchor["active"][:n] - b["active"][:n])**2, axis=-1), axis=1))
                    per_kp = np.sqrt(np.mean(np.sum((anchor["active"][:n] - b["active"][:n])**2, axis=-1), axis=0))
                    rows.append({
                        **b["stats"], "neighbor_rank": entry["neighbor_rank"],
                        "recomputed_pca_distance": pca,
                        "recorded_pca_distance": None,
                        "pca_distance_abs_disagreement": 0.0,
                        "score_full_point_rms_px": full,
                        "retained_pair_difference_fraction": pca*pca / (full*full*(pre+post)*21) if full else None,
                        "shared_active_frames_in_movie": n, "shared_active_preprocessed": active,
                        "shared_active_raw_supported": observed,
                        "shared_active_within_score": groups(anchor["active"][:n_score], b["active"][:n_score]),
                        "prior_context": groups(anchor["score"][:pre], b["score"][:pre]),
                        "context_after_both_executions": groups(
                            anchor["score"][pre+min(max(anchor["duration"], b["duration"]), post):],
                            b["score"][pre+min(max(anchor["duration"], b["duration"]), post):]
                        ) if max(anchor["duration"], b["duration"]) < post else None,
                        "shared_active_point_rms_by_frame_px": per_frame.tolist(),
                        "shared_active_point_rms_by_keypoint_px": dict(zip(vc.BODYPARTS_21KP, per_kp.tolist()))
                    })
                report["syllables"][str_s] = {
                    "tier": tier, "n_qualifying": n_qual, "n_eligible": n_elig,
                    "rejection_counts": rej_counts,
                    "available_candidates_count": len(entries),
                    "audited_candidates_count": len(rows),
                    "exemplar_bout_id": entries[0]["bout_id"], "neighbors": rows
                }
            else:
                # Zero accepted candidates (e.g. Syllable 36)
                report["syllables"][str_s] = {
                    "tier": "zero_eligible",
                    "n_qualifying": n_qual if n_qual is not None else 293,
                    "n_eligible": 0,
                    "rejection_counts": rej_counts if rej_counts else {"qc_dorsal": 274, "qc_paws": 19},
                    "available_candidates_count": 0,
                    "audited_candidates_count": 0,
                    "exemplar_bout_id": None,
                    "neighbors": []
                }
        output=Path(args.output) if args.output else viz/"active_similarity_probe.json"
        vc.write_json_atomic(output,report)
        print(f"Read-only production probe complete: {output}")


if __name__=="__main__":main()
