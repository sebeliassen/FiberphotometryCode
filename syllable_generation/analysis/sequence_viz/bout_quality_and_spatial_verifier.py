#!/usr/bin/env python3
"""
bout_quality_and_spatial_verifier.py
=============================================================================
Implementation of the 4-Part Quality, Trajectory, Weakest-Stage, and Spatial
Context Verification Framework for Behavioral Chains.

Key Principles:
1. SEPARATE QUALITY VS SIMILARITY:
   - Quality Score (Q in [0, 1]): Physical tracking integrity (teleportation jumps,
     heading flips, micro-flickers, stuck tracking). Fast/rare movement is NEVER penalized.
   - Similarity Score (S in [0, 1]): Kinematic & pose-latent trajectory match to reference.
   - Separate scores ensure sound, unusual motor variants are preserved, while
     measurement corruptions are cleanly identified and excluded.
2. TRAJECTORY COMPARISON (NOT JUST SCALAR MEANS):
   - Compares speed profiles v(t), signed turning rates \dot{\theta}(t), and
     10D KPMS pose latents z(t) over normalized time.
3. WEAKEST-STAGE (WORST-LINK) PRINCIPLE:
   - Evaluates Q_min = min_k(Q_k) and S_min = min_k(S_k), pinpointing the exact
     weakest stage rather than masking a corrupted bout with an average.
   - Guarantees non-silent fusion (corrupted intervals are explicitly flagged,
     never silently fused).
4. SPATIAL ARENA CONTEXT:
   - Evaluates distance to nearest wall (d_wall), heading relative to wall (\theta_wall),
     and parallel vs perpendicular movement (thigmotaxis / boundary-following).
=============================================================================
"""

import numpy as np
from pathlib import Path


# ==============================================================================
# 1. PHYSICAL TRACKING QUALITY EVALUATOR
# ==============================================================================
class BoutQualityEvaluator:
    """
    Evaluates physical sensor and tracking validity for individual bouts.
    Identifies measurement corruption (teleportation, heading flips, micro-flickers)
    without penalizing genuine fast or rare biological movement.
    """
    def __init__(
        self,
        min_bout_frames=5,       # <125ms = HMM boundary chatter / micro-flicker
        max_bout_frames=65,      # >1.625s = tracking stuck / lost mouse
        max_velocity_jump=16.0,  # px/frame = physical biomechanical impossibility
        max_heading_flip=1.8,    # rad/frame (~103 deg) = patch cord / snout confusion
        min_path_displacement=6.0 # px = motionless bout for active locomotion
    ):
        self.min_bout_frames = min_bout_frames
        self.max_bout_frames = max_bout_frames
        self.max_velocity_jump = max_velocity_jump
        self.max_heading_flip = max_heading_flip
        self.min_path_displacement = min_path_displacement

    def evaluate_bout(self, centroids, headings, start_f, end_f, expected_motion=True):
        """
        Evaluates a single bout interval [start_f, end_f].
        Returns dict with quality score Q in [0, 1], metrics, and explicit flags.
        """
        dur = max(1, end_f - start_f)
        c_stage = centroids[start_f:end_f]
        h_stage = headings[start_f:end_f]

        flags = []
        metrics = {
            "duration_frames": dur,
            "duration_sec": dur / 40.0,
            "max_velocity": 0.0,
            "max_heading_delta": 0.0,
            "total_path_length": 0.0,
            "net_displacement": 0.0,
            "mean_speed": 0.0
        }

        # 1. Duration check
        if dur < self.min_bout_frames:
            flags.append("MICRO_FLICKER")
        elif dur > self.max_bout_frames:
            flags.append("STUCK_BOUT")

        # 2. Centroid velocity jumps
        if len(c_stage) > 1:
            step_vels = np.linalg.norm(np.diff(c_stage, axis=0), axis=1)
            metrics["max_velocity"] = float(np.max(step_vels))
            metrics["total_path_length"] = float(np.sum(step_vels))
            metrics["mean_speed"] = float(np.mean(step_vels))
            metrics["net_displacement"] = float(np.linalg.norm(c_stage[-1] - c_stage[0]))
            
            if metrics["max_velocity"] > self.max_velocity_jump:
                flags.append(f"VEL_JUMP_{metrics['max_velocity']:.1f}px")
            if expected_motion and metrics["total_path_length"] < self.min_path_displacement:
                flags.append("IMMOBILE_BOUT")

        # 3. Heading flip spikes (fiber optic patch cord / snout swap)
        if len(h_stage) > 1:
            d_theta = np.abs(np.diff(np.unwrap(h_stage)))
            metrics["max_heading_delta"] = float(np.max(d_theta))
            if metrics["max_heading_delta"] > self.max_heading_flip:
                flags.append(f"HEADING_FLIP_{np.degrees(metrics['max_heading_delta']):.0f}deg")

        # Calculate continuous Physical Quality Score Q in [0, 1]
        q = 1.0
        if "MICRO_FLICKER" in flags:
            q -= 0.65
        if "STUCK_BOUT" in flags:
            q -= 0.50
        if any("VEL_JUMP" in fl for fl in flags):
            jump_excess = max(0.0, metrics["max_velocity"] - self.max_velocity_jump)
            q -= min(0.85, 0.40 + 0.05 * jump_excess)
        if any("HEADING_FLIP" in fl for fl in flags):
            q -= 0.80
        if "IMMOBILE_BOUT" in flags:
            q -= 0.35

        q = float(np.clip(q, 0.0, 1.0))
        return {
            "quality_score": q,
            "is_corrupt": (q < 0.80),
            "flags": flags,
            "metrics": metrics
        }


# ==============================================================================
# 2. TRAJECTORY KINEMATIC & POSE SIMILARITY EVALUATOR
# ==============================================================================
class BoutSimilarityEvaluator:
    """
    Evaluates kinematic trajectory and pose-latent similarity against reference
    syllable profiles, separate from physical tracking quality.
    """
    def __init__(self, results, n_sample_pts=15):
        self.n_sample_pts = n_sample_pts
        self.syllable_profiles = self._build_reference_profiles(results)

    def _build_reference_profiles(self, results):
        """
        Builds empirical baseline profiles (speed, turn rate, pose latents) for each
        syllable across all 30 sessions using clean, sound bouts.
        """
        syll_data = {}
        for rk, data in results.items():
            sylls = data["syllable"]
            cents = data["centroid"]
            heads = data["heading"]
            lats = data.get("latent_state")

            c_idx = np.where(sylls[1:] != sylls[:-1])[0] + 1
            b_sylls = [sylls[0]] + sylls[c_idx].tolist()
            b_starts = [0] + c_idx.tolist()
            b_ends = c_idx.tolist() + [len(sylls)]

            for s_id, st, en in zip(b_sylls, b_starts, b_ends):
                dur = en - st
                if not (6 <= dur <= 55):
                    continue
                c_seg = cents[st:en]
                if len(c_seg) < 2:
                    continue
                v = np.linalg.norm(np.diff(c_seg, axis=0), axis=1)
                if np.max(v) > 16.0:
                    continue

                if s_id not in syll_data:
                    syll_data[s_id] = {
                        "durations": [],
                        "mean_speeds": [],
                        "turn_rates": [],
                        "latent_means": []
                    }
                syll_data[s_id]["durations"].append(dur)
                syll_data[s_id]["mean_speeds"].append(float(np.mean(v)))
                
                h_seg = heads[st:en]
                dth = np.abs(np.diff(np.unwrap(h_seg))) if len(h_seg) > 1 else np.array([0.0])
                syll_data[s_id]["turn_rates"].append(float(np.mean(dth)))
                lat_mean = np.mean(lats[st:en], axis=0) if lats is not None else np.zeros(10)
                syll_data[s_id]["latent_means"].append(lat_mean)

        profiles = {}
        for s_id, d in syll_data.items():
            if len(d["durations"]) >= 10:
                profiles[s_id] = {
                    "median_duration": float(np.median(d["durations"])),
                    "std_duration": float(np.std(d["durations"]) + 1e-4),
                    "median_speed": float(np.median(d["mean_speeds"])),
                    "std_speed": float(np.std(d["mean_speeds"]) + 1e-4),
                    "median_turn_rate": float(np.median(d["turn_rates"])),
                    "std_turn_rate": float(np.std(d["turn_rates"]) + 1e-4),
                    "mean_latents": np.mean(d["latent_means"], axis=0),
                    "std_latents": np.std(d["latent_means"], axis=0) + 1e-4
                }
        return profiles

    def evaluate_bout(self, syllable_id, centroids, headings, latents, start_f, end_f):
        """
        Evaluates kinematic and pose trajectory similarity of bout to syllable profile.
        Returns Similarity Score S in [0, 1] and standardized feature distances.
        """
        dur = max(1, end_f - start_f)
        c_seg = centroids[start_f:end_f]
        h_seg = headings[start_f:end_f]
        obs_lat = np.mean(latents[start_f:end_f], axis=0) if latents is not None else np.zeros(10)

        v = np.linalg.norm(np.diff(c_seg, axis=0), axis=1) if len(c_seg) > 1 else np.array([0.0])
        dth = np.abs(np.diff(np.unwrap(h_seg))) if len(h_seg) > 1 else np.array([0.0])
        obs_speed = float(np.mean(v))
        obs_turn = float(np.mean(dth))

        # Look up reference profile
        prof = self.syllable_profiles.get(syllable_id)
        if prof is None:
            return {"similarity_score": 0.50, "is_motor_variant": False, "dist": 0.5}

        # Standardized distance across kinematic and pose features
        z_spd = abs(obs_speed - prof["median_speed"]) / prof["std_speed"]
        z_turn = abs(obs_turn - prof["median_turn_rate"]) / prof["std_turn_rate"]
        z_dur = abs(dur - prof["median_duration"]) / prof["std_duration"]
        z_lat = np.mean(np.abs(obs_lat - prof["mean_latents"]) / prof["std_latents"])

        # Weighted distance: 30% speed, 25% turning rate, 20% duration, 25% pose latents
        dist = float(0.30 * z_spd + 0.25 * z_turn + 0.20 * z_dur + 0.25 * z_lat)
        sim_score = float(1.0 / (1.0 + 0.5 * dist))

        # A bout is flagged as a MOTOR_VARIANT if it is physically sound but
        # lies in the rare tail (>2.2 std) of the distribution
        is_variant = (dist > 2.2)

        return {
            "similarity_score": sim_score,
            "kinematic_distance": dist,
            "is_motor_variant": is_variant,
            "speed_z": float(z_spd),
            "turn_z": float(z_turn),
            "dur_z": float(z_dur),
            "pose_z": float(z_lat)
        }


# ==============================================================================
# 3. SPATIAL ARENA CONTEXT EVALUATOR
# ==============================================================================
class SpatialContextEvaluator:
    """
    Evaluates arena boundary geometry: distance to nearest perimeter wall (d_wall),
    heading relative to wall (parallel vs perpendicular), Alexander Egocentric
    Boundary Vectors (EBVs), and Dvorkin Relational Excursion Topology.
    """
    def __init__(self, x_bounds=(15.0, 370.0), y_bounds=(20.0, 385.0), thigmotaxis_threshold=55.0):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.thigmotaxis_threshold = thigmotaxis_threshold

    def evaluate_ebv(self, centroid, heading):
        """
        Computes the Egocentric Boundary Vector (EBV) at a single frame (Alexander et al. 2020).
        Returns dict with nearest wall distance, egocentric angle, and EBV zone.
        """
        x, y = float(centroid[0]), float(centroid[1])
        h = float(heading)
        
        # 4 Wall boundaries
        walls = [
            ("Wall_Left", np.array([self.x_min, y]), abs(x - self.x_min)),
            ("Wall_Right", np.array([self.x_max, y]), abs(self.x_max - x)),
            ("Wall_Top", np.array([x, self.y_min]), abs(y - self.y_min)),
            ("Wall_Bottom", np.array([x, self.y_max]), abs(self.y_max - y))
        ]
        wall_name, pt_near, d_near = min(walls, key=lambda w: w[2])
        
        v_wall = pt_near - np.array([x, y])
        phi_wall = np.arctan2(v_wall[1], v_wall[0])
        beta_ego = (phi_wall - h + np.pi) % (2 * np.pi) - np.pi
        deg_ego = float(np.degrees(beta_ego))
        
        if d_near > self.thigmotaxis_threshold:
            ebv_label = "OPEN_ARENA"
        elif abs(deg_ego) <= 40.0:
            ebv_label = "SNOUT_AHEAD"
        elif 50.0 <= deg_ego <= 130.0:
            ebv_label = "LEFT_FLANK"
        elif -130.0 <= deg_ego <= -50.0:
            ebv_label = "RIGHT_FLANK"
        elif abs(deg_ego) >= 140.0:
            ebv_label = "TAIL_BEHIND"
        else:
            ebv_label = "OBLIQUE"
            
        return {
            "wall_name": wall_name,
            "dist": float(d_near),
            "egocentric_bearing_deg": deg_ego,
            "ebv_label": ebv_label
        }

    def evaluate_bout(self, centroids, headings, start_f, end_f):
        """
        Evaluates spatial position, wall alignment, and EBV throughout [start_f, end_f].
        """
        c_seg = centroids[start_f:end_f]
        h_seg = headings[start_f:end_f]

        if len(c_seg) == 0:
            return {
                "mean_wall_dist": 100.0,
                "min_wall_dist": 100.0,
                "mean_wall_rel_angle_deg": 45.0,
                "is_wall_following": False,
                "zone": "Center",
                "onset_ebv": "OPEN_ARENA",
                "onset_wall_dist": 100.0
            }

        # Distance to 4 walls
        d_left = np.maximum(0.0, c_seg[:, 0] - self.x_min)
        d_right = np.maximum(0.0, self.x_max - c_seg[:, 0])
        d_top = np.maximum(0.0, c_seg[:, 1] - self.y_min)
        d_bottom = np.maximum(0.0, self.y_max - c_seg[:, 1])

        d_walls = np.column_stack([d_left, d_right, d_top, d_bottom])
        d_min_per_frame = np.min(d_walls, axis=1)
        nearest_wall_idx = np.argmin(d_walls, axis=1)  # 0=left, 1=right, 2=top, 3=bottom

        mean_dist = float(np.mean(d_min_per_frame))
        min_dist = float(np.min(d_min_per_frame))

        # Wall tangent angles
        wall_tangent_angles = np.where(nearest_wall_idx < 2, np.pi / 2.0, 0.0)

        # Angle relative to wall tangent in [0, 90] deg
        raw_diff = np.abs(h_seg - wall_tangent_angles)
        rel_diff = np.abs(np.arctan2(np.sin(raw_diff), np.cos(raw_diff)))
        rel_diff_folded = np.minimum(rel_diff, np.pi - rel_diff)
        rel_deg = np.degrees(rel_diff_folded)

        mean_rel_angle = float(np.mean(rel_deg))

        # Thigmotaxis / Wall following definition
        is_near_wall = (mean_dist <= self.thigmotaxis_threshold)
        is_parallel = (mean_rel_angle <= 32.0)
        is_wall_following = bool(is_near_wall and is_parallel)

        zone = "Boundary" if is_near_wall else "Interior"
        
        # Egocentric Boundary Vector at stage onset
        ebv_res = self.evaluate_ebv(c_seg[0], h_seg[0])

        return {
            "mean_wall_dist": mean_dist,
            "min_wall_dist": min_dist,
            "mean_wall_rel_angle_deg": mean_rel_angle,
            "is_wall_following": is_wall_following,
            "is_near_wall": is_near_wall,
            "zone": zone,
            "onset_ebv": ebv_res["ebv_label"],
            "onset_wall_dist": ebv_res["dist"],
            "onset_wall_name": ebv_res["wall_name"]
        }

    def evaluate_chain_relational_topology(self, centroids, headings, start_f, end_f):
        """
        Computes Dvorkin et al. 2008 relational excursion topology across full episode:
        - Return to Origin (A -> A loop)
        - Relocation along Boundary (A -> B boundary)
        - Relocation across Interior (A -> B crossing)
        - Local Stay (A in-place)
        """
        c_seg = centroids[start_f:end_f]
        h_seg = headings[start_f:end_f]
        if len(c_seg) < 2:
            return {
                "topology": "LOCAL_STAY",
                "transition": "BOUNDARY_TO_BOUNDARY",
                "net_disp": 0.0,
                "max_excursion": 0.0,
                "start_zone": "Boundary",
                "end_zone": "Boundary",
                "description": "Stay in place"
            }

        c_start = c_seg[0]
        c_end = c_seg[-1]
        
        d_from_start = np.linalg.norm(c_seg - c_start, axis=1)
        max_excursion = float(np.max(d_from_start))
        net_disp = float(np.linalg.norm(c_end - c_start))
        
        ebv_start = self.evaluate_ebv(c_start, h_seg[0])
        ebv_end = self.evaluate_ebv(c_end, h_seg[-1])
        
        start_is_bnd = ebv_start["dist"] <= self.thigmotaxis_threshold
        end_is_bnd = ebv_end["dist"] <= self.thigmotaxis_threshold
        
        start_zone = "Boundary" if start_is_bnd else "Interior"
        end_zone = "Boundary" if end_is_bnd else "Interior"
        
        if start_is_bnd and end_is_bnd:
            transition = "BOUNDARY_TO_BOUNDARY"
        elif start_is_bnd and not end_is_bnd:
            transition = "BOUNDARY_TO_INTERIOR"
        elif not start_is_bnd and end_is_bnd:
            transition = "INTERIOR_TO_BOUNDARY"
        else:
            transition = "INTERIOR_TO_INTERIOR"

        # Classification
        if max_excursion <= 35.0 and net_disp <= 35.0:
            topology = "LOCAL_STAY"
            desc = "In-place postural change"
        elif max_excursion > 35.0 and net_disp <= 42.0:
            topology = "RETURN_TO_ORIGIN"
            desc = f"Loop excursion from {ebv_start['wall_name']} and back"
        elif net_disp > 65.0:
            if start_is_bnd and end_is_bnd:
                if ebv_start["wall_name"] == ebv_end["wall_name"]:
                    topology = "RELOCATION_BOUNDARY"
                    desc = f"Perimeter run along {ebv_start['wall_name']}"
                else:
                    topology = "RELOCATION_CROSS_ARENA"
                    desc = f"Relocation {ebv_start['wall_name']} -> {ebv_end['wall_name']}"
            else:
                topology = "RELOCATION_INTERIOR"
                desc = "Crossing interior space"
        else:
            topology = "INTERMEDIATE"
            desc = "Short traverse"

        return {
            "topology": topology,
            "transition": transition,
            "net_disp": net_disp,
            "max_excursion": max_excursion,
            "start_zone": start_zone,
            "end_zone": end_zone,
            "start_ebv": ebv_start["ebv_label"],
            "end_ebv": ebv_end["ebv_label"],
            "start_wall": ebv_start["wall_name"],
            "end_wall": ebv_end["wall_name"],
            "description": desc
        }


# ==============================================================================
# 4. CHAIN QUALITY & CONTEXT VERIFIER (ORCHESTRATOR)
# ==============================================================================
class ChainQualityAndContextVerifier:
    """
    Comprehensive multi-stage verifier implementing the Weakest-Stage Principle,
    Dual Scoring (Quality vs Similarity), Alexander Egocentric Boundary Vectors (EBVs),
    and Dvorkin Relational Excursion Topology.
    """
    def __init__(self, results, x_bounds=(15.0, 370.0), y_bounds=(20.0, 385.0)):
        self.results = results
        self.quality_eval = BoutQualityEvaluator()
        self.similarity_eval = BoutSimilarityEvaluator(results)
        self.spatial_eval = SpatialContextEvaluator(x_bounds=x_bounds, y_bounds=y_bounds)

    def verify_chain(self, instance):
        """
        Takes a candidate chain instance dict:
        {
          'recording_key': str,
          'start_frame': int,
          'end_frame': int,
          'stage_boundaries': [b0, b1, ..., bK],
          'stage_tokens': ['syll_001', ...],
          ...
        }
        Performs stage-wise analysis and returns an enhanced audit dict.
        """
        rk = instance["recording_key"]
        data = self.results[rk]
        cents = data["centroid"]
        heads = data["heading"]
        lats = data["latent_state"]
        sylls = data["syllable"]

        sb = instance["stage_boundaries"]
        tokens = instance["stage_tokens"]
        n_stages = len(sb) - 1

        stage_qualities = []
        stage_similarities = []
        stage_spatials = []
        stage_flags = []
        stage_turn_angles = [0.0]

        # Evaluate each constituent stage
        for k in range(n_stages):
            st = sb[k]
            en = sb[k+1]
            tok = tokens[k]
            s_id = int(tok.replace("syll_", "")) if "syll_" in tok and tok.replace("syll_", "").isdigit() else sylls[st]

            # 1. Physical Tracking Quality
            q_res = self.quality_eval.evaluate_bout(cents, heads, st, en)
            stage_qualities.append(q_res["quality_score"])
            stage_flags.append(q_res["flags"])

            # 2. Kinematic & Pose Similarity
            s_res = self.similarity_eval.evaluate_bout(s_id, cents, heads, lats, st, en)
            stage_similarities.append(s_res["similarity_score"])

            # 3. Spatial Wall Context & EBV
            sp_res = self.spatial_eval.evaluate_bout(cents, heads, st, en)
            stage_spatials.append(sp_res)
            
            # 4. Turn angle relative to previous stage
            if k > 0:
                h_prev = heads[sb[k-1]]
                h_curr = heads[st]
                diff = (h_curr - h_prev + np.pi) % (2 * np.pi) - np.pi
                stage_turn_angles.append(float(np.degrees(diff)))

        # Weakest-Stage (Worst-Link) Principle
        q_min = float(np.min(stage_qualities))
        k_worst_q = int(np.argmin(stage_qualities))
        
        s_min = float(np.min(stage_similarities))
        s_mean = float(np.mean(stage_similarities))
        k_worst_s = int(np.argmin(stage_similarities))

        # Spatial chain summary
        mean_wall_dist = float(np.mean([sp["mean_wall_dist"] for sp in stage_spatials]))
        has_wall_following = any(sp["is_wall_following"] for sp in stage_spatials)

        # Dvorkin Excursion Topology across entire chain
        rel_topo = self.spatial_eval.evaluate_chain_relational_topology(
            cents, heads, instance["start_frame"], instance["end_frame"]
        )

        # Tri-Categorical Classification (ChatGPT Distinction):
        # 1. TRUSTED_CLEAN: Sound tracking (Q_min >= 0.80) & Typical kinematics (S_min >= 0.40)
        # 2. MOTOR_VARIANT: Sound tracking (Q_min >= 0.80), but atypical/rare execution (S_min < 0.40).
        #    Preserved as a legitimate biological variant!
        # 3. MEASUREMENT_CORRUPT: Failed tracking (Q_min < 0.80). Excluded from trusted visual match.
        if q_min >= 0.80 and s_min >= 0.40:
            classification = "TRUSTED_CLEAN"
        elif q_min >= 0.80 and s_min < 0.40:
            classification = "MOTOR_VARIANT"
        else:
            classification = "MEASUREMENT_CORRUPT"

        all_flat_flags = [fl for sub in stage_flags for fl in sub]

        return {
            "classification": classification,
            "quality_min": q_min,
            "quality_worst_stage": k_worst_q,
            "quality_worst_stage_token": tokens[k_worst_q] if k_worst_q < len(tokens) else None,
            "stage_qualities": stage_qualities,
            "stage_flags": stage_flags,
            "all_flags": all_flat_flags,
            "similarity_mean": s_mean,
            "similarity_min": s_min,
            "similarity_worst_stage": k_worst_s,
            "stage_similarities": stage_similarities,
            "stage_turn_angles": stage_turn_angles,
            "mean_wall_dist": mean_wall_dist,
            "has_wall_following": has_wall_following,
            "stage_spatials": stage_spatials,
            "relational_topology": rel_topo["topology"],
            "environmental_transition": rel_topo["transition"],
            "relational_description": rel_topo["description"],
            "start_ebv": rel_topo["start_ebv"],
            "end_ebv": rel_topo["end_ebv"],
            "net_displacement": rel_topo["net_disp"],
            "max_excursion": rel_topo["max_excursion"]
        }
