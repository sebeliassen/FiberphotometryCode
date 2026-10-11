"""Fusion dispatch and result accounting, separate from coordinate algorithms."""

from math import isfinite


def find_matched_pairs(**kwargs):
    if kwargs.get("top_h5_dir") is None or kwargs.get("bot_prod_dir") is None:
        raise ValueError("Pair discovery requires explicit top_h5_dir and bot_prod_dir")
    from fpcode.tracking.pair_matching import find_matched_pairs as implementation
    return implementation(**kwargs)


def fuse_single_pair_static(*args, **kwargs):
    _require_explicit_clocks(kwargs)
    if kwargs.get("transform_info") is None and kwargs.get("transform_path") is None:
        raise ValueError("Static fusion requires an explicit affine calibration or compatibility profile")
    from fpcode.tracking.static_fusion import fuse_single_pair_static as implementation
    return implementation(*args, **kwargs)


def fuse_single_pair(*args, **kwargs):
    _require_explicit_clocks(kwargs)
    from fpcode.tracking.fusion import fuse_single_pair as implementation
    return implementation(*args, **kwargs)


def _require_explicit_clocks(options):
    if not any(options.get(name) is not None for name in (
        "mapping_context", "ttl_cache", "ttl_cache_path", "offset_info", "camera_mapping",
    )):
        raise ValueError("Fusion requires explicit clock inputs or a compatibility profile")


def fusion_options(config):
    source_mode = config.get("mode", "static_21kp")
    # These names are retained by checked-in production/pilot configurations.
    # The old dispatch selected dynamic fusion unless "static" was in the name.
    mode = {"continuous_calibrated": "tandem_23kp", "pilot_static": "static_21kp"}.get(source_mode, source_mode)
    if mode not in {"static_21kp", "static_21kp_v2", "static_23kp", "static_23kp_v2", "tandem_23kp"}:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    width = float(config.get("arena_width", 972.0))
    if not isfinite(width) or width <= 0:
        raise ValueError("arena_width must be finite and positive")
    policy = config.get("unmanifested_policy", "trust")
    if policy not in {"trust", "rebuild"}:
        raise ValueError("unmanifested_policy must be 'trust' or 'rebuild'")
    options = {"arena_width": width, "unmanifested_policy": policy}
    profile = config.get("compatibility_profile")
    if profile is not None:
        if profile != "legacy_multiview":
            raise ValueError(f"Unsupported compatibility_profile: {profile}")
        from fpcode.compat.multiview.profile import TTL_CACHE_PATH
        options["ttl_cache_path"] = str(TTL_CACHE_PATH)
    for name in ("ttl_cache_path", "transform_path", "epoch"):
        if name in config:
            value = config[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            options[name] = value
    for name, default in (("truncate_aborted_tail", True), ("overwrite", False)):
        value = config.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        options[name] = value
    static = mode.startswith("static_")
    if static:
        if config.get("strict_calibration", True) is not True:
            raise ValueError("Static fusion requires strict crystal calibration")
        use_epochs = config.get("use_epochs", False)
        if not isinstance(use_epochs, bool):
            raise ValueError("use_epochs must be a boolean")
        options.update(keypoint_mode="23kp" if "23kp" in mode else "21kp", use_epochs=use_epochs)
        # Static fusion intentionally preserves raw detection likelihoods.
        # Dynamic anchor confidence gating is not a static-mode operation.
    else:
        if "transform_path" in options or "epoch" in options:
            raise ValueError("Affine transform/epoch configuration applies only to static fusion")
        for name, default in (("strict_calibration", True), ("confidence_gating", True), ("require_continuous_sync", True)):
            value = config.get(name, default)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            options[name] = value
    return static, options


def execute_fusion(pairs, config, output_dir, *, static_backend=None, dynamic_backend=None):
    if not pairs:
        raise ValueError("No matched input pairs; fusion would produce no outputs")
    static, options = fusion_options(config)
    backend = (static_backend or fuse_single_pair_static) if static else (dynamic_backend or fuse_single_pair)
    successes, failures = 0, []
    for pair in pairs:
        try:
            supplied = dict(options)
            if static and config.get("compatibility_profile") == "legacy_multiview" and not any(
                name in pair or name in options for name in ("transform_info", "transform_path")
            ):
                from fpcode.compat.multiview.profile import legacy_static_calibration
                transform, source, epoch = legacy_static_calibration(pair.get("arena", "Arena1"),
                    pair.get("ts", pair["sid"][:19]), use_epochs=options.get("use_epochs", False))
                supplied.update(transform_info=transform, transform_source_path=source, epoch=epoch)
            for name in ("ttl_cache", "ttl_cache_path", "offset_info", "mapping_context",
                         "camera_mapping", "calibration_details", "transform_info", "transform_path",
                         "transform_source_path", "epoch"):
                if name in pair:
                    supplied[name] = pair[name]
            # A per-pair clock source overrides configuration/profile defaults.
            # Multiple sources explicitly supplied on the pair remain an error.
            if any(name in pair for name in ("ttl_cache", "offset_info", "mapping_context", "camera_mapping")):
                if "ttl_cache_path" not in pair:
                    supplied.pop("ttl_cache_path", None)
            result = backend(pair, out_dir=output_dir, **supplied)
            if result.get("status") not in {"FUSED", "CACHED", "CACHED_VALID"}:
                raise ValueError(result.get("error", f"Unexpected fusion result: {result.get('status')}"))
            successes += 1
        except Exception as error:
            failures.append({"sid": pair.get("sid"), "error": str(error)})
    status = "SUCCESS" if not failures else "PARTIAL_SUCCESS" if successes else "FAILED"
    return {"status": status, "success": successes, "failed": len(failures), "errors": failures,
            "resolved_options": options}
