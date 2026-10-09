"""Fusion dispatch and result accounting, separate from coordinate algorithms."""

from math import isfinite


def find_matched_pairs(**kwargs):
    from fpcode.tracking.pair_matching import find_matched_pairs as implementation
    return implementation(**kwargs)


def fuse_single_pair_static(*args, **kwargs):
    from fpcode.tracking.static_fusion import fuse_single_pair_static as implementation
    return implementation(*args, **kwargs)


def fuse_single_pair(*args, **kwargs):
    from fpcode.tracking.fusion import fuse_single_pair as implementation
    return implementation(*args, **kwargs)


def fusion_options(config):
    mode = config.get("mode", "static_21kp")
    if mode not in {"static_21kp", "static_21kp_v2", "static_23kp", "static_23kp_v2", "tandem_23kp"}:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    width = float(config.get("arena_width", 972.0))
    if not isfinite(width) or width <= 0:
        raise ValueError("arena_width must be finite and positive")
    policy = config.get("unmanifested_policy", "trust")
    if policy not in {"trust", "rebuild"}:
        raise ValueError("unmanifested_policy must be 'trust' or 'rebuild'")
    options = {"arena_width": width, "unmanifested_policy": policy}
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
            result = backend(pair, out_dir=output_dir, **options)
            if result.get("status") not in {"FUSED", "CACHED", "CACHED_VALID"}:
                raise ValueError(result.get("error", f"Unexpected fusion result: {result.get('status')}"))
            successes += 1
        except Exception as error:
            failures.append({"sid": pair.get("sid"), "error": str(error)})
    status = "SUCCESS" if not failures else "PARTIAL_SUCCESS" if successes else "FAILED"
    return {"status": status, "success": successes, "failed": len(failures), "errors": failures,
            "resolved_options": options}
