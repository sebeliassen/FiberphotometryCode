"""Historical pipeline defaults; values preserved for existing callers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineHyperparameters:
    num_ar_iters: int = 50
    num_slds_iters: int = 200
    total_iters: int = 250
    save_every_n_iters: int = 25
    init_kappa: float = 7.5e13
    decrease_kappa_factor: float = 10.0
    slds_kappa: float = 7.5e12
    target_fps: int = 60
    conf_threshold: float = 0.5
    min_frames: int = 100

__all__ = ["PipelineHyperparameters"]
