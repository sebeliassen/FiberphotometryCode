"""Historical cohort defaults for the V2 compatibility API. New workflows receive explicit inputs."""

from __future__ import annotations

from pathlib import Path

from fpcode.common.paths import REPO_ROOT, resolve_data_path
from fpcode.compat.multiview.profile import RESOURCE_DIR

EXP_DIR = REPO_ROOT / "studies/multiview_experiments"


SCRIPT_DIR = EXP_DIR / "04_reports_and_figures"


COHORT_DIR = resolve_data_path("data/models/cohort_static_21kp_v2")


MODEL_NAME = "model_21kp_static_50ar_300slds_seed101"


MODEL_DIR = COHORT_DIR / MODEL_NAME


RESULTS_PATH = MODEL_DIR / "results.h5"


CONFIG_PATH = COHORT_DIR / "config.yml"


VFI_PATH = COHORT_DIR / "video_frame_indexes.npz"


PREPROCESSED_PATH = COHORT_DIR / "preprocessed_data.pkl"


FUSED_DIR = resolve_data_path("data/fused/multiview/cohort_fused_static_21kp_v2")


TOP_VID_DIR = resolve_data_path("data/raw/karen_pff_vs_monomer/video_files")


BOT_VID_DIR = resolve_data_path("data/raw/karen_pff_vs_monomer/botcam_video_files")


VIZ_VERSION = "v4"


DEFAULT_VIZ_DIR = resolve_data_path(
    f"data/analysis_cache/multiview_visualizations/visualizations_cohort_static_21kp_v2_seed101_{VIZ_VERSION}"
)


DORSAL_BODYPARTS = [
    "nose",
    "left_ear_tip",
    "right_ear_tip",
    "neck",
    "mid_back",
    "mouse_center",
    "mid_backend",
    "mid_backend2",
    "mid_backend3",
    "tail_base",
    "left_shoulder",
    "left_midside",
    "left_hip",
    "right_shoulder",
    "right_midside",
    "right_hip",
    "head_midpoint",
]


PAW_BODYPARTS = ["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"]


BODYPARTS_21KP = DORSAL_BODYPARTS + PAW_BODYPARTS


DORSAL_IDXS = list(range(len(DORSAL_BODYPARTS)))


PAW_IDXS = list(range(len(DORSAL_BODYPARTS), len(BODYPARTS_21KP)))


FRONT_PAW_BODYPARTS = ["left_front_paw", "right_front_paw"]


HIND_PAW_BODYPARTS = ["left_hind_paw", "right_hind_paw"]


FRONT_PAW_IDXS = [BODYPARTS_21KP.index(bp) for bp in FRONT_PAW_BODYPARTS]


HIND_PAW_IDXS = [BODYPARTS_21KP.index(bp) for bp in HIND_PAW_BODYPARTS]


BOT_NOSE = "nose_tip"


BOT_TAIL = "tail_base"


BOT_PAW_NAMES = ("left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw")


CHAMBER_TO_ARENA = {
    "top_left": "Arena1",
    "bottom_left": "Arena2",
    "bottom_right": "Arena3",
    "top_right": "Arena4",
}


SCHEMA_VERSION = 4


DENSITY_PCA_DIM = 4


DENSITY_N_NEIGHBORS = 50


DENSITY_PCA_SAMPLES = 50000


KPMS_FILTER_SIZE = 9  # keypoint_moseq.util.get_instance_trajectories default


DEFAULT_SCORE_PRE_SECONDS = 1.0 / 6.0


DEFAULT_SCORE_POST_SECONDS = 0.5


DEFAULT_MOVIE_PRE_SECONDS = 1.0


DEFAULT_MOVIE_POST_SECONDS = 2.0


DEFAULT_MIN_BOUT_SECONDS = 5.0 / 60.0  # the previous 5-frame minimum at 60 fps


_TRACKING_DIR = Path(__file__).resolve().parents[2] / "tracking"
CALIBRATION_FILES = [
    _TRACKING_DIR / name for name in (
        "camera_correspondence.py", "pair_matching.py", "clock_sync.py",
        "static_fusion.py", "geometry.py", "quality.py",
    )
] + [
    RESOURCE_DIR / name for name in (
        "epoch_arena_transforms.json", "static_arena_transforms.json", "ttl_cohort_offsets.json",
    )
] + [RESOURCE_DIR.parent / "profile.py"]
