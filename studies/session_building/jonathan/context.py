from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import JCFG, JonathanConfig


@dataclass
class BuildContext:
    root: Path
    h5_dir: Path
    video_dir: Optional[Path] = None
    csv_root: Optional[Path] = None
    project: str = "jonathan_full"
    cfg: JonathanConfig = field(default_factory=lambda: JCFG)
