from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class G1RobotCfg:
    asset_path: Path = Path("assets/unitree_g1/g1_29dof_rev_1_0.xml")
    default_root_height: float = 0.78
    action_scale: float = 0.25
    default_joint_pos: tuple[float, ...] = ()
    tracked_body_names: tuple[str, ...] = (
        "pelvis",
        "left_knee_link",
        "right_knee_link",
        "torso_link",
        "left_elbow_link",
        "right_elbow_link",
    )
    body_pos_offsets: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: tuple((0.0, 0.0, 0.02 * idx) for idx in range(30))
    )
