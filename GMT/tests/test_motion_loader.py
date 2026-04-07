from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_LOADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "GMT"
    / "tasks"
    / "tracking"
    / "mdp"
    / "motion_loader.py"
)
_SPEC = importlib.util.spec_from_file_location("tracking_motion_loader", _LOADER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MotionLoader = _MODULE.MotionLoader

_MOTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "soma-retargeter"
    / "assets"
    / "motions"
    / "test-export"
)


def test_motion_loader_loads_joint_human_and_robot_timelines() -> None:
    motion_files = {
        "neutral": [
            str(_MOTION_DIR / "Neutral_throw_ball_001__A057.npz"),
            str(_MOTION_DIR / "Neutral_walk_forward_002__A057.npz"),
        ]
    }

    loader = MotionLoader(
        motion_files,
        body_indexes=[0, 1, 2],
        history_frames=2,
        future_frames=2,
        device="cpu",
    )

    assert loader.num_motions == 2
    assert loader.fps == 50
    assert loader.robot_joint_names is not None
    assert loader.robot_body_names is not None
    assert loader.human_joint_names is not None
    assert len(loader.robot_joint_names) == loader.robot_joint_pos.shape[1]
    assert len(loader.robot_body_names) == loader._robot_body_pos.shape[1]
    assert len(loader.human_joint_names) == loader.human_joint_pos.shape[1]

    assert loader.robot_joint_pos.shape[0] == loader.human_joint_pos.shape[0]
    assert loader.human_local_transforms.shape[0] == loader.human_joint_pos.shape[0]
    assert loader.human_local_transforms.shape[-1] == 7
    assert loader.robot_joint_vel.shape == loader.robot_joint_pos.shape
    assert loader.robot_body_vel.shape == loader.robot_body_pos.shape
    assert loader.robot_body_angvel.shape[-1] == 3
    assert loader.human_joint_pos.shape[-1] == 4
    assert loader.human_joint_vel.shape[-1] == 3
    assert loader.human_body_pos.shape == loader.human_body_vel.shape
    assert loader.human_body_quat.shape[-1] == 4
    assert loader.human_body_angvel.shape[-1] == 3

    expected_total = sum(loader.motion_lengths)
    assert loader.time_step_total == expected_total
    assert loader.motion_indices.shape == (2, 2)
    assert torch.equal(loader.motion_indices[:, 0], torch.tensor([0, loader.motion_lengths[0]]))
    assert torch.equal(
        loader.motion_indices[:, 1],
        torch.tensor([loader.motion_lengths[0], expected_total]),
    )
    assert loader.valid_center_indices.numel() > 0
    assert loader.new_data_flag[loader.motion_lengths[0]]


def test_motion_loader_uses_body_indexes_only_for_robot_body_selection() -> None:
    motion_files = {"single": str(_MOTION_DIR / "Neutral_throw_ball_001__A057.npz")}

    loader = MotionLoader(
        motion_files,
        body_indexes=[0, 5, 10],
        history_frames=1,
        future_frames=1,
        device="cpu",
    )

    assert loader.robot_body_pos.shape[1] == 3
    assert loader._robot_body_pos.shape[1] == len(loader.robot_body_names)
    assert loader.human_body_pos.shape[1] == len(loader.human_joint_names)
