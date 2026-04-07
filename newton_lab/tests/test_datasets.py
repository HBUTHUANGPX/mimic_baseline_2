from __future__ import annotations

from pathlib import Path

import numpy as np

from newton_lab.datasets import DatasetSpec, LocalMotionDataset


def _write_motion_npz(path: Path, *, frames: int = 4) -> None:
    robot_joint_pos = np.zeros((frames, 6), dtype=np.float32)
    robot_body_pos = np.zeros((frames, 3, 3), dtype=np.float32)
    robot_body_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    robot_body_quat[..., 3] = 1.0
    human_local_transforms = np.zeros((frames, 5, 7), dtype=np.float32)
    human_local_transforms[..., 3] = 1.0
    human_global_pos = np.zeros((frames, 5, 3), dtype=np.float32)
    human_global_quat = np.zeros((frames, 5, 4), dtype=np.float32)
    human_global_quat[..., 3] = 1.0
    np.savez(
        path,
        fps=np.array(50, dtype=np.int32),
        robot_joint_names=np.array([f"joint_{idx}" for idx in range(6)]),
        robot_body_names=np.array([f"body_{idx}" for idx in range(3)]),
        human_joint_names=np.array([f"human_{idx}" for idx in range(5)]),
        robot_joint_pos=robot_joint_pos,
        robot_body_pos=robot_body_pos,
        robot_body_quat=robot_body_quat,
        human_local_transforms=human_local_transforms,
        human_global_pos=human_global_pos,
        human_global_quat=human_global_quat,
    )


def test_local_dataset_manifest_resolves_motion_groups(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a)
    _write_motion_npz(motion_b, frames=6)
    manifest = tmp_path / "motions.yaml"
    manifest.write_text(
        """
name: unit-test-motions
groups:
  warmup:
    paths:
      - motion_a.npz
    weight: 1.0
  main:
    paths:
      - motion_b.npz
    weight: 2.5
""".strip()
    )

    dataset = LocalMotionDataset(DatasetSpec(manifest_path=manifest))

    assert dataset.spec.name == "unit-test-motions"
    assert dataset.motion_groups == {
        "warmup": [str(motion_a)],
        "main": [str(motion_b)],
    }
    assert dataset.group_weights == {"warmup": 1.0, "main": 2.5}
