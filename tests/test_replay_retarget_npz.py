from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_retarget_npz import (
    build_replay_motion,
    build_root_state,
    prepare_joint_state_tensors,
    prepare_robot_cfg,
)


def test_build_replay_motion_uses_loader_root_body_state():
    loader = SimpleNamespace(
        fps=50,
        motion_lengths=[2],
        robot_joint_names=["j0", "j1"],
        robot_joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, 0.5], [9.0, 9.0, 9.0]],
                [[1.0, 0.0, 0.6], [8.0, 8.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
        body_quat_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.7, 0.1, 0.2, 0.6], [0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )

    motion = build_replay_motion(loader)

    assert motion.fps == 50.0
    assert motion.num_frames == 2
    assert motion.robot_joint_names == ["j0", "j1"]
    np.testing.assert_allclose(
        motion.robot_root_pos,
        np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.6]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        motion.robot_root_quat,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.2, 0.6]], dtype=np.float32),
    )


def test_build_replay_motion_rejects_multi_motion_loader():
    loader = SimpleNamespace(
        fps=50,
        motion_lengths=[2, 3],
        robot_joint_names=["j0"],
        robot_joint_pos=torch.zeros((5, 1), dtype=torch.float32),
        body_pos_w=torch.zeros((5, 1, 3), dtype=torch.float32),
        body_quat_w=torch.zeros((5, 1, 4), dtype=torch.float32),
    )

    try:
        build_replay_motion(loader)
    except ValueError as exc:
        assert "exactly one motion file" in str(exc)
    else:
        raise AssertionError("build_replay_motion should reject multi-motion loaders")


def test_build_root_state_packs_pose_and_zero_velocity():
    root_pos = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    root_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    env_origins = np.array([[2.0, 3.0, 0.0]], dtype=np.float32)

    root_state = build_root_state(root_pos, root_quat, env_origins)

    np.testing.assert_allclose(root_state[0, :3], np.array([2.0, 3.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(root_state[0, 3:7], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(root_state[0, 7:], np.zeros(6, dtype=np.float32))


def test_prepare_robot_cfg_disables_contact_sensors_without_mutating_input():
    class SpawnStub:
        def __init__(self, active, usd_path="assets/demo.usd"):
            self.activate_contact_sensors = active
            self.usd_path = usd_path

    class CfgStub:
        def __init__(self, active, usd_path="assets/demo.usd"):
            self.spawn = SpawnStub(active, usd_path)

        def copy(self):
            return CfgStub(self.spawn.activate_contact_sensors, self.spawn.usd_path)

    original = CfgStub(True)
    prepared = prepare_robot_cfg(original)

    assert original.spawn.activate_contact_sensors is True
    assert prepared.spawn.activate_contact_sensors is False
    assert prepared.spawn.usd_path.endswith("/assets/demo.usd")


def test_prepare_joint_state_tensors_accepts_numpy_defaults():
    import torch

    default_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    default_vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    frame_joint_pos = torch.tensor([0.1, 0.2], dtype=torch.float32)
    joint_indices = [1, 2]

    joint_pos, joint_vel = prepare_joint_state_tensors(
        default_joint_pos=default_pos,
        default_joint_vel=default_vel,
        frame_joint_pos=frame_joint_pos,
        joint_indices=joint_indices,
        num_envs=1,
        device="cpu",
    )

    assert isinstance(joint_pos, torch.Tensor)
    assert isinstance(joint_vel, torch.Tensor)
    np.testing.assert_allclose(joint_pos.numpy(), np.array([[1.0, 0.1, 0.2]], dtype=np.float32))
    np.testing.assert_allclose(joint_vel.numpy(), np.zeros((1, 3), dtype=np.float32))
