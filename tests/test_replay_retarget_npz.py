from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_retarget_npz import (
    build_root_state,
    load_retarget_motion_npz,
    prepare_joint_state_tensors,
    prepare_robot_cfg,
)


def test_load_retarget_motion_npz_reads_new_schema(tmp_path: Path):
    npz_path = tmp_path / "motion.npz"
    np.savez(
        npz_path,
        fps=np.array(50, dtype=np.int32),
        num_frames=np.array(2, dtype=np.int32),
        robot_name=np.array("unitree_g1"),
        robot_joint_names=np.array(["j0", "j1"]),
        robot_root_pos=np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.6]], dtype=np.float32),
        robot_root_quat=np.array([[1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.2, 0.6]], dtype=np.float32),
        robot_joint_pos=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )

    motion = load_retarget_motion_npz(npz_path)

    assert motion.fps == 50.0
    assert motion.num_frames == 2
    assert motion.robot_name == "unitree_g1"
    assert motion.robot_joint_names == ["j0", "j1"]
    np.testing.assert_allclose(motion.robot_root_pos[1], np.array([1.0, 0.0, 0.6], dtype=np.float32))


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
