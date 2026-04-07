from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from pxr import Gf, Vt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_retarget_npz import (
    _build_soma_animation_arrays,
    _frame_root_pose,
    _requires_articulation_robot,
    _resolve_soma_reference_prim_path,
    _resolve_soma_usd_path,
    _validate_single_motion_loader,
    build_root_state,
    prepare_joint_state_tensors,
    prepare_robot_cfg,
)


def test_frame_root_pose_uses_loader_root_body_state():
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

    root_pos, root_quat = _frame_root_pose(loader, 1)

    assert isinstance(root_pos, torch.Tensor)
    assert isinstance(root_quat, torch.Tensor)
    torch.testing.assert_close(root_pos, torch.tensor([1.0, 0.0, 0.6], dtype=torch.float32))
    torch.testing.assert_close(root_quat, torch.tensor([0.7, 0.1, 0.2, 0.6], dtype=torch.float32))


def test_validate_single_motion_loader_rejects_multi_motion_loader():
    loader = SimpleNamespace(
        fps=50,
        motion_lengths=[2, 3],
        robot_joint_names=["j0"],
        robot_joint_pos=torch.zeros((5, 1), dtype=torch.float32),
        body_pos_w=torch.zeros((5, 1, 3), dtype=torch.float32),
        body_quat_w=torch.zeros((5, 1, 4), dtype=torch.float32),
    )

    try:
        _validate_single_motion_loader(loader)
    except ValueError as exc:
        assert "exactly one motion file" in str(exc)
    else:
        raise AssertionError("_validate_single_motion_loader should reject multi-motion loaders")


def test_requires_articulation_robot_excludes_soma():
    assert _requires_articulation_robot("g1") is True
    assert _requires_articulation_robot("Q1") is True
    assert _requires_articulation_robot("soma") is False


def test_resolve_soma_usd_path_points_to_existing_skeleton_asset():
    usd_path = _resolve_soma_usd_path()

    assert usd_path.name == "soma_base_skel_minimal.usd"
    assert usd_path.is_file()


def test_resolve_soma_reference_prim_path_targets_output_root():
    assert _resolve_soma_reference_prim_path() == "/OUTPUT"


def test_build_soma_animation_arrays_preserves_xyz_and_xyzw_order():
    frame_local_transforms = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9],
            [4.0, 5.0, 6.0, -0.1, -0.2, -0.3, 0.8],
        ],
        dtype=torch.float32,
    )

    translations, rotations = _build_soma_animation_arrays(frame_local_transforms)

    assert isinstance(translations, Vt.Vec3fArray)
    assert isinstance(rotations, Vt.QuatfArray)
    assert translations[0] == Gf.Vec3f(1.0, 2.0, 3.0)
    assert translations[1] == Gf.Vec3f(4.0, 5.0, 6.0)
    assert rotations[0] == Gf.Quatf(0.9, 0.1, 0.2, 0.3)
    assert rotations[1] == Gf.Quatf(0.8, -0.1, -0.2, -0.3)


def test_build_root_state_packs_pose_and_zero_velocity():
    root_pos = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    env_origins = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float32)

    root_state = build_root_state(root_pos, root_quat, env_origins)

    assert isinstance(root_state, torch.Tensor)
    torch.testing.assert_close(
        root_state[0, :3], torch.tensor([2.0, 3.0, 0.5], dtype=torch.float32)
    )
    torch.testing.assert_close(
        root_state[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    )
    torch.testing.assert_close(root_state[0, 7:], torch.zeros(6, dtype=torch.float32))


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


def test_prepare_joint_state_tensors_uses_tensor_inputs():
    default_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    default_vel = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
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
    torch.testing.assert_close(
        joint_pos, torch.tensor([[1.0, 0.1, 0.2]], dtype=torch.float32)
    )
    torch.testing.assert_close(joint_vel, torch.zeros((1, 3), dtype=torch.float32))


def test_prepare_joint_state_tensors_does_not_rewrap_tensor_frame_joint_pos(monkeypatch):
    default_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    default_vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    frame_joint_pos = torch.tensor([0.1, 0.2], dtype=torch.float32)
    joint_indices = [1, 2]
    original_as_tensor = torch.as_tensor

    def guarded_as_tensor(value, *args, **kwargs):
        if value is frame_joint_pos:
            raise AssertionError("frame_joint_pos should not be passed through torch.as_tensor")
        return original_as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", guarded_as_tensor)

    joint_pos, joint_vel = prepare_joint_state_tensors(
        default_joint_pos=default_pos,
        default_joint_vel=default_vel,
        frame_joint_pos=frame_joint_pos,
        joint_indices=joint_indices,
        num_envs=1,
        device="cpu",
    )

    torch.testing.assert_close(
        joint_pos, torch.tensor([[1.0, 0.1, 0.2]], dtype=torch.float32)
    )
    torch.testing.assert_close(joint_vel, torch.zeros((1, 3), dtype=torch.float32))
