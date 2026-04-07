from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    diff = (values[1:] - values[:-1]) / dt
    return torch.cat((diff[:1], diff), dim=0)


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., :3] *= -1.0
    return result


def _normalize_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_to_angular_velocity(quat_xyzw: torch.Tensor, dt: float) -> torch.Tensor:
    if quat_xyzw.shape[0] <= 1:
        return torch.zeros((*quat_xyzw.shape[:-1], 3), dtype=quat_xyzw.dtype, device=quat_xyzw.device)
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
    rel = _quat_mul_xyzw(quat_xyzw[1:], _quat_conjugate_xyzw(quat_xyzw[:-1]))
    rel = _normalize_quat_xyzw(rel)
    xyz = rel[..., :3]
    w = rel[..., 3].clamp(-1.0, 1.0)
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.unsqueeze(-1))
    axis = xyz / sin_half.clamp_min(1e-8)
    angvel = axis * (angle / dt)
    angvel = torch.where(sin_half > 1e-8, angvel, torch.zeros_like(angvel))
    return torch.cat((angvel[:1], angvel), dim=0)


@dataclass
class DatasetSpec:
    manifest_path: Path | None = None
    name: str | None = None
    split: str | None = None


@dataclass
class MotionClip:
    path: str
    fps: int
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_joint_names: list[str]
    robot_joint_pos: torch.Tensor
    robot_joint_vel: torch.Tensor
    robot_body_pos: torch.Tensor
    robot_body_vel: torch.Tensor
    robot_body_quat: torch.Tensor
    robot_body_angvel: torch.Tensor
    human_local_transforms: torch.Tensor
    human_global_pos: torch.Tensor
    human_global_vel: torch.Tensor
    human_global_quat: torch.Tensor
    human_global_angvel: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.robot_joint_pos.shape[0])


@dataclass
class LocalMotionDataset:
    spec: DatasetSpec
    motion_groups: dict[str, list[str]] = field(init=False, default_factory=dict)
    group_weights: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.spec.manifest_path is None:
            self.spec.name = self.spec.name or "empty"
            return
        manifest_path = Path(self.spec.manifest_path)
        data = yaml.safe_load(manifest_path.read_text()) or {}
        self.spec.name = self.spec.name or data.get("name") or manifest_path.stem
        groups = data.get("groups", {})
        for group_name, group_cfg in groups.items():
            rel_paths = group_cfg.get("paths", [])
            self.motion_groups[group_name] = [str((manifest_path.parent / rel_path).resolve()) for rel_path in rel_paths]
            self.group_weights[group_name] = float(group_cfg.get("weight", 1.0))

    def list_motion_paths(self) -> list[str]:
        return [path for paths in self.motion_groups.values() for path in paths]

    def load_motion_clips(self, device: str = "cpu") -> list[MotionClip]:
        clips: list[MotionClip] = []
        for motion_path in self.list_motion_paths():
            raw = np.load(motion_path, allow_pickle=False)
            fps = int(raw["fps"])
            dt = 1.0 / max(fps, 1)
            robot_joint_pos = torch.tensor(raw["robot_joint_pos"], dtype=torch.float32, device=device)
            robot_body_pos = torch.tensor(raw["robot_body_pos"], dtype=torch.float32, device=device)
            robot_body_quat = torch.tensor(raw["robot_body_quat"], dtype=torch.float32, device=device)
            human_local_transforms = torch.tensor(raw["human_local_transforms"], dtype=torch.float32, device=device)
            human_global_pos = torch.tensor(raw["human_global_pos"], dtype=torch.float32, device=device)
            human_global_quat = torch.tensor(raw["human_global_quat"], dtype=torch.float32, device=device)
            clips.append(
                MotionClip(
                    path=motion_path,
                    fps=fps,
                    robot_joint_names=[str(name) for name in raw["robot_joint_names"].tolist()],
                    robot_body_names=[str(name) for name in raw["robot_body_names"].tolist()],
                    human_joint_names=[str(name) for name in raw["human_joint_names"].tolist()],
                    robot_joint_pos=robot_joint_pos,
                    robot_joint_vel=_finite_difference(robot_joint_pos, dt),
                    robot_body_pos=robot_body_pos,
                    robot_body_vel=_finite_difference(robot_body_pos, dt),
                    robot_body_quat=robot_body_quat,
                    robot_body_angvel=_quat_to_angular_velocity(robot_body_quat, dt),
                    human_local_transforms=human_local_transforms,
                    human_global_pos=human_global_pos,
                    human_global_vel=_finite_difference(human_global_pos, dt),
                    human_global_quat=human_global_quat,
                    human_global_angvel=_quat_to_angular_velocity(human_global_quat, dt),
                )
            )
        return clips


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
