from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import newton
import torch

from newton_lab.robots import G1RobotCfg


def _load_model(builder: newton.ModelBuilder, asset_path: Path) -> None:
    suffix = asset_path.suffix.lower()
    if suffix == ".xml":
        builder.add_mjcf(str(asset_path))
        return
    if suffix == ".urdf":
        builder.add_urdf(str(asset_path))
        return
    builder.add_usd(str(asset_path))


def _label_leaf(label: object) -> str:
    return str(label).split("/")[-1]


@dataclass
class NewtonSimCfg:
    num_envs: int = 1
    device: str = "cpu"
    dt: float = 0.02


class NewtonSim:
    def __init__(self, cfg: NewtonSimCfg, robot_cfg: G1RobotCfg) -> None:
        self.cfg = cfg
        self.robot_cfg = robot_cfg
        self.model = None
        self.state = None
        self.metadata: dict[str, object] = {}
        self.device = torch.device(cfg.device)
        self.joint_pos: torch.Tensor | None = None
        self.joint_vel: torch.Tensor | None = None
        self.body_pos: torch.Tensor | None = None
        self.body_quat: torch.Tensor | None = None
        self.body_lin_vel: torch.Tensor | None = None
        self.body_ang_vel: torch.Tensor | None = None
        self.default_joint_pos: torch.Tensor | None = None
        self.base_body_offsets: torch.Tensor | None = None
        self.action_dim = 0
        self.root_coord_dim = 7
        self.root_dof_dim = 6

    def build(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        _load_model(builder, self.robot_cfg.asset_path)
        self.model = builder.finalize(device=str(self.device))
        self.state = self.model.state(requires_grad=False)

        body_count = int(self.model.body_count)
        joint_coord_count = int(self.model.joint_coord_count)
        joint_dof_count = int(self.model.joint_dof_count)
        self.action_dim = joint_coord_count - self.root_coord_dim
        self.metadata = {
            "body_count": body_count,
            "joint_coord_count": joint_coord_count,
            "joint_dof_count": joint_dof_count,
            "body_names": [_label_leaf(label) for label in self.model.body_label],
            "joint_names": [_label_leaf(label) for label in self.model.joint_label],
        }

        self.default_joint_pos = torch.zeros((self.cfg.num_envs, joint_coord_count), dtype=torch.float32, device=self.device)
        self.default_joint_pos[:, 2] = self.robot_cfg.default_root_height
        self.default_joint_pos[:, 6] = 1.0
        self.joint_pos = self.default_joint_pos.clone()
        self.joint_vel = torch.zeros((self.cfg.num_envs, joint_dof_count), dtype=torch.float32, device=self.device)

        offsets = list(self.robot_cfg.body_pos_offsets)
        if len(offsets) < body_count:
            offsets.extend([(0.0, 0.0, 0.02 * idx) for idx in range(len(offsets), body_count)])
        self.base_body_offsets = torch.tensor(offsets[:body_count], dtype=torch.float32, device=self.device)
        self.body_pos = torch.zeros((self.cfg.num_envs, body_count, 3), dtype=torch.float32, device=self.device)
        self.body_quat = torch.zeros((self.cfg.num_envs, body_count, 4), dtype=torch.float32, device=self.device)
        self.body_quat[..., 3] = 1.0
        self.body_lin_vel = torch.zeros((self.cfg.num_envs, body_count, 3), dtype=torch.float32, device=self.device)
        self.body_ang_vel = torch.zeros((self.cfg.num_envs, body_count, 3), dtype=torch.float32, device=self.device)
        self._refresh_body_state()

    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        assert self.default_joint_pos is not None
        env_ids = self._normalize_env_ids(env_ids)
        self.joint_pos[env_ids] = self.default_joint_pos[env_ids]
        self.joint_vel[env_ids].zero_()
        self.body_lin_vel[env_ids].zero_()
        self.body_ang_vel[env_ids].zero_()
        self._refresh_body_state(env_ids)

    def write_joint_state(
        self,
        env_ids: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        if joint_pos is not None:
            self.joint_pos[env_ids] = joint_pos.to(device=self.device, dtype=torch.float32)
        if joint_vel is not None:
            self.joint_vel[env_ids] = joint_vel.to(device=self.device, dtype=torch.float32)
        self._refresh_body_state(env_ids)

    def step(self, action_targets: torch.Tensor | None = None) -> None:
        if action_targets is not None:
            targets = action_targets.to(device=self.device, dtype=torch.float32)
            current = self.joint_pos[:, self.root_coord_dim :]
            next_pos = current + targets
            self.joint_vel[:, self.root_dof_dim :] = (next_pos - current) / max(self.cfg.dt, 1e-6)
            self.joint_pos[:, self.root_coord_dim :] = next_pos
            self.joint_vel[:, : self.root_dof_dim] *= 0.95
        self._refresh_body_state()

    def get_joint_positions(self) -> torch.Tensor:
        return self.joint_pos.clone()

    def get_joint_velocities(self) -> torch.Tensor:
        return self.joint_vel.clone()

    def get_body_positions(self) -> torch.Tensor:
        return self.body_pos.clone()

    def get_body_quaternions(self) -> torch.Tensor:
        return self.body_quat.clone()

    def get_root_linear_velocity(self) -> torch.Tensor:
        return self.body_lin_vel[:, 0].clone()

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.cfg.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _refresh_body_state(self, env_ids: torch.Tensor | None = None) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        root_pos = self.joint_pos[env_ids, :3]
        self.body_pos[env_ids] = root_pos.unsqueeze(1) + self.base_body_offsets.unsqueeze(0)
        self.body_quat[env_ids].zero_()
        self.body_quat[env_ids, :, 3] = 1.0
        mean_joint_vel = self.joint_vel[env_ids, self.root_dof_dim :].mean(dim=1, keepdim=True)
        lin_vel = torch.zeros((env_ids.shape[0], 3), dtype=torch.float32, device=self.device)
        lin_vel[:, 0] = mean_joint_vel.squeeze(-1) * 0.05
        self.body_lin_vel[env_ids] = lin_vel.unsqueeze(1).repeat(1, self.body_pos.shape[1], 1)
        self.body_ang_vel[env_ids].zero_()
