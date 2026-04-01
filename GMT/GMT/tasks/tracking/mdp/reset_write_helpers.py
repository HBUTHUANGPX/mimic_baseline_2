from __future__ import annotations

from collections.abc import Sequence

import torch


def build_env_mask(
    *, num_envs: int, env_ids: Sequence[int] | torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Build a dense environment mask from selected environment ids."""
    env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_mask[env_ids_tensor] = True
    return env_mask


def write_reset_state_to_sim_mask(
    robot,
    *,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor,
) -> None:
    """Write reset state using full buffers and an environment mask."""
    env_mask = build_env_mask(
        num_envs=joint_pos.shape[0],
        env_ids=env_ids,
        device=joint_pos.device,
    )
    robot.write_joint_state_to_sim_mask(
        position=joint_pos,
        velocity=joint_vel,
        env_mask=env_mask,
    )
    robot.write_root_pose_to_sim_mask(
        root_pose=torch.cat([root_pos, root_ori], dim=-1),
        env_mask=env_mask,
    )
    robot.write_root_velocity_to_sim_mask(
        root_velocity=torch.cat([root_lin_vel, root_ang_vel], dim=-1),
        env_mask=env_mask,
    )
