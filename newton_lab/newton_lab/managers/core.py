from __future__ import annotations

from dataclasses import dataclass, field

import torch

from newton_lab.datasets import LocalMotionDataset, MotionClip


@dataclass
class ActionManagerCfg:
    scale: float = 0.25


@dataclass
class ObservationManagerCfg:
    include_reference: bool = True


@dataclass
class CommandManagerCfg:
    mode: str = "velocity"


@dataclass
class RewardManagerCfg:
    mode: str = "velocity"


@dataclass
class TerminationManagerCfg:
    max_episode_length: int = 1000
    min_root_height: float = 0.2


@dataclass
class ResetManagerCfg:
    enabled: bool = True


class BaseManager:
    def build(self, env) -> None:
        self.env = env

    def reset(self, env_ids: torch.Tensor) -> None:
        return None

    def pre_step(self, actions: torch.Tensor) -> None:
        return None

    def post_step(self) -> None:
        return None

    def export(self):
        return {}


class ActionManager(BaseManager):
    def __init__(self, cfg: ActionManagerCfg) -> None:
        self.cfg = cfg
        self.action_dim = 0
        self.last_actions: torch.Tensor | None = None

    def build(self, env) -> None:
        super().build(env)
        self.action_dim = env.sim.action_dim
        self.last_actions = torch.zeros((env.num_envs, self.action_dim), dtype=torch.float32, device=env.device)

    def pre_step(self, actions: torch.Tensor) -> None:
        self.env.manager_order.append("action.pre_step")
        actions = actions.to(device=self.env.device, dtype=torch.float32)
        self.last_actions = actions
        self.env.sim.step(action_targets=actions * self.cfg.scale)


class VelocityCommandManager(BaseManager):
    def __init__(self) -> None:
        self.commands: torch.Tensor | None = None

    def build(self, env) -> None:
        super().build(env)
        self.commands = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
        self.commands[:, 0] = 0.4

    def reset(self, env_ids: torch.Tensor) -> None:
        self.commands[env_ids, 0] = 0.4
        self.commands[env_ids, 1:] = 0.0

    def pre_step(self, actions: torch.Tensor) -> None:
        self.env.manager_order.append("command.pre_step")

    def export(self) -> dict[str, torch.Tensor]:
        return {"command": self.commands}


class TrackingCommandManager(BaseManager):
    def pre_step(self, actions: torch.Tensor) -> None:
        self.env.manager_order.append("command.pre_step")


class ReferenceMotionManager(BaseManager):
    def __init__(self, dataset: LocalMotionDataset) -> None:
        self.dataset = dataset
        self.motion_clips: list[MotionClip] = []
        self.motion_ids: torch.Tensor | None = None
        self.frame_ids: torch.Tensor | None = None

    def build(self, env) -> None:
        super().build(env)
        self.motion_clips = self.dataset.load_motion_clips(device=str(env.device))
        self.motion_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.frame_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        if not self.motion_clips:
            return
        self.motion_ids[env_ids] = 0
        self.frame_ids[env_ids] = 0

    def post_step(self) -> None:
        if not self.motion_clips:
            return
        self.frame_ids += 1
        for env_id in range(self.env.num_envs):
            clip = self.motion_clips[int(self.motion_ids[env_id])]
            self.frame_ids[env_id] %= clip.num_frames

    def export(self) -> dict[str, torch.Tensor]:
        if not self.motion_clips:
            return {}
        joints = []
        anchors = []
        for env_id in range(self.env.num_envs):
            clip = self.motion_clips[int(self.motion_ids[env_id])]
            frame = int(self.frame_ids[env_id])
            joints.append(clip.robot_joint_pos[frame])
            anchors.append(clip.robot_body_pos[frame, 0])
        return {
            "reference_joint_pos": torch.stack(joints, dim=0),
            "reference_anchor_pos": torch.stack(anchors, dim=0),
        }


class RewardManager(BaseManager):
    def __init__(self, cfg: RewardManagerCfg) -> None:
        self.cfg = cfg
        self.reward = None

    def build(self, env) -> None:
        super().build(env)
        self.reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def post_step(self) -> None:
        self.env.manager_order.append("reward.post_step")
        if self.cfg.mode == "tracking" and self.env.reference_motion_manager is not None:
            ref = self.env.reference_motion_manager.export()["reference_joint_pos"]
            current = self.env.sim.get_joint_positions()[:, self.env.sim.root_coord_dim :]
            dims = min(current.shape[1], ref.shape[1])
            error = (current[:, :dims] - ref[:, :dims]).pow(2).mean(dim=1)
            self.reward = torch.exp(-error)
            return
        actions = self.env.action_manager.last_actions
        command = self.env.command_manager.export().get("command")
        if command is None:
            self.reward = 1.0 - actions.pow(2).mean(dim=1)
            return
        speed = command[:, 0]
        act_mag = actions.abs().mean(dim=1)
        self.reward = 1.0 - (speed - act_mag).abs()

    def export(self) -> torch.Tensor:
        return self.reward


class TerminationManager(BaseManager):
    def __init__(self, cfg: TerminationManagerCfg) -> None:
        self.cfg = cfg
        self.done = None

    def build(self, env) -> None:
        super().build(env)
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def post_step(self) -> None:
        self.env.manager_order.append("termination.post_step")
        timeout = self.env.episode_length_buf >= self.cfg.max_episode_length
        fallen = self.env.sim.get_joint_positions()[:, 2] < self.cfg.min_root_height
        self.done = timeout | fallen

    def export(self) -> torch.Tensor:
        return self.done


class ObservationManager(BaseManager):
    def __init__(self, cfg: ObservationManagerCfg) -> None:
        self.cfg = cfg

    def export(self) -> dict[str, torch.Tensor]:
        self.env.manager_order.append("observation.export")
        joint_pos = self.env.sim.get_joint_positions()[:, self.env.sim.root_coord_dim :]
        joint_vel = self.env.sim.get_joint_velocities()[:, self.env.sim.root_dof_dim :]
        action = self.env.action_manager.last_actions
        policy_terms = [joint_pos, joint_vel, action]
        critic_terms = [joint_pos, joint_vel, action]
        command = self.env.command_manager.export().get("command")
        if command is not None:
            policy_terms.append(command)
            critic_terms.append(command)
        if self.cfg.include_reference and self.env.reference_motion_manager is not None:
            reference = self.env.reference_motion_manager.export()
            if reference:
                policy_terms.append(reference["reference_joint_pos"])
                critic_terms.append(reference["reference_joint_pos"])
        return {
            "policy": torch.cat(policy_terms, dim=1),
            "critic": torch.cat(critic_terms, dim=1),
        }
