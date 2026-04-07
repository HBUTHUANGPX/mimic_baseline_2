from __future__ import annotations

from dataclasses import dataclass, field

import torch

from newton_lab.datasets import DatasetSpec
from newton_lab.managers import (
    ActionManager,
    ActionManagerCfg,
    CommandManagerCfg,
    ObservationManager,
    ObservationManagerCfg,
    ReferenceMotionManager,
    RewardManager,
    RewardManagerCfg,
    TerminationManager,
    TerminationManagerCfg,
    TrackingCommandManager,
    VelocityCommandManager,
)
from newton_lab.robots import G1RobotCfg
from newton_lab.sim import NewtonSim, NewtonSimCfg


@dataclass
class NewtonSceneCfg:
    num_envs: int = 1
    robot: G1RobotCfg = field(default_factory=G1RobotCfg)
    terrain: str = "plane"


@dataclass
class ManagerBasedEnvCfg:
    scene: NewtonSceneCfg = field(default_factory=NewtonSceneCfg)
    sim: NewtonSimCfg = field(default_factory=NewtonSimCfg)
    observations: ObservationManagerCfg = field(default_factory=ObservationManagerCfg)
    actions: ActionManagerCfg = field(default_factory=ActionManagerCfg)
    commands: CommandManagerCfg = field(default_factory=CommandManagerCfg)
    rewards: RewardManagerCfg = field(default_factory=RewardManagerCfg)
    terminations: TerminationManagerCfg = field(default_factory=TerminationManagerCfg)
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    decimation: int = 1
    episode_length_s: float = 20.0
    headless: bool = True
    device: str = "cpu"


class ManagerBasedEnv:
    is_vector_env = True

    def __init__(self, cfg: ManagerBasedEnvCfg) -> None:
        self.cfg = cfg
        self.cfg.sim.num_envs = cfg.scene.num_envs
        self.cfg.sim.device = cfg.device
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.scene.num_envs
        self.manager_order: list[str] = []
        self.extras: dict[str, object] = {}
        self.obs_buf: dict[str, torch.Tensor] = {}
        self.sim = NewtonSim(cfg.sim, cfg.scene.robot)
        self.action_manager = ActionManager(cfg.actions)
        self.command_manager = None
        self.reward_manager = RewardManager(cfg.rewards)
        self.termination_manager = TerminationManager(cfg.terminations)
        self.observation_manager = ObservationManager(cfg.observations)
        self.reference_motion_manager = None
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def max_episode_length(self) -> int:
        return self.cfg.terminations.max_episode_length

    def build(self) -> None:
        self.sim.build()
        if self.cfg.commands.mode == "tracking":
            self.command_manager = TrackingCommandManager()
        else:
            self.command_manager = VelocityCommandManager()
        self.action_manager.build(self)
        self.command_manager.build(self)
        if self.cfg.commands.mode == "tracking" and self.cfg.dataset.manifest_path is not None:
            from newton_lab.datasets import LocalMotionDataset

            self.reference_motion_manager = ReferenceMotionManager(LocalMotionDataset(self.cfg.dataset))
            self.reference_motion_manager.build(self)
        self.reward_manager.build(self)
        self.termination_manager.build(self)
        self.observation_manager.build(self)

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        env_ids = self._normalize_env_ids(env_ids)
        self.sim.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.command_manager.reset(env_ids)
        if self.reference_motion_manager is not None:
            self.reference_motion_manager.reset(env_ids)
        self.manager_order = []
        obs = self.observation_manager.export()
        extras = self.get_extras()
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, object]]:
        self.manager_order = []
        self.action_manager.pre_step(actions)
        self.command_manager.pre_step(actions)
        if self.reference_motion_manager is not None:
            self.reference_motion_manager.post_step()
        self.reward_manager.post_step()
        self.termination_manager.post_step()
        self.episode_length_buf += 1
        obs = self.observation_manager.export()
        rewards = self.reward_manager.export()
        dones = self.termination_manager.export()
        if torch.any(dones):
            self.reset(torch.nonzero(dones, as_tuple=False).squeeze(-1))
        return obs, rewards, dones, self.get_extras()

    def get_observations(self) -> dict[str, torch.Tensor]:
        self.obs_buf = self.observation_manager.export()
        return self.obs_buf

    def get_extras(self) -> dict[str, object]:
        extras: dict[str, object] = {"manager_order": list(self.manager_order)}
        if self.reference_motion_manager is not None:
            extras["reference_motion"] = self.reference_motion_manager.export()
        self.extras = extras
        return extras

    def close(self) -> None:
        return None

    @property
    def unwrapped(self) -> "ManagerBasedEnv":
        return self

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)
