from __future__ import annotations

from dataclasses import asdict

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from newton_lab.envs import ManagerBasedEnv
from newton_lab.rl.config import RslRlAdapterCfg


class RslRlAdapter(VecEnv):
    def __init__(self, env: ManagerBasedEnv, cfg: RslRlAdapterCfg) -> None:
        self.env = env
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_actions = env.action_manager.action_dim
        self.max_episode_length = env.max_episode_length
        self.env.reset()

    @property
    def unwrapped(self) -> ManagerBasedEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return TensorDict(self.env.get_observations(), batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        return TensorDict(obs, batch_size=[self.num_envs]), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        return TensorDict(obs, batch_size=[self.num_envs]), rewards, dones.to(torch.long), extras

    def close(self) -> None:
        self.env.close()

    def asdict(self) -> dict:
        return asdict(self.cfg)
