from __future__ import annotations

from pathlib import Path

import torch

from newton_lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, NewtonSceneCfg
from newton_lab.managers import (
    ActionManagerCfg,
    ObservationManagerCfg,
    RewardManagerCfg,
    TerminationManagerCfg,
)
from newton_lab.robots.g1 import G1RobotCfg


def test_manager_based_env_reset_and_step_return_finite_tensors() -> None:
    cfg = ManagerBasedEnvCfg(
        scene=NewtonSceneCfg(
            num_envs=3,
            robot=G1RobotCfg(asset_path=Path("assets/unitree_g1/g1_29dof_rev_1_0.xml")),
        ),
        observations=ObservationManagerCfg(),
        actions=ActionManagerCfg(scale=0.25),
        rewards=RewardManagerCfg(mode="velocity"),
        terminations=TerminationManagerCfg(max_episode_length=8),
    )
    env = ManagerBasedEnv(cfg)
    env.build()

    obs, extras = env.reset()
    assert obs["policy"].shape[0] == 3
    assert "manager_order" in extras

    action_dim = env.action_manager.action_dim
    actions = torch.zeros((3, action_dim), dtype=torch.float32)
    next_obs, rewards, dones, info = env.step(actions)

    assert next_obs["policy"].shape[0] == 3
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert torch.isfinite(next_obs["policy"]).all()
    assert torch.isfinite(rewards).all()
    assert info["manager_order"] == [
        "action.pre_step",
        "command.pre_step",
        "reward.post_step",
        "termination.post_step",
        "observation.export",
    ]
