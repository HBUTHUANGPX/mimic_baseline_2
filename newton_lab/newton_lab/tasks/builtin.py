from __future__ import annotations

from dataclasses import replace

from newton_lab.datasets import DatasetSpec
from newton_lab.envs import ManagerBasedEnvCfg, NewtonSceneCfg
from newton_lab.managers import (
    ActionManagerCfg,
    CommandManagerCfg,
    ObservationManagerCfg,
    RewardManagerCfg,
    TerminationManagerCfg,
)
from newton_lab.rl import RslRlAdapterCfg
from newton_lab.tasks.registry import TaskSpec, register_task


def register_builtin_tasks() -> None:
    velocity_cfg = ManagerBasedEnvCfg(
        scene=NewtonSceneCfg(num_envs=4),
        observations=ObservationManagerCfg(include_reference=False),
        actions=ActionManagerCfg(scale=0.25),
        commands=CommandManagerCfg(mode="velocity"),
        rewards=RewardManagerCfg(mode="velocity"),
        terminations=TerminationManagerCfg(max_episode_length=64),
    )
    velocity_play = replace(velocity_cfg, headless=False, scene=replace(velocity_cfg.scene, num_envs=1))
    register_task(
        TaskSpec(
            name="newton_lab.g1.velocity",
            train_env_cfg=velocity_cfg,
            play_env_cfg=velocity_play,
            agent_cfg=RslRlAdapterCfg(experiment_name="g1_velocity"),
        )
    )

    tracking_cfg = ManagerBasedEnvCfg(
        scene=NewtonSceneCfg(num_envs=4),
        observations=ObservationManagerCfg(include_reference=True),
        actions=ActionManagerCfg(scale=0.25),
        commands=CommandManagerCfg(mode="tracking"),
        rewards=RewardManagerCfg(mode="tracking"),
        terminations=TerminationManagerCfg(max_episode_length=64),
        dataset=DatasetSpec(),
    )
    tracking_play = replace(tracking_cfg, headless=False, scene=replace(tracking_cfg.scene, num_envs=1))
    register_task(
        TaskSpec(
            name="newton_lab.g1.tracking",
            train_env_cfg=tracking_cfg,
            play_env_cfg=tracking_play,
            agent_cfg=RslRlAdapterCfg(experiment_name="g1_tracking"),
        )
    )
