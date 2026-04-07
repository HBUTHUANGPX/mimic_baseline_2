from __future__ import annotations

from dataclasses import replace

from newton_lab.envs import ManagerBasedEnvCfg, NewtonSceneCfg
from newton_lab.rl import RslRlAdapterCfg
from newton_lab.tasks import TaskSpec, get_task_spec, list_tasks, register_task


def test_registry_round_trip_returns_deep_copies() -> None:
    task_name = "UnitTest-Registry-v0"
    scene = NewtonSceneCfg(num_envs=2)
    env_cfg = ManagerBasedEnvCfg(scene=scene)
    play_cfg = replace(env_cfg, headless=False)
    agent_cfg = RslRlAdapterCfg(experiment_name="registry")

    register_task(
        TaskSpec(
            name=task_name,
            train_env_cfg=env_cfg,
            play_env_cfg=play_cfg,
            agent_cfg=agent_cfg,
        )
    )

    loaded = get_task_spec(task_name)

    assert task_name in list_tasks()
    assert loaded.name == task_name
    assert loaded is not env_cfg
    assert loaded.train_env_cfg is not env_cfg
    assert loaded.play_env_cfg.headless is False
    loaded.train_env_cfg.scene.num_envs = 99
    assert get_task_spec(task_name).train_env_cfg.scene.num_envs == 2
