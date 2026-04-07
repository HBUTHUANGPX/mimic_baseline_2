from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from newton_lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from newton_lab.rl import PlatformRunner, RslRlAdapterCfg


@dataclass
class TaskSpec:
    name: str
    train_env_cfg: ManagerBasedEnvCfg
    play_env_cfg: ManagerBasedEnvCfg
    agent_cfg: RslRlAdapterCfg
    runner_cls: type | None = None

    def make_env(self, *, play: bool = False) -> ManagerBasedEnv:
        cfg = deepcopy(self.play_env_cfg if play else self.train_env_cfg)
        return ManagerBasedEnv(cfg)

    def make_runner(self, env: ManagerBasedEnv):
        runner_cls = self.runner_cls or PlatformRunner
        return runner_cls(env, deepcopy(self.agent_cfg))


_REGISTRY: dict[str, TaskSpec] = {}


def register_task(task_spec: TaskSpec) -> None:
    if task_spec.name in _REGISTRY:
        raise ValueError(f"Task '{task_spec.name}' is already registered")
    _REGISTRY[task_spec.name] = deepcopy(task_spec)


def get_task_spec(name: str) -> TaskSpec:
    return deepcopy(_REGISTRY[name])


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())
