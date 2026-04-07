from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from rsl_rl.runners import OnPolicyRunner

from newton_lab.datasets import dataclass_to_dict
from newton_lab.envs import ManagerBasedEnv
from newton_lab.rl.adapter import RslRlAdapter
from newton_lab.rl.config import RslRlAdapterCfg
from newton_lab.rl.on_policy_runner import NewtonLabOnPolicyRunner


class PlatformRunner:
    def __init__(self, env: ManagerBasedEnv, agent_cfg: RslRlAdapterCfg) -> None:
        self.env = env
        self.agent_cfg = agent_cfg

    def create_log_dir(self) -> Path:
        root = Path("logs") / "newton_lab" / self.agent_cfg.experiment_name
        log_dir = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def dump_configs(self, log_dir: Path) -> None:
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        (params_dir / "env.yaml").write_text(yaml.safe_dump(dataclass_to_dict(self.env.cfg), sort_keys=False))
        (params_dir / "agent.yaml").write_text(yaml.safe_dump(asdict(self.agent_cfg), sort_keys=False))

    def make_vecenv(self) -> RslRlAdapter:
        return RslRlAdapter(self.env, self.agent_cfg)

    def dry_run(self, steps: int = 1) -> None:
        env = self.make_vecenv()
        obs, _ = env.reset()
        _ = obs
        import torch

        for _ in range(steps):
            actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
            env.step(actions)

    def train(self, log_dir: Path) -> None:
        runner_cls: type[OnPolicyRunner] = NewtonLabOnPolicyRunner
        vecenv = self.make_vecenv()
        runner = runner_cls(vecenv, asdict(self.agent_cfg), log_dir=str(log_dir), device=str(self.env.device))
        runner.learn(num_learning_iterations=self.agent_cfg.max_iterations, init_at_random_ep_len=True)
