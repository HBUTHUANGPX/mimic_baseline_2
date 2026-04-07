from __future__ import annotations

from pathlib import Path

from newton_lab.rl import PlatformRunner
from newton_lab.tasks import get_task_spec, import_tasks


def build_env_from_task(task: str, *, play: bool = False, manifest_path: Path | None = None):
    import_tasks()
    spec = get_task_spec(task)
    env = spec.make_env(play=play)
    if manifest_path is not None:
        env.cfg.dataset.manifest_path = Path(manifest_path)
    return env


def load_policy_for_task(task: str, checkpoint_path: Path, *, manifest_path: Path | None = None):
    import_tasks()
    spec = get_task_spec(task)
    env = build_env_from_task(task, play=True, manifest_path=manifest_path)
    env.build()
    runner = PlatformRunner(env, spec.agent_cfg)
    return runner.load_inference_components(Path(checkpoint_path))
