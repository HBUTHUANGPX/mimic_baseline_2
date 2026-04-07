from __future__ import annotations

from pathlib import Path

from newton_lab.tasks import get_task_spec, import_tasks


def build_env_from_task(task: str, *, play: bool = False, manifest_path: Path | None = None):
    import_tasks()
    spec = get_task_spec(task)
    env = spec.make_env(play=play)
    if manifest_path is not None:
        env.cfg.dataset.manifest_path = Path(manifest_path)
    return env
