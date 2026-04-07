from __future__ import annotations

import argparse

from newton_lab.app.common import build_env_from_task
from newton_lab.rl import PlatformRunner
from newton_lab.tasks import get_task_spec, import_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--manifest-path", type=str, default=None)
    args = parser.parse_args()

    import_tasks()
    spec = get_task_spec(args.task)
    env = build_env_from_task(args.task, manifest_path=args.manifest_path)
    env.build()
    runner: PlatformRunner = spec.make_runner(env)
    log_dir = runner.create_log_dir()
    runner.dump_configs(log_dir)
    runner.agent_cfg.max_iterations = max(args.max_iterations, 1)
    runner.train(log_dir)
    print(f"[INFO] training complete: {log_dir}")


if __name__ == "__main__":
    main()
