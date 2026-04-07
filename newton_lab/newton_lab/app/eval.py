from __future__ import annotations

import argparse

import torch

from newton_lab.app.common import build_env_from_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--manifest-path", type=str, default=None)
    args = parser.parse_args()

    env = build_env_from_task(args.task, play=True, manifest_path=args.manifest_path)
    env.build()
    obs, _ = env.reset()
    for _ in range(args.steps):
        actions = torch.zeros((env.num_envs, env.action_manager.action_dim), dtype=torch.float32)
        obs, rewards, dones, _ = env.step(actions)
        _ = (obs, rewards, dones)
    print("[INFO] eval run complete")


if __name__ == "__main__":
    main()
