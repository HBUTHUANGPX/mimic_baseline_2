from __future__ import annotations

import argparse

import torch

from newton_lab.app.common import build_env_from_task, load_policy_for_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint is not None:
        policy, env = load_policy_for_task(args.task, args.checkpoint, manifest_path=args.manifest_path)
        obs, _ = env.reset()
        for _ in range(args.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    else:
        env = build_env_from_task(args.task, play=True, manifest_path=args.manifest_path)
        env.build()
        env.reset()
        for _ in range(args.steps):
            actions = torch.zeros((env.num_envs, env.action_manager.action_dim), dtype=torch.float32)
            env.step(actions)
    print("[INFO] play run complete")


if __name__ == "__main__":
    main()
