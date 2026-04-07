from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from newton_lab.app.common import build_env_from_task
from newton_lab.rl import PlatformRunner
from newton_lab.tasks import get_task_spec, import_tasks


def _write_motion_npz(path: Path, *, frames: int = 6) -> None:
    robot_joint_pos = np.zeros((frames, 6), dtype=np.float32)
    robot_body_pos = np.zeros((frames, 3, 3), dtype=np.float32)
    robot_body_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    robot_body_quat[..., 3] = 1.0
    human_local_transforms = np.zeros((frames, 5, 7), dtype=np.float32)
    human_local_transforms[..., 3] = 1.0
    human_global_pos = np.zeros((frames, 5, 3), dtype=np.float32)
    human_global_quat = np.zeros((frames, 5, 4), dtype=np.float32)
    human_global_quat[..., 3] = 1.0
    np.savez(
        path,
        fps=np.array(50, dtype=np.int32),
        robot_joint_names=np.array([f"joint_{idx}" for idx in range(6)]),
        robot_body_names=np.array([f"body_{idx}" for idx in range(3)]),
        human_joint_names=np.array([f"human_{idx}" for idx in range(5)]),
        robot_joint_pos=robot_joint_pos,
        robot_body_pos=robot_body_pos,
        robot_body_quat=robot_body_quat,
        human_local_transforms=human_local_transforms,
        human_global_pos=human_global_pos,
        human_global_quat=human_global_quat,
    )


def test_velocity_task_spec_builds_stepable_env() -> None:
    import_tasks()
    spec = get_task_spec("newton_lab.g1.velocity")
    env = spec.make_env()
    env.build()
    obs, _ = env.reset()
    actions = torch.zeros((env.num_envs, env.action_manager.action_dim), dtype=torch.float32)
    _, rewards, dones, _ = env.step(actions)

    assert obs["policy"].shape[0] == env.num_envs
    assert rewards.shape == (env.num_envs,)
    assert dones.shape == (env.num_envs,)


def test_tracking_task_spec_uses_manifest_dataset(tmp_path: Path) -> None:
    import_tasks()
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion)
    manifest = tmp_path / "tracking.yaml"
    manifest.write_text(
        """
name: smoke
groups:
  demo:
    paths:
      - motion.npz
""".strip()
    )

    spec = get_task_spec("newton_lab.g1.tracking")
    spec.train_env_cfg.dataset.manifest_path = manifest
    env = spec.make_env()
    env.build()
    obs, extras = env.reset()

    assert obs["policy"].shape[0] == env.num_envs
    assert "reference_motion" in extras


def test_build_env_from_task_applies_manifest_override(tmp_path: Path) -> None:
    import_tasks()
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion)
    manifest = tmp_path / "tracking.yaml"
    manifest.write_text(
        """
name: smoke
groups:
  demo:
    paths:
      - motion.npz
""".strip()
    )

    env = build_env_from_task("newton_lab.g1.tracking", manifest_path=manifest)
    env.build()
    _, extras = env.reset()

    assert "reference_motion" in extras


def test_velocity_runner_checkpoint_can_be_reloaded_for_play(tmp_path: Path) -> None:
    import_tasks()
    spec = get_task_spec("newton_lab.g1.velocity")
    spec.agent_cfg.max_iterations = 1
    spec.agent_cfg.experiment_name = "test_velocity_roundtrip"
    env = spec.make_env()
    env.build()
    runner = PlatformRunner(env, spec.agent_cfg)
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    runner.dump_configs(log_dir)
    checkpoint = runner.train(log_dir)

    policy, play_env = runner.load_inference_components(checkpoint)
    obs, _ = play_env.reset()
    actions = policy(obs)
    next_obs, rewards, dones, _ = play_env.step(actions)

    assert checkpoint.exists()
    assert actions.shape[0] == play_env.num_envs
    assert next_obs["policy"].shape[0] == play_env.num_envs
    assert rewards.shape == (play_env.num_envs,)
    assert dones.shape == (play_env.num_envs,)
