"""Replay retargeted motion npz files in Isaac Lab.

Example:
    source mimic_baseline_2/bin/activate
    python scripts/replay_retarget_npz.py --motion_file soma-retargeter/assets/motions/test-export/Neutral_throw_ball_001__A057.npz --robot g1
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay retargeted motion npz files with Isaac Lab."
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default="soma-retargeter/assets/motions/test-export/Neutral_throw_ball_001__A057.npz",
        help="Path to a retargeted motion npz file.",
    )
    parser.add_argument("--robot", choices=["Q1", "g1"], default="g1")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of replay environments."
    )
    parser.add_argument(
        "--env_spacing", type=float, default=2.0, help="Environment spacing."
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=3.0,
        help="Camera distance behind the root.",
    )
    return parser


@dataclass
class RetargetMotion:
    fps: float
    num_frames: int
    robot_name: str
    robot_joint_names: list[str]
    robot_root_pos: np.ndarray
    robot_root_quat: np.ndarray
    robot_joint_pos: np.ndarray


def load_retarget_motion_npz(npz_path: str | Path) -> RetargetMotion:
    payload = np.load(npz_path, allow_pickle=False)
    return RetargetMotion(
        fps=float(payload["fps"]),
        num_frames=int(payload["num_frames"]),
        robot_name=str(payload["robot_name"].tolist()),
        robot_joint_names=payload["robot_joint_names"].tolist(),
        robot_root_pos=np.asarray(payload["robot_root_pos"], dtype=np.float32),
        robot_root_quat=np.asarray(payload["robot_root_quat"], dtype=np.float32),
        robot_joint_pos=np.asarray(payload["robot_joint_pos"], dtype=np.float32),
    )


def build_root_state(
    root_pos: np.ndarray, root_quat: np.ndarray, env_origins: np.ndarray
) -> np.ndarray:
    num_envs = root_pos.shape[0]
    root_state = np.zeros((num_envs, 13), dtype=np.float32)
    root_state[:, :3] = root_pos + env_origins
    root_state[:, 3:7] = root_quat
    return root_state


def _resolve_asset_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_repo_root() / path).resolve())


def prepare_robot_cfg(robot_cfg: Any):
    prepared = robot_cfg.copy() if hasattr(robot_cfg, "copy") else copy.deepcopy(robot_cfg)
    spawn = getattr(prepared, "spawn", None)
    if spawn is None:
        return prepared

    # Replay only writes pose/joint state directly and does not consume contact data.
    if hasattr(spawn, "activate_contact_sensors"):
        spawn.activate_contact_sensors = False
    if hasattr(spawn, "usd_path") and spawn.usd_path:
        spawn.usd_path = _resolve_asset_path(spawn.usd_path)
    if hasattr(spawn, "asset_path") and spawn.asset_path:
        spawn.asset_path = _resolve_asset_path(spawn.asset_path)
    return prepared


def prepare_joint_state_tensors(
    default_joint_pos,
    default_joint_vel,
    frame_joint_pos,
    joint_indices,
    num_envs: int,
    device: str,
):
    import torch

    joint_pos = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=device).clone()
    joint_vel = torch.as_tensor(default_joint_vel, dtype=torch.float32, device=device).clone()
    if joint_pos.shape[0] != num_envs:
        joint_pos = joint_pos.repeat(num_envs, 1)
    if joint_vel.shape[0] != num_envs:
        joint_vel = joint_vel.repeat(num_envs, 1)
    joint_vel.zero_()
    joint_pos[:, joint_indices] = (
        torch.as_tensor(frame_joint_pos, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(num_envs, 1)
    )
    return joint_pos, joint_vel


def _load_robot_cfg(robot_name: str):
    if robot_name == "Q1":
        from GMT.robots.q1 import Q1_CYLINDER_CFG as robot_cfg
    else:
        from GMT.robots.g1 import G1_CYLINDER_CFG as robot_cfg
    return prepare_robot_cfg(robot_cfg)


def _build_scene_cfg(robot_cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0,
                color=(0.75, 0.75, 0.75),
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
    return ReplaySceneCfg


def run_simulator(
    sim: "sim_utils.SimulationContext",
    scene: "InteractiveScene",
    motion: RetargetMotion,
    camera_distance: float,
    simulation_app,
):
    import torch
    import warp as wp

    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    joint_indices = robot.find_joints(motion.robot_joint_names, preserve_order=True)[0]
    if len(joint_indices) != len(motion.robot_joint_names):
        raise RuntimeError("Failed to match motion joint names to robot joints.")

    env_origins = scene.env_origins.cpu().numpy().astype(np.float32)
    robot_joint_pos = torch.from_numpy(motion.robot_joint_pos).to(sim.device)
    time_step = 0

    while simulation_app.is_running():
        frame_idx = time_step % motion.num_frames

        root_pos = np.repeat(
            motion.robot_root_pos[frame_idx][None, :], scene.num_envs, axis=0
        )
        root_quat = np.repeat(
            motion.robot_root_quat[frame_idx][None, :], scene.num_envs, axis=0
        )
        root_state = torch.from_numpy(
            build_root_state(root_pos, root_quat, env_origins)
        ).to(sim.device)
        # robot.write_root_state_to_sim(root_state)
        robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7])
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:])
        # robot.write_root_pose_to_sim_index(root_pose=root_pose)

        joint_pos, joint_vel = prepare_joint_state_tensors(
            default_joint_pos=robot.data.default_joint_pos,
            default_joint_vel=robot.data.default_joint_vel,
            frame_joint_pos=robot_joint_pos[frame_idx],
            joint_indices=joint_indices,
            num_envs=scene.num_envs,
            device=sim.device,
        )
        # robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_state[0, :3].detach().cpu().numpy()
        sim.set_camera_view(
            pos_lookat + np.array([camera_distance, camera_distance, 0.5]),
            pos_lookat,
        )
        time_step += 1


def main():
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        motion = load_retarget_motion_npz(args_cli.motion_file)
        robot_cfg = _load_robot_cfg(args_cli.robot)
        scene_cfg_cls = _build_scene_cfg(robot_cfg)

        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        sim_cfg.dt = 1.0 / max(motion.fps, 1e-6)
        print(f"Using simulation dt of {sim_cfg.dt:.6f} seconds based on motion fps of {motion.fps:.2f}.")
        sim = SimulationContext(sim_cfg)
        scene_cfg = scene_cfg_cls(
            num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        run_simulator(
            sim=sim,
            scene=scene,
            motion=motion,
            camera_distance=args_cli.camera_distance,
            simulation_app=simulation_app,
        )
    finally:
        print("Closing simulation...")
        simulation_app.close()


if __name__ == "__main__":
    main()
