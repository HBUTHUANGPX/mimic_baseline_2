"""Replay retargeted motion npz files in Isaac Lab.

Example:
    source mimic_baseline_2/bin/activate
    python scripts/replay_retarget_npz.py --motion_file soma-retargeter/assets/motions/test-export/Neutral_throw_ball_001__A057.npz --robot g1
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

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

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def prepare_robot_cfg(robot_cfg):
    prepared = robot_cfg.copy()
    if hasattr(prepared, "spawn") and hasattr(
        prepared.spawn, "activate_contact_sensors"
    ):
        prepared.spawn.activate_contact_sensors = False
    return prepared


import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if args_cli.robot == "Q1":
    from GMT.robots.q1 import Q1_CYLINDER_CFG as ROBOT_CFG
else:
    from GMT.robots.g1 import G1_CYLINDER_CFG as ROBOT_CFG
ROBOT_CFG = prepare_robot_cfg(ROBOT_CFG)


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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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


def parse_args():

    return parser.parse_args()


motion = load_retarget_motion_npz(args_cli.motion_file)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
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
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        joint_pos[:, joint_indices] = (
            robot_joint_pos[frame_idx].unsqueeze(0).repeat(scene.num_envs, 1)
        )
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_state[0, :3].detach().cpu().numpy()
        sim.set_camera_view(
            pos_lookat
            + np.array([args_cli.camera_distance, args_cli.camera_distance, 0.5]),
            pos_lookat,
        )
        time_step += 1

    simulation_app.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / max(motion.fps, 1e-6)
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplaySceneCfg(
        num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
