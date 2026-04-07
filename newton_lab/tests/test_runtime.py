from __future__ import annotations

from pathlib import Path

import torch

from newton_lab.robots.g1 import G1RobotCfg
from newton_lab.sim import NewtonSim, NewtonSimCfg


def test_newton_sim_builds_g1_and_reads_state() -> None:
    asset_path = Path("assets/unitree_g1/g1_29dof_rev_1_0.xml")
    sim = NewtonSim(
        NewtonSimCfg(num_envs=2, device="cpu"),
        G1RobotCfg(asset_path=asset_path),
    )
    sim.build()

    joint_pos = sim.get_joint_positions()
    joint_vel = sim.get_joint_velocities()
    body_pos = sim.get_body_positions()

    assert joint_pos.shape[0] == 2
    assert joint_vel.shape[0] == 2
    assert body_pos.shape[0] == 2
    assert sim.metadata["body_count"] > 0
    assert sim.metadata["joint_coord_count"] > 0

    env_ids = torch.tensor([1], dtype=torch.long)
    target = torch.full((1, joint_pos.shape[1]), 0.05)
    sim.write_joint_state(env_ids=env_ids, joint_pos=target)
    sim.step(action_targets=torch.zeros((2, sim.action_dim), dtype=torch.float32))

    updated = sim.get_joint_positions()
    assert torch.allclose(updated[1], target[0], atol=1e-5)
