from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import textwrap

import torch

_HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "GMT"
    / "tasks"
    / "tracking"
    / "mdp"
    / "reset_write_helpers.py"
)
_SPEC = importlib.util.spec_from_file_location("tracking_reset_write_helpers", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_env_mask = _MODULE.build_env_mask
write_reset_state_to_sim_mask = _MODULE.write_reset_state_to_sim_mask
_NEWTON_ARTICULATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "IsaacLab"
    / "source"
    / "isaaclab_newton"
    / "isaaclab_newton"
    / "assets"
    / "articulation"
    / "articulation.py"
)


class _FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def write_joint_state_to_sim_mask(
        self, *, position: torch.Tensor, velocity: torch.Tensor, env_mask: torch.Tensor
    ) -> None:
        self.calls.append(("joint", position.clone(), env_mask.clone()))
        self._joint_velocity = velocity.clone()

    def write_root_pose_to_sim_mask(
        self, *, root_pose: torch.Tensor, env_mask: torch.Tensor
    ) -> None:
        self.calls.append(("root_pose", root_pose.clone(), env_mask.clone()))

    def write_root_velocity_to_sim_mask(
        self, *, root_velocity: torch.Tensor, env_mask: torch.Tensor
    ) -> None:
        self.calls.append(("root_velocity", root_velocity.clone(), env_mask.clone()))


def _load_newton_write_joint_state_to_sim_mask():
    source = _NEWTON_ARTICULATION_PATH.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Articulation":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "write_joint_state_to_sim_mask"
                ):
                    method_source = ast.get_source_segment(source, item)
                    namespace: dict[str, object] = {}
                    exec(
                        "from __future__ import annotations\n"
                        "class ExtractedArticulation:\n"
                        f"{textwrap.indent(method_source, '    ')}\n",
                        namespace,
                    )
                    return namespace["ExtractedArticulation"].write_joint_state_to_sim_mask
    raise AssertionError("Could not locate Newton write_joint_state_to_sim_mask")


def test_build_env_mask_marks_only_selected_envs() -> None:
    env_ids = torch.tensor([1, 3], dtype=torch.long)

    mask = build_env_mask(num_envs=5, env_ids=env_ids, device=torch.device("cpu"))

    assert torch.equal(mask, torch.tensor([False, True, False, True, False]))


def test_write_reset_state_to_sim_mask_uses_full_buffers_and_env_mask() -> None:
    robot = _FakeRobot()
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    joint_pos = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    joint_vel = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    root_pos = torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])
    root_ori = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.7]]
    )
    root_lin_vel = torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
    root_ang_vel = torch.tensor([[0.3, 0.4, 0.5], [1.3, 1.4, 1.5], [2.3, 2.4, 2.5]])

    write_reset_state_to_sim_mask(
        robot,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_pos,
        root_ori=root_ori,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        env_ids=env_ids,
    )

    expected_mask = torch.tensor([True, False, True])
    assert [call[0] for call in robot.calls] == ["joint", "root_pose", "root_velocity"]
    assert torch.equal(robot.calls[0][1], joint_pos)
    assert torch.equal(robot._joint_velocity, joint_vel)
    assert torch.equal(robot.calls[0][2], expected_mask)
    assert torch.equal(robot.calls[1][1], torch.cat([root_pos, root_ori], dim=-1))
    assert torch.equal(robot.calls[1][2], expected_mask)
    assert torch.equal(robot.calls[2][1], torch.cat([root_lin_vel, root_ang_vel], dim=-1))
    assert torch.equal(robot.calls[2][2], expected_mask)


def test_newton_write_joint_state_to_sim_mask_forwards_keyword_only_args() -> None:
    method = _load_newton_write_joint_state_to_sim_mask()

    class _KeywordOnlyRobot:
        def __init__(self) -> None:
            self.position_call: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            self.velocity_call: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        write_joint_state_to_sim_mask = method

        def write_joint_position_to_sim_mask(
            self,
            *,
            position: torch.Tensor,
            env_mask: torch.Tensor,
            joint_mask: torch.Tensor,
        ) -> None:
            self.position_call = (position.clone(), env_mask.clone(), joint_mask.clone())

        def write_joint_velocity_to_sim_mask(
            self,
            *,
            velocity: torch.Tensor,
            env_mask: torch.Tensor,
            joint_mask: torch.Tensor,
        ) -> None:
            self.velocity_call = (velocity.clone(), env_mask.clone(), joint_mask.clone())

    robot = _KeywordOnlyRobot()
    position = torch.tensor([[1.0, 2.0]])
    velocity = torch.tensor([[0.1, 0.2]])
    env_mask = torch.tensor([True])
    joint_mask = torch.tensor([True, False])

    robot.write_joint_state_to_sim_mask(
        position=position,
        velocity=velocity,
        env_mask=env_mask,
        joint_mask=joint_mask,
    )

    assert robot.position_call is not None
    assert robot.velocity_call is not None
    assert torch.equal(robot.position_call[0], position)
    assert torch.equal(robot.position_call[1], env_mask)
    assert torch.equal(robot.position_call[2], joint_mask)
    assert torch.equal(robot.velocity_call[0], velocity)
    assert torch.equal(robot.velocity_call[1], env_mask)
    assert torch.equal(robot.velocity_call[2], joint_mask)
