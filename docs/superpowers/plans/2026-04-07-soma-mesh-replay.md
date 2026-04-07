# SOMA Mesh Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay `--robot soma` motions by driving `soma_base_skel_minimal.usd` as a USD skeleton mesh instead of forcing it through Isaac Lab articulation loading.

**Architecture:** Keep `g1/q1` on the existing articulation-based replay path. Split `replay_retarget_npz.py` so `soma` spawns the mesh USD directly on the stage, then updates the SOMA skeleton animation each frame from `MotionLoader.human_*` tensors.

**Tech Stack:** Isaac Lab, Isaac Sim stage utilities, Pixar USD / UsdSkel, PyTorch, pytest.

---

### Task 1: Prove branch behavior with tests

**Files:**
- Modify: `tests/test_replay_retarget_npz.py`
- Modify: `scripts/replay_retarget_npz.py`

- [ ] Add a failing test that `soma` is excluded from articulation replay helpers and routed to a dedicated human replay branch.
- [ ] Run `source mimic_baseline_2/bin/activate && python -m pytest tests/test_replay_retarget_npz.py -q` and verify it fails for the new expectation.
- [ ] Implement the minimal branch-selection helpers in `scripts/replay_retarget_npz.py`.
- [ ] Re-run `source mimic_baseline_2/bin/activate && python -m pytest tests/test_replay_retarget_npz.py -q` and verify it passes.

### Task 2: Add SOMA mesh playback path

**Files:**
- Modify: `scripts/replay_retarget_npz.py`
- Test: `tests/test_replay_retarget_npz.py`

- [ ] Add a failing test for the helper that resolves the SOMA USD asset path and validates the soma-only replay setup.
- [ ] Run the targeted pytest command and verify the new test fails first.
- [ ] Implement the soma replay path: spawn the USD as a non-articulation asset, initialize UsdSkel handles, and drive mesh animation from `MotionLoader.human_joint_pos` plus root/world motion.
- [ ] Re-run the targeted pytest command.
- [ ] Run a focused manual smoke check: `source mimic_baseline_2/bin/activate && python scripts/replay_retarget_npz.py --robot soma --device cpu`.

### Task 3: Verify no regression for articulation robots

**Files:**
- Modify: `scripts/replay_retarget_npz.py`
- Test: `tests/test_replay_retarget_npz.py`

- [ ] Keep the `g1/q1` articulation path unchanged except for branch wiring.
- [ ] Run `source mimic_baseline_2/bin/activate && python -m py_compile scripts/replay_retarget_npz.py`.
- [ ] Re-run `source mimic_baseline_2/bin/activate && python -m pytest tests/test_replay_retarget_npz.py -q`.
- [ ] Document any remaining manual limitations in the final report.
