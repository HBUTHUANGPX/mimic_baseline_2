# Resampled Retarget NPZ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export Isaac Lab-ready retarget `.npz` files directly from the BVH-to-CSV converter, with default 50 FPS resampling and an optional source-data payload.

**Architecture:** Keep CSV export behavior intact, but replace the old ad-hoc `.npz` payload with a structured schema built from internal animation buffers. Perform robot and human resampling before serialization so downstream Isaac Lab scripts can consume the result without another conversion step.

**Tech Stack:** Python, NumPy, Warp animation buffers, existing soma-retargeter converter pipeline, pytest

---

### Task 1: Lock The New NPZ Schema With Tests

**Files:**
- Modify: `soma-retargeter/tests/test_animation_npz_export.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run `python3 -m pytest soma-retargeter/tests/test_animation_npz_export.py` and verify it fails for missing schema helpers**
- [ ] **Step 3: Add minimal fixture coverage for default resampled payload and optional source payload**
- [ ] **Step 4: Re-run the focused test and verify it passes**

### Task 2: Implement Resampling And Structured Export

**Files:**
- Modify: `soma-retargeter/soma_retargeter/utils/animation_npz.py`

- [ ] **Step 1: Add helper functions for robot trajectory extraction, human local/global sampling, and npz serialization**
- [ ] **Step 2: Keep the implementation NumPy-first and reuse animation buffer sampling rather than CSV round-tripping**
- [ ] **Step 3: Support `output_fps=50` by default and `include_source_data=False` by default**
- [ ] **Step 4: Re-run focused tests and verify they pass**

### Task 3: Wire Converter Config Into Export

**Files:**
- Modify: `soma-retargeter/app/bvh_to_csv_converter.py`
- Modify: `soma-retargeter/assets/default_bvh_to_csv_converter_config.json`
- Modify: `soma-retargeter/assets/default_bvh_to_csv_converter_config_q1.json`

- [ ] **Step 1: Thread `output_fps` and `include_source_data` from config into npz export**
- [ ] **Step 2: Preserve current CSV export behavior**
- [ ] **Step 3: Re-run focused tests and a small real-file export verification**

### Task 4: Verify With A Real NPZ

**Files:**
- Verify only

- [ ] **Step 1: Generate or inspect one exported npz**
- [ ] **Step 2: Confirm required keys, frame counts, and representative shapes**
- [ ] **Step 3: Run `python3 -m py_compile` on touched modules**
