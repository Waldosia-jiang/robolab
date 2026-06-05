# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert TienKung-Lab AMP expert `.pkl` files to GMR intermediate format for robolab retargeting.

TienKung-Lab format (nested dict):
    {motion_name: {dof, root_trans_offset, root_rot, fps, slice?, ...}}

GMR intermediate format (flat dict, compatible with gmr_to_lab.extract_gmr_data):
    {fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list, dof_names?}

Usage:
    python scripts/tools/retarget/tienkung_to_gmr.py
    python scripts/tools/retarget/tienkung_to_gmr.py \\
        --input_dir ../../TienKung-Lab/legged_lab/envs/v1_1_29dof/datasets/motion_amp_expert \\
        --output_dir ../../data/motions/v1_1_29dof_gmr \\
        --slice-mode all
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np

# Revolute joint order in TienKung v1.1 29-DoF MuJoCo / URDF (rectified model).
GMR_DOF_NAMES_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _normalize_slices(slice_cfg, total_frames: int) -> list[list[int]]:
    if slice_cfg is None:
        return [[0, total_frames]]

    if isinstance(slice_cfg, np.ndarray):
        slice_cfg = slice_cfg.tolist()

    if isinstance(slice_cfg, tuple):
        slice_cfg = list(slice_cfg)

    if not isinstance(slice_cfg, list) or len(slice_cfg) == 0:
        return [[0, total_frames]]

    if isinstance(slice_cfg[0], (int, float)):
        slice_cfg = [slice_cfg]

    normalized: list[list[int]] = []
    for item in slice_cfg:
        if len(item) != 2:
            raise ValueError(f"Each slice must contain exactly 2 indices, got: {item}")
        start, end = int(item[0]), int(item[1])
        start = max(0, min(start, total_frames))
        end = max(0, min(end, total_frames))
        if end <= start:
            continue
        normalized.append([start, end])

    if not normalized:
        return [[0, total_frames]]
    return normalized


def _extract_motion_dict(raw_data: dict) -> tuple[str, dict]:
    if "dof" in raw_data or "dof_pos" in raw_data:
        return "motion", raw_data

    for name, value in raw_data.items():
        if isinstance(value, dict) and ("dof" in value or "dof_pos" in value):
            return str(name), value

    raise ValueError("Could not find motion payload with 'dof' or 'dof_pos' in pickle.")


def _slice_arrays(motion: dict, start: int, end: int) -> dict[str, np.ndarray]:
    skip_keys = {"fps", "slice", "local_body_pos", "link_body_list", "dof_names"}
    sliced: dict[str, np.ndarray] = {}
    for key, value in motion.items():
        if key in skip_keys or not isinstance(value, np.ndarray):
            continue
        sliced[key] = value[start:end]
    return sliced


def tienkung_motion_to_gmr(motion: dict, start: int, end: int) -> dict:
    segment = _slice_arrays(motion, start, end)

    if "dof" in segment:
        dof_pos = np.asarray(segment["dof"], dtype=np.float32)
    elif "dof_pos" in segment:
        dof_pos = np.asarray(segment["dof_pos"], dtype=np.float32)
    else:
        raise ValueError("Motion segment does not contain 'dof' or 'dof_pos'.")

    if "root_trans_offset" in segment:
        root_pos = np.asarray(segment["root_trans_offset"], dtype=np.float32)
    elif "root_pos" in segment:
        root_pos = np.asarray(segment["root_pos"], dtype=np.float32)
    else:
        raise ValueError("Motion segment does not contain 'root_trans_offset' or 'root_pos'.")

    if "root_rot" not in segment:
        raise ValueError("Motion segment does not contain 'root_rot'.")
    root_rot = np.asarray(segment["root_rot"], dtype=np.float32)

    num_frames = dof_pos.shape[0]
    if root_pos.shape[0] != num_frames or root_rot.shape[0] != num_frames:
        raise ValueError(
            f"Frame count mismatch: dof={num_frames}, root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}"
        )

    if dof_pos.shape[1] != len(GMR_DOF_NAMES_29):
        raise ValueError(
            f"Expected {len(GMR_DOF_NAMES_29)} DoFs, got {dof_pos.shape[1]} in exported segment."
        )

    fps = motion.get("fps", 40)
    if isinstance(fps, np.ndarray):
        fps = int(fps.item())
    else:
        fps = int(fps)

    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,  # xyzw, same as GMR / TienKung storage
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "dof_names": list(GMR_DOF_NAMES_29),
        "source_slice": [start, end],
    }


def _output_name(stem: str, start: int, end: int, num_slices: int) -> str:
    if num_slices <= 1:
        return f"{stem}.pkl"
    return f"{stem}__{start:06d}_{end:06d}.pkl"


def convert_file(input_path: Path, output_dir: Path, slice_mode: str) -> list[Path]:
    raw_data = joblib.load(input_path)
    if not isinstance(raw_data, dict):
        raise ValueError(f"{input_path} does not contain a dictionary.")

    motion_name, motion = _extract_motion_dict(raw_data)
    total_frames = motion.get("dof", motion.get("dof_pos")).shape[0]
    slices = _normalize_slices(motion.get("slice"), total_frames)

    if slice_mode == "none":
        export_slices = [[0, total_frames]]
    elif slice_mode == "first":
        export_slices = [slices[0]]
    elif slice_mode == "all":
        export_slices = slices
    else:
        raise ValueError(f"Unsupported slice_mode: {slice_mode}")

    written: list[Path] = []
    stem = input_path.stem
    for start, end in export_slices:
        gmr_data = tienkung_motion_to_gmr(motion, start, end)
        gmr_data["source_motion_name"] = motion_name
        gmr_data["source_file"] = str(input_path)

        out_name = _output_name(stem, start, end, len(export_slices))
        out_path = output_dir / out_name
        with open(out_path, "wb") as f:
            pickle.dump(gmr_data, f)
        written.append(out_path)
        print(
            f"Saved {out_path.name}: frames={end - start}, fps={gmr_data['fps']}, "
            f"slice=[{start}, {end}]"
        )
    return written


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    robolab_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Convert TienKung-Lab AMP pkl to GMR intermediate format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(
            repo_root
            / "TienKung-Lab"
            / "legged_lab"
            / "envs"
            / "v1_1_29dof"
            / "datasets"
            / "motion_amp_expert"
        ),
        help="Directory containing TienKung-Lab expert .pkl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(robolab_root / "data" / "motions" / "v1_1_29dof_gmr"),
        help="Directory to write GMR-format .pkl files.",
    )
    parser.add_argument(
        "--slice-mode",
        type=str,
        choices=["none", "first", "all"],
        default="all",
        help=(
            "How to handle TienKung 'slice' metadata: "
            "'all' exports every slice segment, 'first' exports only the first slice, "
            "'none' exports the full clip."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(input_dir.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Slice mode: {args.slice_mode}")
    print(f"Found {len(input_files)} source files.\n")

    total_written = 0
    for input_path in input_files:
        print(f"Converting {input_path.name} ...")
        written = convert_file(input_path, output_dir, args.slice_mode)
        total_written += len(written)
        print()

    print(f"Done. Wrote {total_written} GMR file(s) to {output_dir}.")


if __name__ == "__main__":
    main()
