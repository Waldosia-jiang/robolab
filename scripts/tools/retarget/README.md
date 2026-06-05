# Motion Retargeting (GMR → Isaac Lab)

[English](README.md) | [中文](README_CN.md)

Tools to convert humanoid motion data into the Isaac Lab format used by AMP / BeyondMimic training in `robolab`.

Motion from [GMR](https://github.com/Roboparty/GMR) (or TienKung-Lab exports) stores joints in **URDF / MuJoCo tree order**. Isaac Lab articulations use a **different joint order**. These scripts reorder DOFs, replay motions in simulation, and record key-body world positions.

All commands below assume the working directory is `robolab/`:

```bash
cd ~/roboparty_train/robolab
```

Requires an Isaac Lab environment with Isaac Sim available (same setup as training).

---

## Pipeline overview

```
TienKung-Lab .pkl          GMR intermediate .pkl         Isaac Lab .pkl
(motion_amp_expert)   →    (URDF joint order)       →    (lab order + key bodies)
     tienkung_to_gmr.py              single_retarget.py / dataset_retarget.py
```

For motions already exported by GMR, skip the first step and start from `single_retarget.py` or `dataset_retarget.py`.

---

## Directory layout

| Path | Description |
|------|-------------|
| `tienkung_to_gmr.py` | TienKung-Lab AMP expert → GMR intermediate format |
| `gmr_to_lab.py` | Core conversion logic (imported by the CLI scripts) |
| `single_retarget.py` | Convert one `.pkl` file; supports frame clipping |
| `dataset_retarget.py` | Batch-convert all `.pkl` files in a directory |
| `config/rpo.yaml` | Joint mapping for the RPO robot |
| `config/v1_1_29dof.yaml` | Joint mapping for v1.1 29-DoF humanoid |

---

## Data formats

### GMR intermediate (input to retarget)

Pickle dict:

| Key | Shape / type | Notes |
|-----|--------------|-------|
| `fps` | `int` | Frame rate |
| `root_pos` | `(T, 3)` | Root translation |
| `root_rot` | `(T, 4)` | Quaternion **(x, y, z, w)** |
| `dof_pos` | `(T, N)` | Joint angles in GMR / URDF order |
| `local_body_pos` | optional | Unused |
| `link_body_list` | optional | Unused |

### Isaac Lab (output)

Pickle dict:

| Key | Shape / type | Notes |
|-----|--------------|-------|
| `fps` | `int` | Frame rate |
| `root_pos` | `(T, 3)` | Root translation |
| `root_rot` | `(T, 4)` | Quaternion **(w, x, y, z)** |
| `dof_pos` | `(T, N)` | Joint angles in Isaac Lab order |
| `loop_mode` | `int` | `0` = clamp, `1` = wrap |
| `key_body_pos` | `(T, K, 3)` | Key link positions in world frame |

---

## Supported robots

| `--robot` | Config file | Typical motion dirs |
|-----------|-------------|---------------------|
| `rpo` | `config/rpo.yaml` | `data/motions/rpo_gmr` → `data/motions/rpo_lab` |
| `v1_1_29dof` | `config/v1_1_29dof.yaml` | `data/motions/v1_1_29dof_gmr` → `data/motions/v1_1_29dof_lab` |

Add a new robot by:

1. Defining `gmr_dof_names`, `lab_dof_names`, and `lab_key_body_names` in `config/<robot>.yaml`.
2. Ensuring the robot `ArticulationCfg` sets `usd_dir` under `data/.cache/usd/` (avoids writing to `/tmp/IsaacLab` on shared machines).
3. Passing `--robot <name>` to the retarget scripts.

---

## Usage

### 1. TienKung-Lab → GMR (v1.1 29-DoF only)

Convert AMP expert clips from TienKung-Lab into GMR intermediate files:

```bash
python scripts/tools/retarget/tienkung_to_gmr.py \
    --input_dir ../TienKung-Lab/legged_lab/envs/v1_1_29dof/datasets/motion_amp_expert \
    --output_dir data/motions/v1_1_29dof_gmr \
    --slice-mode all
```

`--slice-mode`:

- `all` — export every segment listed in TienKung `slice` metadata (default)
- `first` — export only the first slice
- `none` — export the full clip, ignore slice metadata

This step does **not** require Isaac Sim.

### 2. Single-file retarget (GMR → Isaac Lab)

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/tools/retarget/single_retarget.py \
    --robot v1_1_29dof \
    --input_file data/motions/v1_1_29dof_gmr/Forward_backward_02.pkl \
    --output_file data/motions/v1_1_29dof_lab/Forward_backward_02.pkl \
    --config_file scripts/tools/retarget/config/v1_1_29dof.yaml \
    --loop clamp \
    --device cuda:0
```

Optional frame range (inclusive):

```bash
python scripts/tools/retarget/single_retarget.py \
    --robot rpo \
    --input_file data/motions/rpo_gmr/walk.pkl \
    --output_file data/motions/rpo_lab/walk_clip.pkl \
    --config_file scripts/tools/retarget/config/rpo.yaml \
    --frame_range 10 100 \
    --loop wrap \
    --headless
```

### 3. Batch retarget (GMR → Isaac Lab)

Converts every `.pkl` in the input directory. All motions are replayed in one simulation session (`num_envs` = number of files). Does **not** support `--frame_range`.

```bash
python scripts/tools/retarget/dataset_retarget.py \
    --robot rpo \
    --input_dir data/motions/rpo_gmr \
    --output_dir data/motions/rpo_lab \
    --config_file scripts/tools/retarget/config/rpo.yaml \
    --loop clamp \
    --headless
```

v1.1 29-DoF example:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/tools/retarget/dataset_retarget.py \
    --robot v1_1_29dof \
    --input_dir data/motions/v1_1_29dof_gmr \
    --output_dir data/motions/v1_1_29dof_lab \
    --config_file scripts/tools/retarget/config/v1_1_29dof.yaml \
    --loop clamp \
    --headless \
    --device cuda:0
```

---

## Common arguments

Shared by `single_retarget.py` and `dataset_retarget.py`:

| Argument | Description |
|----------|-------------|
| `--robot` | Robot name: `rpo` or `v1_1_29dof` |
| `--config_file` | YAML with `gmr_dof_names`, `lab_dof_names`, `lab_key_body_names` |
| `--loop {clamp,wrap}` | End-of-motion behavior (`clamp` = hold last frame) |
| `--headless` | Run Isaac Sim without GUI |
| `--device {cpu,cuda:0,...}` | Simulation device (default: `cuda:0`) |

`single_retarget.py` only:

| Argument | Description |
|----------|-------------|
| `--input_file` / `--output_file` | Paths to input GMR and output Lab pickles |
| `--frame_range START END` | Inclusive frame indices to export |

`dataset_retarget.py` only:

| Argument | Description |
|----------|-------------|
| `--input_dir` / `--output_dir` | Directories of GMR and Lab pickles |

---

## Config file

Each YAML under `config/` defines three lists:

- **`gmr_dof_names`** — joint order in the GMR / URDF source data
- **`lab_dof_names`** — joint order in the Isaac Lab articulation
- **`lab_key_body_names`** — link names for `key_body_pos` (must exist on the robot URDF and match the AMP task config)

See `config/v1_1_29dof.yaml` for the v1.1 29-DoF mapping (interleaved L/R legs + waist triple).

---

## Troubleshooting

**`PermissionError: ... /tmp/IsaacLab/...`**

Another user owns `/tmp/IsaacLab` on shared machines. Robot configs should set `usd_dir` to a project-local path, e.g. `data/.cache/usd/v1_1_29dof`. First URDF load will create the cache there.

**CUDA / GPU warnings with `CUDA_VISIBLE_DEVICES`**

Isaac Sim remaps visible GPUs. Use `--device cuda:0` when only one GPU is visible via `CUDA_VISIBLE_DEVICES`. Prefer a GPU with display attached (`Disp.A=On`) for windowed runs, or pass `--headless`.

**`DOF name '...' not found in GMR DOF names`**

The config `gmr_dof_names` does not match the source pickle. Regenerate with `tienkung_to_gmr.py` or verify GMR export joint order.

**`Key body name '...' not found`**

Update `lab_key_body_names` in the YAML to match link names in the robot URDF.

**Batch run: motions with different fps**

`dataset_retarget.py` warns and uses the fps of the first file. Normalize fps before batch conversion if possible.

---

## References

- [GMR](https://github.com/Roboparty/GMR) — motion retargeting from mocap
- [MimicKit gmr_to_mimickit](https://github.com/xbpeng/MimicKit/blob/main/tools/gmr_to_mimickit/gmr_to_mimickit.py)
- [whole_body_tracking csv_to_npz](https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py)
