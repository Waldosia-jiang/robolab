# 动作重定向（GMR → Isaac Lab）

[English](README.md) | [中文](README_CN.md)

将人形机器人动作数据转换为 `robolab` 中 AMP / BeyondMimic 训练所需的 Isaac Lab 格式。

[GMR](https://github.com/Roboparty/GMR)（或 TienKung-Lab 导出）中的关节顺序为 **URDF / MuJoCo 树形顺序**，与 Isaac Lab 关节顺序 **不同**。本目录下的脚本会重排 DOF、在仿真中回放动作，并记录关键连杆的世界坐标。

以下命令均假设工作目录为 `robolab/`：

```bash
cd ~/roboparty_train/robolab
```

需要已配置好的 Isaac Lab 环境（与训练相同，需能启动 Isaac Sim）。

---

## 流程概览

```
TienKung-Lab .pkl          GMR 中间格式 .pkl            Isaac Lab .pkl
(motion_amp_expert)   →    (URDF 关节顺序)         →    (Lab 顺序 + 关键体)
     tienkung_to_gmr.py              single_retarget.py / dataset_retarget.py
```

若动作已由 GMR 直接导出，可跳过第一步，从 `single_retarget.py` 或 `dataset_retarget.py` 开始。

---

## 目录结构

| 路径 | 说明 |
|------|------|
| `tienkung_to_gmr.py` | TienKung-Lab AMP 专家数据 → GMR 中间格式 |
| `gmr_to_lab.py` | 核心转换逻辑（供 CLI 脚本导入） |
| `single_retarget.py` | 转换单个 `.pkl`；支持帧裁剪 |
| `dataset_retarget.py` | 批量转换目录内所有 `.pkl` |
| `config/rpo.yaml` | RPO 机器人关节映射 |
| `config/v1_1_29dof.yaml` | v1.1 29 自由度人形关节映射 |

---

## 数据格式

### GMR 中间格式（重定向输入）

Pickle 字典：

| 键 | 形状 / 类型 | 说明 |
|----|-------------|------|
| `fps` | `int` | 帧率 |
| `root_pos` | `(T, 3)` | 根节点平移 |
| `root_rot` | `(T, 4)` | 四元数 **(x, y, z, w)** |
| `dof_pos` | `(T, N)` | GMR / URDF 顺序下的关节角 |
| `local_body_pos` | 可选 | 未使用 |
| `link_body_list` | 可选 | 未使用 |

### Isaac Lab 格式（输出）

Pickle 字典：

| 键 | 形状 / 类型 | 说明 |
|----|-------------|------|
| `fps` | `int` | 帧率 |
| `root_pos` | `(T, 3)` | 根节点平移 |
| `root_rot` | `(T, 4)` | 四元数 **(w, x, y, z)** |
| `dof_pos` | `(T, N)` | Isaac Lab 顺序下的关节角 |
| `loop_mode` | `int` | `0` = 钳位（clamp），`1` = 循环（wrap） |
| `key_body_pos` | `(T, K, 3)` | 关键连杆在世界坐标系下的位置 |

---

## 支持的机器人

| `--robot` | 配置文件 | 典型动作目录 |
|-----------|----------|--------------|
| `rpo` | `config/rpo.yaml` | `data/motions/rpo_gmr` → `data/motions/rpo_lab` |
| `v1_1_29dof` | `config/v1_1_29dof.yaml` | `data/motions/v1_1_29dof_gmr` → `data/motions/v1_1_29dof_lab` |

新增机器人步骤：

1. 在 `config/<robot>.yaml` 中定义 `gmr_dof_names`、`lab_dof_names`、`lab_key_body_names`。
2. 在机器人 `ArticulationCfg` 中设置 `usd_dir` 到 `data/.cache/usd/` 下（避免在共享机器上写入 `/tmp/IsaacLab`）。
3. 在重定向脚本中传入 `--robot <name>`。

---

## 使用方法

### 1. TienKung-Lab → GMR（仅 v1.1 29-DoF）

将 TienKung-Lab 的 AMP 专家片段转为 GMR 中间文件：

```bash
python scripts/tools/retarget/tienkung_to_gmr.py \
    --input_dir ../TienKung-Lab/legged_lab/envs/v1_1_29dof/datasets/motion_amp_expert \
    --output_dir data/motions/v1_1_29dof_gmr \
    --slice-mode all
```

`--slice-mode` 选项：

- `all` — 导出 TienKung `slice` 元数据中的每个片段（默认）
- `first` — 仅导出第一个片段
- `none` — 导出完整片段，忽略 slice 元数据

此步骤 **不需要** Isaac Sim。

### 2. 单文件重定向（GMR → Isaac Lab）

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/tools/retarget/single_retarget.py \
    --robot v1_1_29dof \
    --input_file data/motions/v1_1_29dof_gmr/Forward_backward_02.pkl \
    --output_file data/motions/v1_1_29dof_lab/Forward_backward_02.pkl \
    --config_file scripts/tools/retarget/config/v1_1_29dof.yaml \
    --loop clamp \
    --device cuda:0
```

可选帧范围（闭区间，含首尾）：

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

### 3. 批量重定向（GMR → Isaac Lab）

转换输入目录下所有 `.pkl`。所有动作在同一次仿真中回放（`num_envs` = 文件数量）。**不支持** `--frame_range`。

```bash
python scripts/tools/retarget/dataset_retarget.py \
    --robot rpo \
    --input_dir data/motions/rpo_gmr \
    --output_dir data/motions/rpo_lab \
    --config_file scripts/tools/retarget/config/rpo.yaml \
    --loop clamp \
    --headless
```

v1.1 29-DoF 示例：

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

## 常用参数

`single_retarget.py` 与 `dataset_retarget.py` 共有：

| 参数 | 说明 |
|------|------|
| `--robot` | 机器人名称：`rpo` 或 `v1_1_29dof` |
| `--config_file` | 含 `gmr_dof_names`、`lab_dof_names`、`lab_key_body_names` 的 YAML |
| `--loop {clamp,wrap}` | 动作结束行为（`clamp` = 保持最后一帧） |
| `--headless` | 无 GUI 运行 Isaac Sim |
| `--device {cpu,cuda:0,...}` | 仿真设备（默认 `cuda:0`） |

仅 `single_retarget.py`：

| 参数 | 说明 |
|------|------|
| `--input_file` / `--output_file` | 输入 GMR 与输出 Lab pickle 路径 |
| `--frame_range START END` | 导出帧范围（含首尾） |

仅 `dataset_retarget.py`：

| 参数 | 说明 |
|------|------|
| `--input_dir` / `--output_dir` | GMR 与 Lab pickle 目录 |

---

## 配置文件

`config/` 下每个 YAML 定义三个列表：

- **`gmr_dof_names`** — GMR / URDF 源数据中的关节顺序
- **`lab_dof_names`** — Isaac Lab 关节顺序
- **`lab_key_body_names`** — 用于 `key_body_pos` 的连杆名（须存在于机器人 URDF，并与 AMP 任务配置一致）

v1.1 29-DoF 的左右腿交错 + 腰部三关节映射见 `config/v1_1_29dof.yaml`。

---

## 故障排查

**`PermissionError: ... /tmp/IsaacLab/...`**

共享机器上 `/tmp/IsaacLab` 可能属于其他用户。机器人配置应把 `usd_dir` 设到项目本地路径，例如 `data/.cache/usd/v1_1_29dof`。首次加载 URDF 时会在该目录生成缓存。

**使用 `CUDA_VISIBLE_DEVICES` 时的 CUDA / GPU 警告**

Isaac Sim 会重映射可见 GPU。通过 `CUDA_VISIBLE_DEVICES` 只暴露一块 GPU 时，请使用 `--device cuda:0`。带窗口运行优先选 `Disp.A=On` 的 GPU，或加 `--headless`。

**`DOF name '...' not found in GMR DOF names`**

配置中的 `gmr_dof_names` 与源 pickle 不匹配。请用 `tienkung_to_gmr.py` 重新生成，或核对 GMR 导出的关节顺序。

**`Key body name '...' not found`**

在 YAML 中更新 `lab_key_body_names`，使其与机器人 URDF 中的连杆名一致。

**批量转换：各文件 fps 不一致**

`dataset_retarget.py` 会警告并使用第一个文件的 fps。建议在批量转换前统一帧率。

---

## 参考

- [GMR](https://github.com/Roboparty/GMR) — 动捕动作重定向
- [MimicKit gmr_to_mimickit](https://github.com/xbpeng/MimicKit/blob/main/tools/gmr_to_mimickit/gmr_to_mimickit.py)
- [whole_body_tracking csv_to_npz](https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py)
