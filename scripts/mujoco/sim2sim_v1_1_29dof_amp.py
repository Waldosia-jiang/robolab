# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""MuJoCo sim2sim for v1_1_29dof AMP policies (96-dim obs, 29-dim actions)."""

import sys
import time

import cv2
import glfw
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robolab.assets import ISAAC_DATA_DIR

NUM_ACTIONS = 29
NUM_SINGLE_OBS = 96
ANGULAR_VEL_SENSOR = "imu-pelvis-angular-velocity"
STANDSTILL_VX = 0.15
FORWARD_CMD_ON = 0.05
WALK_START_VX = 0.5
WALK_BOOTSTRAP_VX = 1.0
STATUS_LOG_INTERVAL_S = 2.0
KEY_DEBOUNCE_S = 0.15

_progress_bar = None

# Isaac Lab articulation order (policy obs joint order).
LAB_DOF_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# MuJoCo actuator / qpos order in v1.1_u3.0_0303_v0_29dof_rectified3.xml.
MUJOCO_DOF_NAMES = [
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
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
]

# lab index i -> mujoco joint index (same role as usd2urdf in sim2sim_rpo_amp.py).
LAB2MUJOCO = [MUJOCO_DOF_NAMES.index(name) for name in LAB_DOF_NAMES]


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1

    min_vx = -0.5
    max_vx = 2.5
    min_vy = -0.5
    max_vy = 0.5
    min_dyaw = -1.5
    max_dyaw = 1.5
    camera_follow = True
    reset_requested = False
    gait_start_requested = False

    @classmethod
    def update_vx(cls, delta):
        prev_vx = cls.vx
        if delta > 0.0 and prev_vx <= FORWARD_CMD_ON:
            # Match --cmd_vx 0.5 cold-start; obs bootstrap still feeds 1.0 m/s while standstill.
            cls.vx = np.clip(WALK_START_VX, cls.min_vx, cls.max_vx)
            cls.gait_start_requested = True
        else:
            cls.vx = np.clip(cls.vx + delta, cls.min_vx, cls.max_vx)
        _log(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def update_vy(cls, delta):
        cls.vy = np.clip(cls.vy + delta, cls.min_vy, cls.max_vy)
        _log(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def update_dyaw(cls, delta):
        cls.dyaw = np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw)
        _log(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        _log(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def reset(cls):
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        _log(f"Velocities reset: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")


_last_key_time = 0.0
_glfw_key_prev: dict[int, bool] = {}


def dispatch_control_key(key_name: str) -> bool:
    """Apply one control action. Returns True if key_name is a control binding."""
    global _last_key_time
    now = time.time()
    if now - _last_key_time < KEY_DEBOUNCE_S:
        return True
    _last_key_time = now

    key_name = key_name.lower()
    if key_name == "up" or key_name == "8":
        cmd.update_vx(cmd.vx_increment)
    elif key_name == "down" or key_name == "2":
        cmd.update_vx(-cmd.vx_increment)
    elif key_name == "left" or key_name == "7":
        cmd.update_dyaw(cmd.dyaw_increment)
    elif key_name == "right" or key_name == "9":
        cmd.update_dyaw(-cmd.dyaw_increment)
    elif key_name == "4":
        cmd.update_vy(cmd.vy_increment)
    elif key_name == "6":
        cmd.update_vy(-cmd.vy_increment)
    elif key_name == "f":
        cmd.toggle_camera_follow()
    elif key_name == "0":
        cmd.reset_requested = True
        _log("Reset requested (0 key pressed)")
    else:
        return False
    return True


def _pynput_key_name(key) -> str | None:
    if key == keyboard.Key.up:
        return "up"
    if key == keyboard.Key.down:
        return "down"
    if key == keyboard.Key.left:
        return "left"
    if key == keyboard.Key.right:
        return "right"
    if hasattr(key, "char") and key.char is not None:
        return key.char.lower()
    # Linux numpad may arrive as KeyCode without char.
    if hasattr(key, "vk") and key.vk is not None:
        vk = key.vk
        numpad_map = {
            65433: "8", 65435: "2", 65430: "4", 65432: "6",
            65429: "7", 65431: "9", 65456: "0",
            104: "8", 98: "2", 100: "4", 102: "6", 96: "0",
            103: "7", 105: "9",
        }
        if vk in numpad_map:
            return numpad_map[vk]
    return None


def on_press(key):
    try:
        name = _pynput_key_name(key)
        if name is not None:
            dispatch_control_key(name)
    except AttributeError:
        pass


def on_release(key):
    pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


_GLFW_KEY_BINDINGS = (
    (glfw.KEY_UP, "up"),
    (glfw.KEY_DOWN, "down"),
    (glfw.KEY_LEFT, "left"),
    (glfw.KEY_RIGHT, "right"),
    (glfw.KEY_8, "8"),
    (glfw.KEY_2, "2"),
    (glfw.KEY_4, "4"),
    (glfw.KEY_6, "6"),
    (glfw.KEY_7, "7"),
    (glfw.KEY_9, "9"),
    (glfw.KEY_0, "0"),
    (glfw.KEY_F, "f"),
    (glfw.KEY_KP_8, "8"),
    (glfw.KEY_KP_2, "2"),
    (glfw.KEY_KP_4, "4"),
    (glfw.KEY_KP_6, "6"),
    (glfw.KEY_KP_7, "7"),
    (glfw.KEY_KP_9, "9"),
    (glfw.KEY_KP_0, "0"),
)


def setup_viewer_input(window) -> None:
    """Use sticky keys; actual repeat suppression is done via rising-edge polling."""
    glfw.set_input_mode(window, glfw.STICKY_KEYS, glfw.TRUE)


def poll_viewer_control_keys(window) -> None:
    """Poll GLFW keys once per sim step (rising edge only, no auto-repeat)."""
    global _glfw_key_prev
    for glfw_key, name in _GLFW_KEY_BINDINGS:
        pressed = glfw.get_key(window, glfw_key) == glfw.PRESS
        if pressed and not _glfw_key_prev.get(glfw_key, False):
            dispatch_control_key(name)
        _glfw_key_prev[glfw_key] = pressed


def _log(message: str) -> None:
    global _progress_bar
    if _progress_bar is not None:
        _progress_bar.write(message)
    else:
        print(message, file=sys.stderr, flush=True)


def print_controls_guide() -> None:
    _log("=" * 60)
    _log("Keyboard controls (MuJoCo window OR terminal — both work):")
    _log("  ↑/↓ or 8/2     : forward / backward (vx)")
    _log("  ←/→ or 7/9     : turn left / right (dyaw)")
    _log("  4/6            : strafe left / right (vy)")
    _log("  0              : reset robot pose and zero commands")
    _log("  F              : toggle camera follow")
    _log("  Tip: first ↑ from standstill = --cmd_vx 0.5 cold start (v1_1 MuJoCo).")
    _log("=" * 60)


def _policy_vx_command(actual_vx: float, vx_cmd: float) -> float:
    """While nearly stationary, bootstrap obs vx so the policy can start walking."""
    if abs(actual_vx) < STANDSTILL_VX and abs(vx_cmd) > FORWARD_CMD_ON:
        return float(np.sign(vx_cmd) * WALK_BOOTSTRAP_VX)
    return float(vx_cmd)


def _apply_walk_cold_start(model, data, initial_qpos, initial_qvel, action, target_pos):
    """Reset sim state to match --cmd_vx 0.5 startup (pose, PD target, policy history)."""
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0.0
    action[:] = 0.0
    target_pos[:] = 0.0
    mujoco.mj_forward(model, data)


def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor(ANGULAR_VEL_SENSOR).data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _scatter_lab_to_mujoco(lab_values):
    mujoco_values = np.zeros(NUM_ACTIONS, dtype=np.double)
    for lab_idx, mujoco_idx in enumerate(LAB2MUJOCO):
        mujoco_values[mujoco_idx] = lab_values[lab_idx]
    return mujoco_values


def _build_robot_config():
    lab_kps = [
        240, 240, 300, 240, 240, 350, 200, 200, 350, 240, 240,
        80, 80, 120, 120, 130, 130, 120, 120, 50, 50, 50, 50,
        40, 40, 40, 40, 40, 40,
    ]
    lab_kds = [
        6, 6, 5, 6, 6, 5, 5, 5, 5, 6, 6,
        1.5, 1.5, 3, 3, 2.5, 2.5, 3, 3, 1.5, 1.5, 1.5, 1.5,
        2, 2, 2, 2, 2, 2,
    ]
    lab_tau = [
        150, 150, 90, 150, 150, 90, 90, 90, 90, 150, 150,
        80, 80, 54, 54, 80, 80, 54, 54, 80, 80, 80, 80,
        11.5, 11.5, 11.5, 11.5, 11.5, 11.5,
    ]
    lab_default = [
        -0.11, -0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25,
        0.15, 0.15, -0.14, -0.14, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
        1.10, 1.10, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
    ]

    class robot_config:
        kps = _scatter_lab_to_mujoco(lab_kps)
        kds = _scatter_lab_to_mujoco(lab_kds)
        default_pos = _scatter_lab_to_mujoco(lab_default)
        tau_limit = _scatter_lab_to_mujoco(lab_tau)
        frame_stack = 1
        num_single_obs = NUM_SINGLE_OBS
        num_observations = NUM_SINGLE_OBS * frame_stack
        num_actions = NUM_ACTIONS
        action_scale = 0.25
        lab2mujoco = LAB2MUJOCO
        joint_names = LAB_DOF_NAMES

    return robot_config


def run_mujoco(policy, cfg, headless=False, max_steps=None, initial_cmd=(0.0, 0.0, 0.0)):
    global _progress_bar

    cmd.vx, cmd.vy, cmd.dyaw = initial_cmd
    cmd.reset_requested = False
    cmd.gait_start_requested = False

    print_controls_guide()
    # Always listen on the terminal; GLFW polling handles the MuJoCo window.
    keyboard_listener = start_keyboard_listener()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions :] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    render_width = 640 if headless else 1920
    render_height = 480 if headless else 1080

    if headless:
        renderer = mujoco.Renderer(model, width=render_width, height=render_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam = mujoco.MjvCamera()
        cam.distance = 4.0
        cam.azimuth = 45.0
        cam.elevation = -20.0
        cam.lookat = [0, 0, 1]
        out = cv2.VideoWriter(
            "simulation.mp4",
            fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (render_width, render_height),
        )
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=1920, height=1080)
        setup_viewer_input(viewer.window)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]

    target_pos = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    action = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_observations), dtype=np.double)
    count_lowlevel = 0
    is_first_frame = True
    last_status_log = time.time()

    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = []
    actual_ang_vel_data = []

    start_time = time.time()
    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    joint_pos_slice = slice(9, 9 + NUM_ACTIONS)
    joint_vel_slice = slice(9 + NUM_ACTIONS, 9 + 2 * NUM_ACTIONS)
    action_slice = slice(9 + 2 * NUM_ACTIONS, 9 + 3 * NUM_ACTIONS)

    step_iter = (
        tqdm(range(total_steps), desc="Simulating...", mininterval=1.0, file=sys.stderr)
        if headless
        else range(total_steps)
    )
    _progress_bar = step_iter if headless else None

    for step in step_iter:
        if not headless:
            poll_viewer_control_keys(viewer.window)

        if cmd.reset_requested:
            _log("Performing reset: restoring qpos/qvel and zeroing commands")
            cmd.reset()
            _apply_walk_cold_start(model, data, initial_qpos, initial_qvel, action, target_pos)
            is_first_frame = True
            count_lowlevel = 0
            cmd.gait_start_requested = False
            cmd.reset_requested = False

        q, dq, _, v, omega, gvec = get_obs(data)

        if cmd.gait_start_requested:
            _log("Walk cold start: same as --cmd_vx {:.1f}".format(WALK_START_VX))
            _apply_walk_cold_start(model, data, initial_qpos, initial_qvel, action, target_pos)
            is_first_frame = True
            count_lowlevel = 0
            cmd.gait_start_requested = False
            q, dq, _, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot_config.num_actions :]
        dq = dq[-cfg.robot_config.num_actions :]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            dq_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            q_rel = q - cfg.robot_config.default_pos
            for lab_idx, mujoco_idx in enumerate(cfg.robot_config.lab2mujoco):
                q_obs[lab_idx] = q_rel[mujoco_idx]
                dq_obs[lab_idx] = dq[mujoco_idx]

            obs = np.zeros([1, cfg.robot_config.num_observations], dtype=np.float32)
            obs[0, 0:3] = omega
            obs[0, 3:6] = gvec
            obs[0, 6] = _policy_vx_command(v[0], cmd.vx)
            obs[0, 7] = cmd.vy
            obs[0, 8] = cmd.dyaw
            obs[0, joint_pos_slice] = q_obs
            obs[0, joint_vel_slice] = dq_obs
            obs[0, action_slice] = action

            now = time.time()
            if now - last_status_log >= STATUS_LOG_INTERVAL_S:
                last_status_log = now
                _log(
                    "cmd vx={:.2f} vy={:.2f} dyaw={:.2f} | "
                    "actual vx={:.2f} vy={:.2f} wz={:.2f} | base_z={:.3f}".format(
                        cmd.vx, cmd.vy, cmd.dyaw, v[0], v[1], omega[2], data.qpos[2]
                    )
                )

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale
            for lab_idx, mujoco_idx in enumerate(cfg.robot_config.lab2mujoco):
                target_pos[mujoco_idx] = target_q[lab_idx]
            target_pos = target_pos + cfg.robot_config.default_pos

            q_low_freq = q.copy()
            v_low_freq = v[:2].copy()
            omega_low_freq = omega[2].copy()

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq)
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v_low_freq)
            actual_ang_vel_data.append(omega_low_freq)

            if headless:
                renderer.update_scene(data, camera=cam)
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                out.write(renderer.render())
            else:
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()

        target_vel = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
        tau = pd_control(target_pos, q, cfg.robot_config.kps, target_vel, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

        elapsed_real_time = time.time() - start_time
        target_sim_time = (step + 1) * cfg.sim_config.dt
        if elapsed_real_time < target_sim_time:
            time.sleep(target_sim_time - elapsed_real_time)

    if headless:
        out.release()
    else:
        viewer.close()

    _progress_bar = None
    keyboard_listener.stop()
    _save_plots(cfg, time_data, commanded_joint_pos_data, actual_joint_pos_data,
                commanded_lin_vel_x_data, commanded_lin_vel_y_data, commanded_ang_vel_z_data,
                actual_lin_vel_data, actual_ang_vel_data)


def _save_plots(cfg, time_data, commanded_joint_pos_data, actual_joint_pos_data,
                commanded_lin_vel_x_data, commanded_lin_vel_y_data, commanded_ang_vel_z_data,
                actual_lin_vel_data, actual_ang_vel_data):
    if not time_data:
        return

    print("Simulation finished. Generating plots...")
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)

    num_joints = cfg.robot_config.num_actions
    n_cols = 4
    n_rows = (num_joints + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    for i in range(num_joints):
        ax = axes1[i]
        ax.plot(time_data, commanded_joint_pos_data[:, i], label="Commanded", linestyle="--")
        ax.plot(time_data, actual_joint_pos_data[:, i], label="Actual")
        ax.set_title(cfg.robot_config.joint_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)

    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])

    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes2[0].plot(time_data, commanded_lin_vel_x_data, label="Commanded Vx", linestyle="--")
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label="Actual Vx")
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    axes2[1].plot(time_data, commanded_lin_vel_y_data, label="Commanded Vy", linestyle="--")
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label="Actual Vy")
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    axes2[2].plot(time_data, commanded_ang_vel_z_data, label="Commanded Dyaw", linestyle="--")
    axes2[2].plot(time_data, actual_ang_vel_data, label="Actual Dyaw")
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()

    fig1.savefig("joint_positions.png")
    fig2.savefig("base_velocities.png")
    print("Plots finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v1_1_29dof AMP MuJoCo sim2sim deployment.")
    parser.add_argument(
        "--load_model",
        type=str,
        required=True,
        help="Path to exported/policy.pt (TorchScript, 96-dim input).",
    )
    parser.add_argument(
        "--mujoco_model",
        type=str,
        default=f"{ISAAC_DATA_DIR}/robots/roboparty/v1_1_29dof/mjcf/v1.1_u3.0_0303_v0_29dof_rectified3.xml",
        help="MuJoCo XML model path.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save simulation.mp4")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap on physics steps (for smoke tests).")
    parser.add_argument("--cmd_vx", type=float, default=0.0, help="Initial forward velocity command [m/s].")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Initial lateral velocity command [m/s].")
    parser.add_argument("--cmd_dyaw", type=float, default=0.0, help="Initial yaw rate command [rad/s].")
    args = parser.parse_args()

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = args.mujoco_model
            sim_duration = 1_000_000.0
            dt = 0.005
            decimation = 4

        robot_config = _build_robot_config()

    policy = torch.jit.load(args.load_model, map_location="cpu")
    run_mujoco(
        policy,
        Sim2simCfg(),
        headless=args.headless,
        max_steps=args.max_steps,
        initial_cmd=(args.cmd_vx, args.cmd_vy, args.cmd_dyaw),
    )
