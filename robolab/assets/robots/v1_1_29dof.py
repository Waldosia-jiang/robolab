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


import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

V1_1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/v1_1_29dof/urdf/v1_1_29dof.urdf",
        usd_dir=f"{ISAAC_DATA_DIR}/.cache/usd/v1_1_29dof",
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={".*": 1000.0},
                damping={".*": 100.0}
            )
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.87),
        joint_pos={
            ".*_hip_pitch_joint": -0.11,
            ".*_knee_joint": 0.25,
            ".*_ankle_pitch_joint": -0.14,
            ".*_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.15,
            "right_shoulder_roll_joint": -0.15,
            ".*_elbow_joint": 1.10,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        # joint_pos={
            # ".*_hip_pitch_joint": -0.1,
            # ".*_knee_joint": 0.24,
            # ".*_ankle_pitch_joint": -0.14,
            # ".*_elbow_joint": 0.87,
            # "left_shoulder_roll_joint": 0.18,
            # "right_shoulder_roll_joint": -0.18,
            # ".*_shoulder_pitch_joint": 0.35,
            # "left_wrist_roll_joint": 0.15,
            # "right_wrist_roll_joint": -0.15,

        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 90.0,
                ".*_knee_joint": 150.0,
                ".*_ankle_pitch_joint": 54.0,
                ".*_ankle_roll_joint": 54.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 14.6608,
                ".*_hip_roll_joint": 14.6608,
                ".*_hip_yaw_joint": 15.1811,
                ".*_knee_joint": 14.6608,
                ".*_ankle_pitch_joint": 9.3201,
                ".*_ankle_roll_joint": 9.3201,
            },
            stiffness={
                ".*_hip_pitch_joint": 240.0,
                ".*_hip_roll_joint": 240.0,
                ".*_hip_yaw_joint": 200.0,
                ".*_knee_joint": 240.0,
                ".*_ankle_pitch_joint": 120.0,
                ".*_ankle_roll_joint": 120.0,
            },
            damping={
                ".*_hip_pitch_joint": 6.0,
                ".*_hip_roll_joint": 6.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 6.0,
                ".*_ankle_pitch_joint": 3.0,
                ".*_ankle_roll_joint": 3.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*waist_pitch_joint",
                ".*waist_roll_joint",
                ".*waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*waist_yaw_joint": 90,
                ".*waist_roll_joint": 90,
                ".*waist_pitch_joint": 90,
            },
            velocity_limit_sim={
                ".*waist_yaw_joint": 15.1811,
                ".*waist_roll_joint": 14.8702,
                ".*waist_pitch_joint": 14.8702,
            },
            stiffness={
                ".*waist_yaw_joint": 300.0,
                ".*waist_roll_joint": 350.0,
                ".*waist_pitch_joint": 350.0,
            },
            damping={
                ".*waist_yaw_joint": 5.0,
                ".*waist_roll_joint": 5.0,
                ".*waist_pitch_joint": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 80.0,
                ".*_shoulder_roll_joint": 80.0,
                ".*_shoulder_yaw_joint": 80.0,
                ".*_elbow_joint": 80.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 3.0369,
                ".*_shoulder_roll_joint": 3.0369,
                ".*_shoulder_yaw_joint": 3.6652,
                ".*_elbow_joint": 3.6652,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 80,
                ".*_shoulder_roll_joint": 130,
                ".*_shoulder_yaw_joint": 50,
                ".*_elbow_joint": 50,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.5,
                ".*_shoulder_roll_joint": 2.5,
                ".*_shoulder_yaw_joint": 1.5,
                ".*_elbow_joint": 1.5,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",  
            ],
            effort_limit_sim={
                ".*_wrist_roll_joint": 11.5,
                ".*_wrist_pitch_joint": 11.5,
                ".*_wrist_yaw_joint": 11.5,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 3.6652,
                ".*_wrist_pitch_joint": 3.6652,
                ".*_wrist_yaw_joint": 3.6652,
            },
            stiffness={
                ".*_wrist_roll_joint": 40,
                ".*_wrist_pitch_joint": 40,
                ".*_wrist_yaw_joint": 40,
            },
            damping=2,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)

LEG_JOINT_NAMES = [
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                ".*waist_yaw_joint",
                ".*waist_roll_joint",
                ".*waist_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",             
]

