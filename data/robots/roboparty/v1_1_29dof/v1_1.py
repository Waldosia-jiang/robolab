import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

V1_1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # asset_path=f"{ISAAC_ASSET_DIR}/v1_1_29dof/urdf/v1.1_1125_woUI_V2.urdf",
        # asset_path=f"{ISAAC_ASSET_DIR}/v1_1_29dof/urdf/v1.1_Whole_u2.1_0114_v1.urdf",
        # asset_path=f"{ISAAC_ASSET_DIR}/v1_1_29dof/urdf/v1.1_Whole_u3.0_0122_v1.urdf",
        asset_path=f"{ISAAC_ASSET_DIR}/v1_1_29dof/urdf/v1.1_u3.0_0303_v0_29dof.urdf",
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
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.24,
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
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 150,
                ".*_hip_roll_joint": 150,
                ".*_hip_yaw_joint": 90,
                ".*_knee_joint": 150,
                ".*_ankle_pitch_joint": 54,
                ".*_ankle_roll_joint": 54,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 14.6608,
                ".*_hip_roll_joint": 14.6608,
                ".*_hip_yaw_joint": 15.1811,
                ".*_knee_joint": 14.6608,
                ".*_ankle_pitch_joint": 9.3201,
                ".*_ankle_roll_joint": 9.3201,
            },
            # stiffness={
            #     ".*_hip_pitch_joint": 120.0,
            #     ".*_hip_roll_joint": 120.0,
            #     ".*_hip_yaw_joint": 70.0,
            #     ".*_knee_joint": 180.0,
            #     ".*_ankle_pitch_joint": 80.0,
            #     ".*_ankle_roll_joint": 80.0,
            # },
            # damping={
            #     ".*_hip_pitch_joint": 8.0,
            #     ".*_hip_roll_joint": 8.0,
            #     ".*_hip_yaw_joint": 4.0,
            #     ".*_knee_joint": 10.0,
            #     ".*_ankle_pitch_joint": 5.0,
            #     ".*_ankle_roll_joint": 5.0,
            # },
            stiffness={
                ".*_hip_pitch_joint": 180.0,
                ".*_hip_roll_joint": 180.0,
                ".*_hip_yaw_joint": 70.0,
                ".*_knee_joint": 180.0,
                ".*_ankle_pitch_joint": 80.0,
                ".*_ankle_roll_joint": 80.0,
            },
            damping={
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_yaw_joint": 4.0,
                ".*_knee_joint": 10.0,
                ".*_ankle_pitch_joint": 5.0,
                ".*_ankle_roll_joint": 5.0,
            },
            friction={
                ".*_hip_pitch_joint": 0.5*5,
                ".*_hip_roll_joint": 0.2*5,
                ".*_hip_yaw_joint": 0.5*5,
                ".*_knee_joint": 0.5*5,
                ".*_ankle_pitch_joint": 0.6*5,
                ".*_ankle_roll_joint": 0.4*5,
            },
            
            dynamic_friction={
                ".*_hip_pitch_joint": 0.5,
                ".*_hip_roll_joint": 0.2,
                ".*_hip_yaw_joint": 0.5,
                ".*_knee_joint": 0.5,
                ".*_ankle_pitch_joint": 0.6,
                ".*_ankle_roll_joint": 0.4,
            },
            armature={
                ".*_hip_pitch_joint": 0.06984,
                ".*_hip_roll_joint": 0.06984,
                ".*_hip_yaw_joint": 0.0486,
                ".*_knee_joint": 0.06984,
                ".*_ankle_pitch_joint": 0.02462,
                ".*_ankle_roll_joint": 0.02462,
                },
        ),
        "waist": ImplicitActuatorCfg(
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
                ".*waist_roll_joint": 250.0,
                ".*waist_pitch_joint": 250.0,
            },
            damping={
                ".*waist_yaw_joint": 6.0,
                ".*waist_roll_joint": 6.0,
                ".*waist_pitch_joint": 6.0,
            },
            friction={
                ".*waist_yaw_joint": 0.1*5,
                ".*waist_roll_joint": 0.1*5,
                ".*waist_pitch_joint": 0.1*5,
            },
            
            dynamic_friction={
                ".*waist_yaw_joint": 0.1,
                ".*waist_roll_joint": 0.1,
                ".*waist_pitch_joint": 0.1,
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
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
                ".*_elbow_joint": 70,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.5,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 1.5,
            },
            friction={
                ".*_shoulder_pitch_joint": 0.1*5,
                ".*_shoulder_roll_joint": 0.1*5,
                ".*_shoulder_yaw_joint": 0.1*5,
                ".*_elbow_joint": 0.1*5,
            },
            
            dynamic_friction={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_joint": 0.1,
            },
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
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
                ".*_wrist_roll_joint": 40.0,
                ".*_wrist_pitch_joint": 40.0,
                ".*_wrist_yaw_joint": 40.0,
            },
            damping=2.0,
            friction={
                ".*_wrist_roll_joint": 0.1*5,
                ".*_wrist_pitch_joint": 0.1*5,
                ".*_wrist_yaw_joint": 0.1*5,
            },
            
            dynamic_friction={
                ".*_wrist_roll_joint": 0.1,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
            armature=0.01,
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
