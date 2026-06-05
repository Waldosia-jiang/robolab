# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry transforms for v1_1_29dof AMP training."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

JOINT_NUM = 29

# Isaac Lab joint order for v1_1_29dof (29 actuated joints).
LEFT_JOINT_IDX = [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
RIGHT_JOINT_IDX = [1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
JOINT_SIGN_FLIP_IDX = [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26]

# policy: ang_vel(3) + gravity(3) + cmd(3) + joint_pos(29) + joint_vel(29) + action(29) = 96
POLICY_JOINT_POS_SLICE = slice(9, 9 + JOINT_NUM)
POLICY_JOINT_VEL_SLICE = slice(9 + JOINT_NUM, 9 + 2 * JOINT_NUM)
POLICY_ACTION_SLICE = slice(9 + 2 * JOINT_NUM, 9 + 3 * JOINT_NUM)

# critic single frame: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + joint_pos(29) + joint_vel(29) + action(29) = 99
CRITIC_JOINT_POS_SLICE = slice(12, 12 + JOINT_NUM)
CRITIC_JOINT_VEL_SLICE = slice(12 + JOINT_NUM, 12 + 2 * JOINT_NUM)
CRITIC_ACTION_SLICE = slice(12 + 2 * JOINT_NUM, 12 + 3 * JOINT_NUM)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions with left-right symmetry."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    obs[:, POLICY_JOINT_POS_SLICE] = _switch_joints_left_right(obs[:, POLICY_JOINT_POS_SLICE])
    obs[:, POLICY_JOINT_VEL_SLICE] = _switch_joints_left_right(obs[:, POLICY_JOINT_VEL_SLICE])
    obs[:, POLICY_ACTION_SLICE] = _switch_joints_left_right(obs[:, POLICY_ACTION_SLICE])
    return obs


def _transform_critic_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    obs[:, CRITIC_JOINT_POS_SLICE] = _switch_joints_left_right(obs[:, CRITIC_JOINT_POS_SLICE])
    obs[:, CRITIC_JOINT_VEL_SLICE] = _switch_joints_left_right(obs[:, CRITIC_JOINT_VEL_SLICE])
    obs[:, CRITIC_ACTION_SLICE] = _switch_joints_left_right(obs[:, CRITIC_ACTION_SLICE])
    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_switched = joint_data.clone()
    joint_data_switched[..., LEFT_JOINT_IDX] = joint_data[..., RIGHT_JOINT_IDX]
    joint_data_switched[..., RIGHT_JOINT_IDX] = joint_data[..., LEFT_JOINT_IDX]
    joint_data_switched[..., JOINT_SIGN_FLIP_IDX] *= -1.0
    return joint_data_switched
