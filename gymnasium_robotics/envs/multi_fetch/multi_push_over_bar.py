from enum import Enum
import os
import tempfile

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch.fetch_env import get_base_fetch_env
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv

from .multi_fetch import MultiMujocoFetchEnv
from .xml import generate_xml_with_bar
from gymnasium_robotics.utils import rotations


class MultiMujocoFetchPushOverBarEnv(MultiMujocoFetchEnv, EzPickle):

    def __init__(
        self,
        reward_type="sparse",
        num_blocks=1,
        **kwargs,
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        for i in range(num_blocks):
            initial_qpos[f"object{i}:joint"] = [
                1.25,
                0.53,
                0.45,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        MultiMujocoFetchEnv.__init__(
            self,
            num_blocks=num_blocks,
            with_bar=True,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.1,
            target_range=0.1,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        goals = []

        init_grip_xpos = self.initial_gripper_xpos[:2]

        object_poses = []
        for i in range(0, self.num_blocks):
            object_pos = self._utils.get_site_xpos(
                self.model, self.data, self.object_names[i]
            )
            object_pos[2] = self.height_offset
            object_poses.append(object_pos)

        bar_pos = self._utils.get_site_xpos(self.model, self.data, "bar")

        for i in range(0, self.num_blocks):
            goal_object = init_grip_xpos + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )

            while not np.all(
                [
                    np.linalg.norm(object_pos[:2] - init_grip_xpos) > 0.1
                    for object_pos in object_poses
                ] +
                [
                    np.linalg.norm(bar_pos[:2] - init_grip_xpos) > 0.1
                ]
            ):
                goal_object = init_grip_xpos + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )

            goal_object = np.append(goal_object, self.height_offset)
            goals.append(goal_object)

        self.goals = np.concatenate(goals, axis=0).copy()
        return self.goals

    def _reset_sim(self):
        # reset bar
        bar_xypos = np.array([1.35, 0.75])
        bar_qpos = self._utils.get_joint_qpos(
            self.model, self.data, f"bar:joint"
        )
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xypos.copy()
        self._utils.set_joint_qpos(
            self.model, self.data, f"bar:joint", object_qpos.copy()
        )
        self._mujoco.mj_forward(self.model, self.data)

        # reset block to left side
        super()._reset_sim()

        return True