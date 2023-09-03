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

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        prev_obj_xpos = []
        self._init_states = []

        bar_pos = np.array([1.3, 0.75, self.height_offset])
        bar_x_range = np.array([1.18, 1.42])
        bar_y_range = np.array([0.705, 0.785])

        obj_range1 = np.array([1.2, 0.55])
        obj_range2 = np.array([1.4, 0.95])

        for object_name in self.object_names:
            object_xypos = self.np_random.uniform(low=obj_range1, high=obj_range2)

            while not (
                (np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1)
                and np.all(
                    [
                        np.linalg.norm(object_xypos - other_xpos) >= 0.1
                        for other_xpos in prev_obj_xpos
                    ]
                )
                and not (
                    object_xypos[0] > bar_x_range[0]
                    and object_xypos[0] < bar_x_range[1]
                    and object_xypos[1] > bar_y_range[0]
                    and object_xypos[1] > bar_y_range[1]
                )
            ):
                object_xypos = self.np_random.uniform(low=obj_range1, high=obj_range2)
                # object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                #     low=obj_range1, high=obj_range2
                # )

            prev_obj_xpos.append(object_xypos)

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, f"{object_name}:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos.copy()
            self._utils.set_joint_qpos(
                self.model, self.data, f"{object_name}:joint", object_qpos.copy()
            )
            self._init_states.append(object_qpos.copy())

            self._mujoco.mj_forward(self.model, self.data)
        return True

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

        bar_pos = np.array([1.3, 0.75, self.height_offset])
        bar_x_range = np.array([1.18, 1.42])
        bar_y_range = np.array([0.70, 0.8])

        obj_range1 = np.array([1.15, 0.55])
        obj_range2 = np.array([1.45, 0.95])

        for i in range(0, self.num_blocks):
            goal_object = self.np_random.uniform(low=obj_range1, high=obj_range2)

            while not (
                np.all(
                    [
                        np.linalg.norm(object_pos[:2] - goal_object) > 0.1
                        for object_pos in object_poses
                    ]
                )
                and not (
                    goal_object[0] > bar_x_range[0]
                    and goal_object[0] < bar_x_range[1]
                    and goal_object[1] > bar_y_range[0]
                    and goal_object[1] < bar_y_range[1]
                )
            ):
                goal_object = self.np_random.uniform(low=obj_range1, high=obj_range2)

            goal_object = np.append(goal_object, self.height_offset)
            goals.append(goal_object)

        self.goals = np.concatenate(goals, axis=0).copy()
        return self.goals

    def _get_hold_back_position(self, position1, position2):
        new_pos = position1.copy()
        # get distance between position1 and position2
        dist = np.linalg.norm(position1 - position2, ord=2)
        dist_x = position1[0] - position2[0]
        dist_y = position1[1] - position2[1]
        ratio = 0.06 / dist
        # get deviation x, y of position1
        dx = dist_x * ratio
        dy = dist_y * ratio
        # get hold back position
        new_pos[0] += dx
        new_pos[1] += dy
        return new_pos

    def _get_obstacles_index(self, curr_obj_i):
        curr_obj_pos = self._utils.get_site_xpos(
            self.model, self.data, f"object{curr_obj_i}"
        )
        # get the goal position
        goal_pos = self.goal[curr_obj_i * 3 : curr_obj_i * 3 + 3]
        left = min(curr_obj_pos[0], goal_pos[0])
        right = max(curr_obj_pos[0], goal_pos[0])
        bottom = min(curr_obj_pos[1], goal_pos[1])
        top = max(curr_obj_pos[1], goal_pos[1])
        # get the obstacles object index
        obstacles = []
        # get the obstacles position
        for i in range(len(self.subgoal_finished)):
            if i != curr_obj_i and self.subgoal_finished[i] is False:
                # get the object position
                obj_pos = self._utils.get_site_xpos(self.model, self.data, f"object{i}")
                # check if there are obstacles between obj_pos and goal
                if left < obj_pos[0] < right and bottom < obj_pos[1] < top:
                    # check if obstacle is upper or lower than the line between obj_pos and goal
                    if self._is_above_line(obj_pos, curr_obj_pos, goal_pos):
                        obstacles.append((i, 1))
                    else:
                        obstacles.append((i, -1))
        return obstacles

    def _is_above_line(self, point, line_point1, line_point2):
        # calculate slope of the line
        slope = (line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0])
        # calculate y-intercept of the line
        y_intercept = line_point1[1] - slope * line_point1[0]
        # calculate expected y-value of the point on the line
        expected_y = slope * point[0] + y_intercept
        # compare actual y-value of the point to expected y-value
        if point[1] > expected_y:
            return True
        else:
            return False

    def get_demo_action_(self, obs):
        grip_pos = obs["observation"][:3]

        block_pos = obs["achieved_goal"][:3]

        desired_goal = obs["desired_goal"][:3]

        # check if bar is in the way to desired goal
        bar_pos = np.array([1.3, 0.75, self.height_offset])
        bar_x_range = np.array([1.2, 1.4])
        bar_y_range = np.array([0.725, 0.775])

        if (block_pos[1] - bar_pos[1]) * (desired_goal[1] - bar_pos[1]) < 0:
            # bar is in the way, reset goal to the top/bottom of the bar
            if block_pos[0] < bar_pos[0]:
                # left side
                desired_goal[0] = bar_x_range[0] - 0.05
                desired_goal[1] = bar_pos[1]
            else:
                # right side
                desired_goal[0] = bar_x_range[1] + 0.05
                desired_goal[1] = bar_pos[1]

        # find easiest one to push
        curr_block_pos = block_pos
        curr_goal_pos = desired_goal
        easiest_block = 0

        # check if grip approach the hold-back position
        new_goal_pos = self._get_hold_back_position(
            curr_block_pos.copy(), curr_goal_pos.copy()
        )

        can_reset = False
        # check x-y first
        dist_xy = np.linalg.norm(grip_pos[:2] - new_goal_pos[:2])
        if dist_xy >= 0.025:
            new_goal_pos[2] = 0.6
            action = new_goal_pos - grip_pos
            action = np.append(action, np.array(0.0))

            new_subgoal = obs["achieved_goal"].copy()
            new_subgoal[easiest_block * 3 : easiest_block * 3 + 3] = desired_goal
            return action * 3, [grip_pos, new_goal_pos, can_reset, new_subgoal]

        dist = np.linalg.norm(grip_pos - new_goal_pos)
        if dist < 0.03:
            action = curr_goal_pos - grip_pos
            g = curr_goal_pos
            can_reset = True
        else:
            action = new_goal_pos - grip_pos
            g = new_goal_pos

        # if np.any(action < 0.0002):
        #     times = 10
        # elif np.any(action < 0.002):
        #     times = 5
        # else:
        #     times = 1
        # action = times * action
        action = np.append(action, np.array(0.0))

        new_subgoal = obs["achieved_goal"].copy()
        new_subgoal[easiest_block * 3 : easiest_block * 3 + 3] = desired_goal
        return action * 3, [grip_pos, g, can_reset, new_subgoal]
