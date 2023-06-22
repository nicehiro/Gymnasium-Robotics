import os
from enum import Enum
import numpy as np

from gymnasium.utils.ezpickle import EzPickle

from .multi_fetch import MultiMujocoFetchEnv


class Task(Enum):
    Reset = 0
    ApproachObj = 1
    Pick = 2
    Place = 3
    ApproachGoal = 4
    ResetRelex = 5


class MultiMujocoFetchPickAndPlaceEnv(MultiMujocoFetchEnv, EzPickle):
    def __init__(
        self, reward_type="sparse", num_blocks=3, goal_level_prob=[0, 0, 0, 1], **kwargs
    ):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.25, 0.53, 0.5, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.25, 0.53, 0.6, 1.0, 0.0, 0.0, 0.0],
        }
        for i in range(num_blocks):
            initial_qpos[f"object{i}:joint"] = [
                1.25,
                0.53,
                0.4 + i * 0.1,
                1.0,
                0.0,
                0.0,
                0.0,
            ]

        MultiMujocoFetchEnv.__init__(
            self,
            num_blocks=num_blocks,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        self.goal_level_prob = goal_level_prob

    def _sample_level_4_goal(self):
        goals = []

        goal_object0 = self.initial_gripper_xpos[:2] + self.np_random.uniform(
            -self.target_range, self.target_range, size=2
        )

        while not np.all(
            [
                np.linalg.norm(goal_object0 - obj_pos[:2]) >= 0.1
                for obj_pos in self._init_states
            ]
        ):
            goal_object0 = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.target_range, self.target_range, size=2
            )

        goal_object0 = np.append(goal_object0, self.height_offset)

        # Start off goals array with the first block
        goals.append(goal_object0)

        # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
        previous_xys = [goal_object0[:2]]
        current_tower_heights = [goal_object0[2]]

        num_configured_blocks = self.num_blocks - 1

        for i in range(num_configured_blocks):
            goal_objecti = goal_object0[:2]
            objecti_xy = goal_objecti

            # Check if any of current block xy matches any previous xy's
            for _ in range(len(previous_xys)):
                previous_xy = previous_xys[_]
                if np.linalg.norm(previous_xy - objecti_xy) < 0.071:
                    goal_objecti = previous_xy

                    new_height_offset = current_tower_heights[_] + 0.05
                    current_tower_heights[_] = new_height_offset
                    goal_objecti = np.append(goal_objecti, new_height_offset)

            # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
            if len(goal_objecti) == 2:
                goal_objecti = np.append(goal_objecti, self.height_offset)
                previous_xys.append(objecti_xy)
                current_tower_heights.append(self.height_offset)

            goals.append(goal_objecti)

        return goals

    def _sample_goal(self):
        # sample easy goal for collect data and pretrain
        # only need to move one or two blocks each time
        # level: 1: move first block; 2: move second block over first block
        # level: 3: move third block over first two blocks
        # level: 4: move three blocks
        assert self.goal_level in [1, 2, 3, 4]

        object_poses = []
        for i in range(0, self.num_blocks):
            object_pos = self._utils.get_site_xpos(
                self.model, self.data, self.object_names[i]
            )
            object_pos[2] = self.height_offset
            object_poses.append(object_pos)

        if self.goal_level == 1:
            goals = []
            goal_object0 = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.target_range, self.target_range, size=2
            )
            while not np.all(
                [
                    np.linalg.norm(goal_object0 - obj_pos[:2]) >= 0.1
                    for obj_pos in self._init_states
                ]
            ):
                goal_object0 = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=2
                )
            goal_object0 = np.append(goal_object0, self.height_offset)
            # Start off goals array with the first block
            goals.append(goal_object0)
            # keep others at initial position
            for i in range(1, self.num_blocks):
                goals.append(object_poses[i])
        elif self.goal_level == 2:
            goal_object1 = np.append(object_poses[0][:2], self.height_offset + 0.05)
            goals = [object_poses[0], goal_object1, object_poses[2]]
        elif self.goal_level == 3:
            goal_object1 = np.append(object_poses[0][:2], self.height_offset + 0.05)
            goal_object2 = np.append(object_poses[0][:2], self.height_offset + 0.1)
            goals = [object_poses[0], goal_object1, goal_object2]
        elif self.goal_level == 4:
            goals = self._sample_level_4_goal()
        return np.concatenate(goals, axis=0).copy()

    def reset(self, seed=None, options=None):
        # check subgoal finished sequatially
        self.pick_and_place = [
            Task.ApproachObj,
            Task.Pick,
            Task.Reset,
            Task.ApproachGoal,
            Task.Place,
            Task.ResetRelex,
        ]
        self.subgoal_finished = [
            [False for _ in range(len(self.pick_and_place))]
            for _ in range(self.num_blocks)
        ]
        self.hold_times = 0
        self.relex_times = 0

        total_levels = [1, 2, 3, 4]

        self.goal_level = np.random.choice(
            total_levels, p=self.goal_level_prob, size=1
        )[0]

        obs = super().reset(seed=seed, options=options)
        return obs

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = self._utils.get_joint_qpos(self.model, self.data, "robot0:grip")[
            :3
        ]

        # Get all the goals EXCEPT the zero'd out grip position
        block_goals = goal

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i * 3 : (i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def compute_reward(self, achieved_goal, goal, info):
        subgoal_distances = self.subgoal_distances(achieved_goal, goal)
        # Using incremental reward for each block in correct position
        reward = (
            -np.sum(
                [
                    (d > self.distance_threshold).astype(np.float32)
                    for d in subgoal_distances
                ],
                axis=0,
            )
            / 3
        )
        reward = np.asarray(reward)

        # If blocks are successfully aligned with goals, add a bonus for the gripper being away from the goals
        np.putmask(
            reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal)
        )
        return reward

    def get_demo_action(self):
        demo_distance_threshold = 0.02
        num_task = len(self.pick_and_place)
        i, j = 0, 0
        for i in range(self.num_blocks):
            d = False
            for j in range(num_task):
                if not self.subgoal_finished[i][j]:
                    d = True
                    break
            if d:
                break
        current_subgoal = i
        current_task = j

        if current_subgoal >= self.num_blocks:
            return self.last_action

        obs = self._get_obs()
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        obj_pos = self._utils.get_site_xpos(
            self.model, self.data, self.object_names[current_subgoal]
        )

        desired_goal = obs["desired_goal"]
        desired_subgoal = desired_goal[current_subgoal * 3 : current_subgoal * 3 + 3]

        dist, grasp_ctrl = 0.0, 0.0
        if self.pick_and_place[current_task] == Task.ApproachObj:
            # approach obj, ignore z
            new_obj_pos = obj_pos.copy()
            new_obj_pos[2] = grip_pos[2]
            dist = new_obj_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                self.subgoal_finished[current_subgoal][current_task] = True
            grasp_ctrl = 1.0
        elif self.pick_and_place[current_task] == Task.Pick:
            # move to obj and take it
            dist = obj_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                grasp_ctrl = -1.0
                self.hold_times += 1
                if self.hold_times > 5:
                    self.subgoal_finished[current_subgoal][current_task] = True
            else:
                grasp_ctrl = 1.0
        elif self.pick_and_place[current_task] == Task.Reset:
            # reset, lift z position of gripper, with gripper close
            reset_pos = obj_pos.copy()
            reset_pos[2] = 0.6
            dist = reset_pos - grip_pos
            grasp_ctrl = -1.0
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                self.subgoal_finished[current_subgoal][current_task] = True
        elif self.pick_and_place[current_task] == Task.ApproachGoal:
            new_goal_pos = desired_subgoal.copy()
            new_goal_pos[2] = grip_pos[2]
            dist = new_goal_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                self.subgoal_finished[current_subgoal][current_task] = True
            grasp_ctrl = -1.0
        elif self.pick_and_place[current_task] == Task.Place:
            dist = desired_subgoal - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                self.subgoal_finished[current_subgoal][current_task] = True
                grasp_ctrl = 1.0
            grasp_ctrl = -1.0
        elif self.pick_and_place[current_task] == Task.ResetRelex:
            # reset, lift z position of gripper, with gripper close
            reset_pos = desired_subgoal.copy()
            reset_pos[2] += 0.2
            dist = reset_pos - grip_pos
            grasp_ctrl = 1.0
            if self.relex_times < 5:
                self.relex_times += 1
            else:
                self.subgoal_finished[current_subgoal][current_task] = True
                # subgoal finished, reset hold_times and relex_times
                self.hold_times = 0
                self.relex_times = 0

        action = np.append(dist, np.array(grasp_ctrl))
        self.last_action = action
        return action * 5, []
