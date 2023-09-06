from enum import Enum

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from .multi_fetch import MultiMujocoFetchEnv


class Task(Enum):
    ApproachObj = 1
    # ApproachObjY = 2
    HoldBack = 3
    Push = 4
    # PushY = 5
    Reset = 6


class MultiMujocoFetchPushEnv(MultiMujocoFetchEnv, EzPickle):
    def __init__(
        self,
        reward_type="sparse",
        num_blocks=4,
        distance_threshold=0.05,
        goal_level_prob=[0, 0, 0, 0, 1, 0],
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
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.1,
            target_range=0.1,
            distance_threshold=distance_threshold,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
        self.goal_level_prob = goal_level_prob

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


        goal_object0 = init_grip_xpos + np.array([0.1, 0.25])
        goal_object0 = np.append(goal_object0, self.height_offset)
        goal_object1 = init_grip_xpos + np.array([-0.2, 0.25])
        goal_object1 = np.append(goal_object1, self.height_offset)
        goal_object2 = init_grip_xpos + np.array([0.1, -0.25])
        goal_object2 = np.append(goal_object2, self.height_offset)
        goal_object3 = init_grip_xpos + np.array([-0.2, -0.25])
        goal_object3 = np.append(goal_object3, self.height_offset)

        if self.goal_level == 1:
            goals = [goal_object0, object_poses[1], object_poses[2], object_poses[3]]
        elif self.goal_level == 2:
            goals = [object_poses[0], goal_object1, object_poses[2], object_poses[3]]
        elif self.goal_level == 3:
            goals = [object_poses[0], object_poses[1], goal_object2, object_poses[3]]
        elif self.goal_level == 4:
            goals = [object_poses[0], object_poses[1], object_poses[2], goal_object3]
        elif self.goal_level == 5:
            goals = [goal_object0, goal_object1, goal_object2, goal_object3]
        elif self.goal_level == 6:
            goal_objects = []

            for i in range(0, self.num_blocks):
                goal_object = init_grip_xpos + self.np_random.uniform(
                    np.array([-0.2, -0.25]), np.array([0.1, 0.25]), size=2
                )

                while not np.all(
                    [
                        np.linalg.norm(goal_object - obj_pos[:2]) >= 0.05
                        for obj_pos in self._init_states
                    ] +
                    [
                        np.linalg.norm(goal_object - goal_pos[:2]) >= 0.1
                        for goal_pos in goal_objects
                    ]
                ):
                    goal_object = init_grip_xpos + self.np_random.uniform(
                        np.array([-0.2, -0.25]), np.array([0.1, 0.25]), size=2
                    )

                goal_object = np.append(goal_object, self.height_offset)
                goal_objects.append(goal_object)

            # goal_object0, goal_object1, goal_object2, goal_object3 = goal_objects
            goals = goal_objects

        self.goals = np.concatenate(goals, axis=0).copy()
        return self.goals

    def reset(self, seed=None, options=None):
        self.push = [
            Task.Reset,
            Task.ApproachObj,
            Task.HoldBack,
            Task.Push,
            Task.Reset,
        ]

        self.push_times = 0
        self.relex_times = 0
        self.hold_times = 0

        # obstacles list
        self.obstacles = [[]] * self.num_blocks
        self.ranks = [0] * self.num_blocks

        total_levels = [1, 2, 3, 4, 5, 6]

        self.goal_level = np.random.choice(
            total_levels, p=self.goal_level_prob, size=1
        )[0]

        obs, info = super().reset(seed=seed, options=options)

        # subgoal finished or not
        self.subgoal_finished = [False] * self.num_blocks
        # current work queue
        # [obj_i, desired_subgoal, push, push, push, push, push]
        self.work_queue = []

        return obs, info

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
        reward = -np.sum(
            [
                (d > self.distance_threshold).astype(np.float32)
                for d in subgoal_distances
            ],
            axis=0,
        )
        reward = np.asarray(reward)

        # If blocks are successfully aligned with goals, add a bonus for the gripper being away from the goals
        # np.putmask(
        #     reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal)
        # )
        return reward

    def get_demo_action(self):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        if all(self.subgoal_finished):
            reset_pos = grip_pos.copy()
            reset_pos[2] = 0.6
            dist = reset_pos - grip_pos
            action = np.append(dist, np.array(0.0))
            return action, [reset_pos, reset_pos]

        if len(self.work_queue) == 0:
            # get obstacles index list for each block
            obstacles = [[]] * self.num_blocks
            ranks = [0] * self.num_blocks
            for i in range(self.num_blocks):
                if self.subgoal_finished[i]:
                    obstacles[i] = [(1, 1)] * 10
                    ranks[i] = len(obstacles[i])
                    continue
                obstacles[i] = self._get_obstacles_index(i).copy()
                ranks[i] = len(obstacles[i])
            # rank the lenght of obstacles list
            t = sorted(ranks)
            easiest_block = ranks.index(t[0])
            # add easiest subgoal to work queue
            self.work_queue.append(
                [easiest_block, self.goal[easiest_block * 3 : easiest_block * 3 + 3]]
                + [False] * len(self.push)
            )
            # add obstacles clean subgoal to work queue
            for j, sign in obstacles[easiest_block]:
                obstacle_pos = self._utils.get_site_xpos(
                    self.model, self.data, f"object{j}"
                )
                # move obstacle to vertical position of the line between object pos and goal
                obstacle_goal = obstacle_pos.copy()
                obstacle_goal[0] += sign * 0.05
                obstacle_goal[1] += sign * 0.05
                self.work_queue.append([j, obstacle_goal] + [False] * len(self.push))

        # do the work queue, finish then pop
        demo_distance_threshold = 0.04
        # current subgoal: [obj_i, desired_subgoal, push, push, push, push, push]
        current_subgoal = self.work_queue[-1]
        current_task = None
        for i in range(len(self.push)):
            if current_subgoal[i + 2] is False:
                current_task = i
                break
        if current_task is None:
            self.work_queue.pop(-1)
            if len(self.work_queue) == 0:
                return self.last_action
            current_subgoal = self.subgoal_finished[-1]
            current_task = 0

        obj_i = current_subgoal[0]
        obj_pos = self._utils.get_site_xpos(self.model, self.data, f"object{obj_i}")
        desired_subgoal = current_subgoal[1]

        dist = 0.0
        if self.push[current_task] == Task.ApproachObj:
            # approach obj, ignore z
            new_obj_pos = self._get_hold_back_position(
                obj_pos.copy(), desired_subgoal.copy()
            )
            new_obj_pos[2] = grip_pos[2]
            dist = new_obj_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                current_subgoal[current_task + 2] = True
                # print("approach obj")
        elif self.push[current_task] == Task.HoldBack:
            # hold object's back
            hold_back_pos = grip_pos.copy()
            hold_back_pos[2] = obj_pos[2]
            dist = hold_back_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                current_subgoal[current_task + 2] = True
                # print("hold back")
        elif self.push[current_task] == Task.Push:
            # push to goal
            new_obj_pos = obj_pos.copy()
            dist = desired_subgoal - new_obj_pos
            distance = np.linalg.norm(dist)
            self.push_times += 1
            if self.push_times > 1:
                self.push_times = 0
                # print("push")
                if distance < demo_distance_threshold:
                    current_subgoal[current_task + 2] = True
                    self.work_queue.pop(-1)
                    if len(self.work_queue) == 0:
                        self.subgoal_finished[obj_i] = True
                    # print("update queue")
                else:
                    for j in range(len(self.push)):
                        current_subgoal[j + 2] = False
        elif self.push[current_task] == Task.Reset:
            # reset, lift z position of gripper, with gripper close
            reset_pos = grip_pos.copy()
            reset_pos[2] = 0.6
            dist = reset_pos - grip_pos
            distance = np.linalg.norm(dist)
            if distance < demo_distance_threshold:
                self.relex_times += 1
                if self.relex_times > 5:
                    current_subgoal[current_task + 2] = True
                    self.relex_times = 0
                    # print("reset")

        # print("dist: ", dist)
        dist = np.clip(dist, a_min=-0.1, a_max=0.1)
        # print("dist: ", dist)
        action = np.append(dist, np.array(0.0))
        self.last_action = action
        return action * 5, [obj_pos, desired_subgoal]

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

        # find easiest one to push
        obstacles = [[]] * self.num_blocks
        ranks = [0] * self.num_blocks
        for i in range(self.num_blocks):
            block_pos = obs["achieved_goal"][i * 3 : i * 3 + 3]
            goal_pos = obs["desired_goal"][i * 3 : i * 3 + 3]
            dist = np.linalg.norm(block_pos - goal_pos)
            if dist < 0.03:
                obstacles[i] = [(1, 1)] * 10
                ranks[i] = len(obstacles[i])
                continue
            obstacles[i] = self._get_obstacles_index(i).copy()
            ranks[i] = len(obstacles[i])
        # rank the lenght of obstacles list
        t = sorted(ranks)
        easiest_block = ranks.index(t[0])

        curr_block_pos = obs["achieved_goal"][easiest_block * 3 : easiest_block * 3 + 3]
        curr_goal_pos = obs["desired_goal"][easiest_block * 3 : easiest_block * 3 + 3]

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
            new_subgoal[easiest_block * 3 : easiest_block * 3 + 3] = obs[
                "desired_goal"
            ][easiest_block * 3 : easiest_block * 3 + 3]
            return action, [grip_pos, new_goal_pos, can_reset, new_subgoal]

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
        new_subgoal[easiest_block * 3 : easiest_block * 3 + 3] = obs["desired_goal"][
            easiest_block * 3 : easiest_block * 3 + 3
        ]
        return action * 3, [grip_pos, g, can_reset, new_subgoal]


if __name__ == "__main__":
    env = MultiMujocoFetchPushEnv()
    env.reset()
    for _ in range(1000):
        # env.render()
        action = env.get_demo_action()
        env.step(action)
