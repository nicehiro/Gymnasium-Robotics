import os
import tempfile
from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium_robotics.envs.fetch.fetch_env import get_base_fetch_env
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

from .xml import generate_xml

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


class MultiMujocoFetchEnv(get_base_fetch_env(MujocoRobotEnv)):
    def __init__(
        self,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        num_blocks=3,
        distance_threshold=0.05,
        **kwargs,
    ):
        self.num_blocks = num_blocks
        self.object_names = ["object{}".format(i) for i in range(self.num_blocks)]

        with tempfile.NamedTemporaryFile(
            mode="wt",
            dir=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/assets/fetch/",
            delete=False,
            suffix=".xml",
        ) as fp:
            fp.write(generate_xml(self.num_blocks))
            MODEL_XML_PATH = fp.name

        super().__init__(
            default_camera_config=default_camera_config,
            model_path=MODEL_XML_PATH,
            distance_threshold=distance_threshold,
            **kwargs,
        )

    def _get_obs(self):
        (
            grip_pos,
            object_poses,
            object_rel_poses,
            gripper_state,
            object_rots,
            object_velps,
            object_velrs,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.concatenate(object_poses, axis=0)

        obs = np.concatenate(
            [
                grip_pos, # 3
                np.concatenate(object_poses).ravel(), # 3 * num_blocks
                np.concatenate(object_rel_poses).ravel(), # 3 * num_blocks
                gripper_state, # 2
                np.concatenate(object_rots).ravel(), # 3 * num_blocks
                np.concatenate(object_velps).ravel(), # 3 * num_blocks
                np.concatenate(object_velrs).ravel(), # 3 * num_blocks
                grip_velp, # 3
                gripper_vel, # 2
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(1 * action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )

        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )

        object_poses = []
        object_rots = []
        object_velps = []
        object_velrs = []
        object_rel_poses = []

        for i in range(len(self.object_names)):
            if self.has_object:
                object_pos = self._utils.get_site_xpos(
                    self.model, self.data, self.object_names[i]
                )
                # rotations
                object_rot = rotations.mat2euler(
                    self._utils.get_site_xmat(
                        self.model, self.data, self.object_names[i]
                    )
                )
                # velocities
                object_velp = (
                    self._utils.get_site_xvelp(
                        self.model, self.data, self.object_names[i]
                    )
                    * dt
                )
                object_velr = (
                    self._utils.get_site_xvelr(
                        self.model, self.data, self.object_names[i]
                    )
                    * dt
                )
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp
            else:
                object_pos = (
                    object_rot
                ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

            object_poses.append(object_pos)
            object_rots.append(object_rot)
            object_velps.append(object_velp)
            object_velrs.append(object_velr)
            object_rel_poses.append(object_rel_pos)
        gripper_state = robot_qpos[-2:]

        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        return (
            grip_pos,
            object_poses,
            object_rel_poses,
            gripper_state,
            object_rots,
            object_velps,
            object_velrs,
            grip_velp,
            gripper_vel,
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        for i in range(self.num_blocks):
            site_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target{}".format(i)
            )
            self.model.site_pos[site_id] = self.goal[i*3:(i+1)*3] - sites_offset[i]
        self._mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        if options is not None and options["fixed"] == True:
            self._reset_given_sim(options["init_pos"], options["goal_pos"])
        else:
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()
            self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}

    def _reset_given_sim(self, init_poses: List, goal_poses: List):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # reset fix object pos
        for i in range(len(self.object_names)):
            object_name = self.object_names[i]
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, f"{object_name}:joint"
            )
            object_qpos[:2] = init_poses[i][:2]
            self._utils.set_joint_qpos(
                self.model, self.data, f"{object_name}:joint", object_qpos.copy()
            )
            self._mujoco.mj_forward(self.model, self.data)

        # reset fix goal pos
        self.goal = np.concatenate(goal_poses)
        return True

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        prev_obj_xpos = []
        self._init_states = []

        for object_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )

            while not (
                (np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1)
                and np.all(
                    [
                        np.linalg.norm(object_xypos - other_xpos) >= 0.1
                        for other_xpos in prev_obj_xpos
                    ]
                )
            ):
                object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )

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

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]

    def get_demo_action(self):
        pass

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return [
            np.linalg.norm(
                goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3],
                axis=-1,
            )
            for i in range(self.num_blocks)
        ]

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        if (
            np.sum(
                [
                    -(d > self.distance_threshold).astype(np.float32)
                    for d in subgoal_distances
                ]
            )
            == 0
        ):
            return True
        else:
            return False


if __name__ == "__main__":
    env = gym.make("FetchPickAndPlace-v3", render_mode="human")
    env.reset()

    while True:
        action = env.get_demo_action()
        env.step(action)

        env.render()
