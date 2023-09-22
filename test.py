import gymnasium as gym


env = gym.make(
    "FetchPush-v4",
    # "FetchPickAndPlace-v4",
    # "FetchPickAndPlaceTest-v2",
    # "FetchPushTest-v2",
    # "FetchPushOverBar-v1",
    # "AntMaze_UMaze_Diverse_GR-v3",
    render_mode="human",
    max_episode_steps=300,
    # goal_level_prob=[0, 0, 0, 0, 0, 1],
    # goal_level_prob=[0, 1, 0., 0],
)

options = {
    "fixed": True,
    "init_pos": [],
    "goal_pos": []
}

obs, _ = env.reset()
t = 0

while True:
    # action = env.action_space.sample()
    action, info = env.get_demo_action_(obs)
    # action, info = env.get_demo_action()

    if t % 30 == 0:
        _, _, _, subgoal = info
        env.update_subgoal(subgoal)

    # env.update_subgoal(obs['desired_goal'])

    # action, _ = env.get_demo_action()
    obs, reward, done, terminal, info = env.step(action)
    env.render()
    t += 1
    if done or terminal:
        obs, _ = env.reset()
        t = 0
