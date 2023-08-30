import gymnasium as gym


env = gym.make(
    # "FetchPush-v4",
    "FetchPickAndPlace-v4",
    # "FetchPickAndPlaceTest-v2",
    # "FetchPushTest-v2",
    # "FetchPushOverBar-v1",
    render_mode="human",
    max_episode_steps=300,
    # goal_level_prob=[0.2, 0.2, 0.2, 0.2, 0.2],
    # goal_level_prob=[0, 1, 0., 0],
)

obs, _ = env.reset()
t = 0

while True:
    # action = env.action_space.sample()
    # action, info = env.get_demo_action_(obs)
    action, info = env.get_demo_action()

    if t % 30 == 0:
        _, _, _, subgoal = info
        env.update_subgoal(subgoal)

    # action, _ = env.get_demo_action()
    obs, reward, done, terminal, info = env.step(action)
    env.render()
    t += 1
    if done or terminal:
        obs, _ = env.reset()
        t = 0
