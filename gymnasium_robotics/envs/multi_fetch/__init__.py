from gymnasium.envs.registration import register


for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    # Pick and Place
    register(
        id=f"FetchPickAndPlace{suffix}-v3",
        entry_point="multi_block_fetch.multi_pnp:MultiMujocoFetchPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=300,
    )

    # Push
    register(
        id=f"FetchPush{suffix}-v3",
        entry_point="multi_block_fetch.multi_push:MultiMujocoFetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=1500,
    )
