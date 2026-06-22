import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from envs.fetchreach_env import make_env


def record_agent(policy, seed, num_episodes, video_folder,
                 policy_name="policy", attack_fn=None, attack_name="clean",
                 on_reset=None, action_fn=None):
    """Record episodes of a policy, optionally under attack.

    Args:
        policy:       Any object with .predict(obs, deterministic=True) -> (action, _).
        seed:         Seed passed to make_env — controls the episode variety across resets.
        num_episodes: Number of episodes to run and record.
        video_folder: Directory where .mp4 files are written.
        policy_name:  Human-readable policy label used in video filenames.
        attack_fn:    Optional callable (obs) -> obs applied between env and policy.
                      Pass None for clean (no attack) episodes.
        attack_name:  Human-readable attack label used in video filenames.
        on_reset:     Optional callable () -> None, called at the start of each episode.
                      Used to reset closure state (e.g. bias_vector, step counter)
                      that lives outside this function.
        action_fn:    Optional callable (action) -> action applied after policy.predict()
                      and before env.step(). Used for action-space attacks (delay, reverse,
                      clipping) that intercept the command rather than the observation.
    """
    env = make_env(seed=seed, rgb_mode=True)
    env = RecordEpisodeStatistics(env, buffer_length=num_episodes)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"{policy_name}_seed{seed}_{attack_name}",
        episode_trigger=lambda x: True,   # record every episode
    )

    results = []

    for episode in range(num_episodes):
        # Reset closure state (bias vector, step counter) before each episode.
        # on_reset reaches back into __main__ scope via the closure — record_agent
        # itself doesn't own or know about that state.
        if on_reset:
            on_reset()

        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Observation attack: modifies what the policy sees.
            policy_obs = attack_fn(obs) if attack_fn else obs
            action, _ = policy.predict(policy_obs, deterministic=True)
            # Action attack: modifies the command sent to the environment.
            if action_fn:
                action = action_fn(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        success = info.get("is_success", 0.0)
        print(f"  Episode {episode}: reward={total_reward:.1f}, success={bool(success)}")
        results.append(success)

    env.close()
    print(f"\nRecorded {num_episodes} episodes | Success rate: {np.mean(results):.1%}")


if __name__ == "__main__":
    from stable_baselines3 import SAC
    from policies.sac_her_policy import SACHerPolicy
    from attacks.sensor_attacks import apply_sensor_dropout, apply_sensor_bias, shift_target
    from attacks.action_attacks import manipulate_action

    env_for_load = make_env(seed=0)
    model = SAC.load("tairo_results/models/sac_her_fetchreach_model", env=env_for_load)
    policy = SACHerPolicy(model)

    def make_sensor_attack_fns():
        """Factory for observation/goal attacks — owns bias_vector, goal_offset, current_step."""
        bias_vector = None
        goal_offset = None
        current_step = 0

        def reset_state():
            nonlocal bias_vector, goal_offset, current_step
            bias_vector = None
            goal_offset = None
            current_step = 0

        def sensor_bias_attack(obs):
            nonlocal bias_vector
            obs, bias_vector = apply_sensor_bias(obs, magnitude=0.1, bias_vector=bias_vector)
            return obs

        def goal_spoof_immediate_attack(obs):
            nonlocal current_step, goal_offset
            obs, goal_offset = shift_target(
                obs, shift_scale=0.1, step=current_step,
                shift_step=None, goal_offset=goal_offset,
            )
            current_step += 1
            return obs

        def goal_spoof_midep_attack(obs):
            nonlocal current_step, goal_offset
            obs, goal_offset = shift_target(
                obs, shift_scale=0.1, step=current_step,
                shift_step=20, goal_offset=goal_offset,
            )
            current_step += 1
            return obs

        return sensor_bias_attack, goal_spoof_immediate_attack, goal_spoof_midep_attack, reset_state

    (sensor_bias_attack,
     goal_spoof_immediate_attack,
     goal_spoof_midep_attack,
     reset_sensor_state) = make_sensor_attack_fns()

    def make_action_attack_fns():
        """Factory for action-space attacks — owns previous_action for delay replay."""
        previous_action = None

        def reset_action_state():
            nonlocal previous_action
            previous_action = None

        def action_delay_attack(action):
            nonlocal previous_action
            executed = manipulate_action(
                action, attack_type="action_delay", previous_action=previous_action,
            )
            previous_action = action.copy()
            return executed

        def action_reverse_attack(action):
            return manipulate_action(action, attack_type="action_reverse")

        def action_clip_attack(action):
            return manipulate_action(action, attack_type="action_clipping", clip_value=0.3)

        return action_delay_attack, action_reverse_attack, action_clip_attack, reset_action_state

    (action_delay_attack,
     action_reverse_attack,
     action_clip_attack,
     reset_action_state) = make_action_attack_fns()

    # --- Conditions to record ---
    # Each entry: (attack_name, obs_attack_fn, action_attack_fn, on_reset)
    conditions = [
        ("clean",                 None,                        None,                 None),
        ("sensor_dropout",        lambda obs: apply_sensor_dropout(obs, fields=["observation"]),
                                                               None,                 None),
        ("sensor_bias",           sensor_bias_attack,          None,                 reset_sensor_state),
        ("goal_spoof_immediate",  goal_spoof_immediate_attack, None,                 reset_sensor_state),
        ("goal_spoof_midep",      goal_spoof_midep_attack,     None,                 reset_sensor_state),
        ("action_delay",          None,                        action_delay_attack,  reset_action_state),
        ("action_reverse",        None,                        action_reverse_attack, None),
        ("action_clipping",       None,                        action_clip_attack,   None),
    ]

    for attack_name, attack_fn, action_fn, on_reset in conditions:
        print(f"\n{'='*40}")
        print(f"Condition: {attack_name}")
        print(f"{'='*40}")
        record_agent(
            policy=policy,
            seed=0,
            num_episodes=10,
            video_folder=f"tairo_results/videos/{attack_name}",
            policy_name="sac_her",
            attack_fn=attack_fn,
            action_fn=action_fn,
            attack_name=attack_name,
            on_reset=on_reset,
        )

