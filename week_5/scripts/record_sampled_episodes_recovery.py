"""
Post-hoc video pass — every 10th episode with recovery enabled.

Mirrors record_sampled_episodes.py exactly, adding TAIRO C5 recovery.
Videos saved to tairo_results/videos/with_recovery/<condition>/episode_<N>.mp4.

RNG matching guarantee
----------------------
Uses the same seed logic as run_failure_analysis_recovery.py: one env per condition,
100 sequential env.reset() calls, throwaway loops for non-sampled episodes. Recovery
state (prev_obs, prev_action, step_distances) is also tracked during throwaway episodes
so that sampled episode N is in the identical system state as in the CSV.
Attack closure state is reset between every episode via on_reset, matching the sweep.
"""

import os
import sys

import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC

from config import MODELS_DIR
from envs.fetchreach_env import make_env, distance_to_goal
from policies.sac_her_policy import SACHerPolicy
from attacks.sensor_attacks import apply_sensor_dropout, apply_sensor_bias, shift_target
from attacks.action_attacks import manipulate_action
from recovery.recovery_logic import maybe_apply_recovery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED            = 0
N_EPISODES      = 100
SAMPLE_EPISODES = set(range(0, 100, 10))
MODEL_PATH      = os.path.join(MODELS_DIR, "sac_her_fetchreach_model")
VIDEO_ROOT      = os.path.join("tairo_results", "videos", "with_recovery")
FPS             = 30

# ---------------------------------------------------------------------------
# Attack factories — identical to run_failure_analysis_recovery.py
# ---------------------------------------------------------------------------

def make_sensor_attack_fns():
    bias_vector = None
    goal_offset = None
    current_step = 0

    def reset():
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

    return sensor_bias_attack, goal_spoof_immediate_attack, goal_spoof_midep_attack, reset


def make_action_attack_fns():
    previous_action = None

    def reset():
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

    return action_delay_attack, action_reverse_attack, action_clip_attack, reset


# ---------------------------------------------------------------------------
# Single-episode runner with recovery; returns frames list (empty if not sampled)
# ---------------------------------------------------------------------------

def run_episode(env, policy, attack_fn, action_fn, capture_frames):
    obs, _ = env.reset()
    done = False
    frames = []

    prev_obs = None
    prev_action = None
    step_distances = []
    step = 0

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while not done:
        policy_obs = attack_fn(obs) if attack_fn else obs
        action, _ = policy.predict(policy_obs, deterministic=True)
        if action_fn:
            action = action_fn(action)

        final_action, _ = maybe_apply_recovery(
            obs=obs,
            action=action,
            prev_obs=prev_obs,
            prev_action=prev_action,
            step_distances=step_distances,
            step=step,
            env=env,
        )

        prev_obs = obs
        prev_action = final_action.copy()

        obs, _, terminated, truncated, _ = env.step(final_action)
        done = terminated or truncated

        step_distances.append(distance_to_goal(obs))
        step += 1

        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_for_load = make_env(seed=SEED)
    model = SAC.load(MODEL_PATH, env=env_for_load)
    policy = SACHerPolicy(model)

    (sensor_bias_attack,
     goal_spoof_immediate_attack,
     goal_spoof_midep_attack,
     reset_sensor) = make_sensor_attack_fns()

    (action_delay_attack,
     action_reverse_attack,
     action_clip_attack,
     reset_action) = make_action_attack_fns()

    conditions = [
        ("clean",                None,                        None,                  None),
        ("sensor_dropout",       lambda obs: apply_sensor_dropout(obs, fields=["observation"]),
                                                              None,                  None),
        ("sensor_bias",          sensor_bias_attack,          None,                  reset_sensor),
        ("goal_spoof_immediate", goal_spoof_immediate_attack, None,                  reset_sensor),
        ("goal_spoof_midep",     goal_spoof_midep_attack,     None,                  reset_sensor),
        ("action_delay",         None,                        action_delay_attack,   reset_action),
        ("action_reverse",       None,                        action_reverse_attack, None),
        ("action_clipping",      None,                        action_clip_attack,    None),
    ]

    for cond_name, attack_fn, action_fn, on_reset in conditions:
        out_dir = os.path.join(VIDEO_ROOT, cond_name)
        os.makedirs(out_dir, exist_ok=True)

        env = make_env(seed=SEED, rgb_mode=True)

        for ep_idx in range(N_EPISODES):
            if on_reset:
                on_reset()

            capture = ep_idx in SAMPLE_EPISODES
            frames = run_episode(env, policy, attack_fn, action_fn, capture_frames=capture)

            if capture:
                video_path = os.path.join(out_dir, f"episode_{ep_idx}.mp4")
                if frames:
                    imageio.mimsave(video_path, frames, fps=FPS)
                else:
                    open(video_path, "wb").close()
                print(f"Recorded [{cond_name}] episode {ep_idx} → {video_path}")

        env.close()


if __name__ == "__main__":
    main()
