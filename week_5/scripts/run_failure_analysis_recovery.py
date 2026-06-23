"""
Failure analysis sweep — SAC+HER with recovery, 100 episodes per condition, headless.

Mirrors run_failure_analysis.py exactly, with use_recovery=True.
Recovery is applied via maybe_apply_recovery() after each action attack, using the
same call pattern as evaluation/episode_runner.py:
  - obs passed to recovery is the RAW unattacked observation (so rule-based replanning
    steers toward the true goal, not the spoofed one)
  - prev_action tracks the final executed action (post-attack, post-recovery)
  - step_distances is appended after env.step() to feed the trend detector

Same seed, same conditions, same attack closures as run_failure_analysis.py so
episode N here is directly comparable to episode N in failure_analysis_episodes.csv.

Outputs:
  tairo_results/outputs/failure_analysis_recovery_episodes.csv
  tairo_results/outputs/failure_analysis_recovery_steps.csv
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC

from config import MODELS_DIR, OUTPUTS_DIR
from envs.fetchreach_env import make_env, distance_to_goal
from policies.sac_her_policy import SACHerPolicy
from attacks.sensor_attacks import apply_sensor_dropout, apply_sensor_bias, shift_target
from attacks.action_attacks import manipulate_action
from recovery.recovery_logic import maybe_apply_recovery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED       = 0
N_EPISODES = 100
MODEL_PATH = os.path.join(MODELS_DIR, "sac_her_fetchreach_model")

EPISODE_CSV = os.path.join(OUTPUTS_DIR, "failure_analysis_recovery_episodes.csv")
STEPS_CSV   = os.path.join(OUTPUTS_DIR, "failure_analysis_recovery_steps.csv")

EPISODE_FIELDS = [
    "condition", "episode_idx", "success", "total_reward", "final_distance",
    "recovery_triggered_count", "first_recovery_step",
    "steps_to_success", "max_distance_from_goal",
]
STEP_FIELDS = [
    "condition", "episode_idx", "step", "distance_to_goal",
    "recovery_triggered", "action_norm",
]

# ---------------------------------------------------------------------------
# Attack factories — identical to run_failure_analysis.py
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
# Episode runner — with recovery
# ---------------------------------------------------------------------------

def run_episode(env, policy, attack_fn, action_fn):
    """Run one episode with recovery enabled; return (episode_row, list_of_step_rows)."""
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    recovery_triggered_count = 0
    first_recovery_step = -1
    steps_to_success = -1
    max_dist = 0.0
    step_rows = []

    # Recovery tracking state — mirrors episode_runner.py
    prev_obs = None
    prev_action = None        # final executed action from previous step
    step_distances = []       # appended after env.step; feeds trend detector

    while not done:
        dist = distance_to_goal(obs)
        max_dist = max(max_dist, dist)

        # Observation attack — policy sees corrupted obs; recovery sees raw obs
        policy_obs = attack_fn(obs) if attack_fn else obs
        action, _ = policy.predict(policy_obs, deterministic=True)

        # Action attack
        if action_fn:
            action = action_fn(action)

        # Recovery (C5) — uses raw obs so rule-based controller steers to real goal
        final_action, recovery_triggered = maybe_apply_recovery(
            obs=obs,
            action=action,
            prev_obs=prev_obs,
            prev_action=prev_action,
            step_distances=step_distances,
            step=step,
            env=env,
        )

        if recovery_triggered:
            recovery_triggered_count += 1
            if first_recovery_step == -1:
                first_recovery_step = step

        prev_obs = obs
        prev_action = final_action.copy()

        obs, reward, terminated, truncated, info = env.step(final_action)
        total_reward += reward
        done = terminated or truncated

        step_distances.append(distance_to_goal(obs))

        if info.get("is_success", False) and steps_to_success == -1:
            steps_to_success = step

        step_rows.append({
            "step": step,
            "distance_to_goal": dist,
            "recovery_triggered": int(recovery_triggered),
            "action_norm": float(np.linalg.norm(final_action)),
        })
        step += 1

    success = bool(info.get("is_success", False))
    ep_row = {
        "success": int(success),
        "total_reward": total_reward,
        "final_distance": distance_to_goal(obs),
        "recovery_triggered_count": recovery_triggered_count,
        "first_recovery_step": first_recovery_step,
        "steps_to_success": steps_to_success,
        "max_distance_from_goal": max_dist,
    }
    return ep_row, step_rows


# ---------------------------------------------------------------------------
# Main sweep
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

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(EPISODE_CSV, "w", newline="") as ep_f, \
         open(STEPS_CSV,   "w", newline="") as st_f:

        ep_writer = csv.DictWriter(ep_f, fieldnames=EPISODE_FIELDS)
        st_writer = csv.DictWriter(st_f, fieldnames=STEP_FIELDS)
        ep_writer.writeheader()
        st_writer.writeheader()

        for cond_name, attack_fn, action_fn, on_reset in conditions:
            env = make_env(seed=SEED)
            successes = 0
            recovery_episodes = 0

            for ep_idx in range(N_EPISODES):
                if on_reset:
                    on_reset()

                try:
                    ep_row, step_rows = run_episode(env, policy, attack_fn, action_fn)
                except Exception as exc:
                    print(f"  [WARN] {cond_name} ep {ep_idx} crashed: {exc!r} — logged as failure")
                    ep_row = {
                        "success": 0, "total_reward": 0.0, "final_distance": -1.0,
                        "recovery_triggered_count": 0, "first_recovery_step": -1,
                        "steps_to_success": -1, "max_distance_from_goal": -1.0,
                    }
                    step_rows = []

                successes += ep_row["success"]
                if ep_row["recovery_triggered_count"] > 0:
                    recovery_episodes += 1

                ep_writer.writerow({"condition": cond_name, "episode_idx": ep_idx, **ep_row})
                for sr in step_rows:
                    st_writer.writerow({"condition": cond_name, "episode_idx": ep_idx, **sr})

            env.close()
            print(
                f"[{cond_name}] done — {successes}/{N_EPISODES} succeeded "
                f"(recovery triggered in {recovery_episodes} episodes)."
            )

    print(f"\nWrote {EPISODE_CSV}")
    print(f"Wrote {STEPS_CSV}")


if __name__ == "__main__":
    main()
