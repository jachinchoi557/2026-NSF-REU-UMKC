"""
Failure analysis sweep — SAC+HER with recovery v2, 100 episodes per condition, headless.

Mirrors run_failure_analysis_recovery.py exactly, but uses recovery_logic_v2 and
RecoveryState. A fresh RecoveryState is instantiated at the start of each episode
and passed into every call to maybe_apply_recovery.

Outputs:
  tairo_results/outputs/failure_analysis_recovery_v2_episodes.csv
  tairo_results/outputs/failure_analysis_recovery_v2_steps.csv

After all conditions complete, prints a summary table and writes:
  tairo_results/outputs/failure_summary_recovery_v2_table.csv
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
from recovery.recovery_logic_v2 import maybe_apply_recovery, RecoveryState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED       = 0
N_EPISODES = 100
MODEL_PATH = os.path.join(MODELS_DIR, "sac_her_fetchreach_model")

EPISODE_CSV = os.path.join(OUTPUTS_DIR, "failure_analysis_recovery_v2_episodes.csv")
STEPS_CSV   = os.path.join(OUTPUTS_DIR, "failure_analysis_recovery_v2_steps.csv")
SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "failure_summary_recovery_v2_table.csv")

EPISODE_FIELDS = [
    "condition", "episode_idx", "success", "total_reward", "final_distance",
    "recovery_triggered_count", "first_recovery_step",
    "steps_to_success", "max_distance_from_goal",
]
STEP_FIELDS = [
    "condition", "episode_idx", "step", "distance_to_goal",
    "recovery_triggered", "action_norm",
]

CONDITION_ORDER = [
    "clean",
    "sensor_dropout",
    "sensor_bias",
    "goal_spoof_immediate",
    "goal_spoof_midep",
    "action_delay",
    "action_reverse",
    "action_clipping",
]

SUMMARY_FIELDS = [
    "condition",
    "success_rate",
    "avg_reward",
    "avg_final_distance",
    "recovery_pct",
    "avg_first_recovery_step",
]

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
# Episode runner — with recovery v2
# ---------------------------------------------------------------------------

def run_episode(env, policy, attack_fn, action_fn):
    """Run one episode with recovery v2; return (episode_row, list_of_step_rows)."""
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    recovery_triggered_count = 0
    first_recovery_step = -1
    steps_to_success = -1
    max_dist = 0.0
    step_rows = []

    prev_obs = None
    prev_action = None
    step_distances = []
    state = RecoveryState()   # fresh state for this episode

    while not done:
        dist = distance_to_goal(obs)
        max_dist = max(max_dist, dist)

        policy_obs = attack_fn(obs) if attack_fn else obs
        action, _ = policy.predict(policy_obs, deterministic=True)

        if action_fn:
            action = action_fn(action)

        final_action, recovery_triggered = maybe_apply_recovery(
            obs=obs,
            action=action,
            prev_obs=prev_obs,
            prev_action=prev_action,
            step_distances=step_distances,
            step=step,
            env=env,
            state=state,
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
# Summary table helpers (mirrors print_failure_summary.py)
# ---------------------------------------------------------------------------

def compute_summary(rows):
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    summary = []
    for cond in CONDITION_ORDER:
        eps = by_cond.get(cond, [])
        if not eps:
            continue
        n = len(eps)
        success_rate   = 100.0 * sum(e["success"] for e in eps) / n
        avg_reward     = sum(e["total_reward"] for e in eps) / n
        avg_final_dist = sum(e["final_distance"] for e in eps) / n
        recovery_pct   = 100.0 * sum(1 for e in eps if e["recovery_triggered_count"] > 0) / n
        fired = [e["first_recovery_step"] for e in eps if e["first_recovery_step"] >= 0]
        avg_first_recovery = (sum(fired) / len(fired)) if fired else None
        summary.append({
            "condition":               cond,
            "success_rate":            success_rate,
            "avg_reward":              avg_reward,
            "avg_final_distance":      avg_final_dist,
            "recovery_pct":            recovery_pct,
            "avg_first_recovery_step": avg_first_recovery,
        })
    return summary


def print_table(summary):
    col_cond = max(len(r["condition"]) for r in summary)
    col_cond = max(col_cond, len("Condition"))

    header = (
        f"{'Condition':<{col_cond}} │ "
        f"{'Success Rate':>12} │ "
        f"{'Avg Reward':>10} │ "
        f"{'Avg Dist':>8} │ "
        f"{'Recovery (%)':>12} │ "
        f"{'Avg 1st Rec Step':>16}"
    )
    sep_inner = (
        f"{'─' * col_cond}─┼─"
        f"{'─' * 12}─┼─"
        f"{'─' * 10}─┼─"
        f"{'─' * 8}─┼─"
        f"{'─' * 12}─┼─"
        f"{'─' * 16}"
    )
    top = (
        f"┌{'─' * col_cond}─┬─"
        f"{'─' * 12}─┬─"
        f"{'─' * 10}─┬─"
        f"{'─' * 8}─┬─"
        f"{'─' * 12}─┬─"
        f"{'─' * 16}┐"
    )
    bot = (
        f"└{'─' * col_cond}─┴─"
        f"{'─' * 12}─┴─"
        f"{'─' * 10}─┴─"
        f"{'─' * 8}─┴─"
        f"{'─' * 12}─┴─"
        f"{'─' * 16}┘"
    )

    print("\n=== Recovery v2 Summary ===")
    print(top)
    print(f"│ {header} │")
    print(f"├{sep_inner}┤")
    for r in summary:
        fmt_pct  = lambda v: f"{v:.1f}%"
        fmt_f    = lambda v: f"{v:.2f}"
        fmt_step = lambda v: f"{v:.1f}" if v is not None else "N/A"
        row = (
            f"{r['condition']:<{col_cond}} │ "
            f"{fmt_pct(r['success_rate']):>12} │ "
            f"{fmt_f(r['avg_reward']):>10} │ "
            f"{fmt_f(r['avg_final_distance']):>8} │ "
            f"{fmt_pct(r['recovery_pct']):>12} │ "
            f"{fmt_step(r['avg_first_recovery_step']):>16}"
        )
        print(f"│ {row} │")
    print(bot)


def write_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in summary:
            fmt_step = lambda v: f"{v:.1f}" if v is not None else "N/A"
            writer.writerow({
                "condition":               r["condition"],
                "success_rate":            f"{r['success_rate']:.1f}",
                "avg_reward":              f"{r['avg_reward']:.2f}",
                "avg_final_distance":      f"{r['avg_final_distance']:.4f}",
                "recovery_pct":            f"{r['recovery_pct']:.1f}",
                "avg_first_recovery_step": fmt_step(r["avg_first_recovery_step"]),
            })
    print(f"Wrote {path}")


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

    all_ep_rows = []

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

                full_ep_row = {"condition": cond_name, "episode_idx": ep_idx, **ep_row}
                ep_writer.writerow(full_ep_row)
                all_ep_rows.append(full_ep_row)

                for sr in step_rows:
                    st_writer.writerow({"condition": cond_name, "episode_idx": ep_idx, **sr})

            env.close()
            print(
                f"[{cond_name}] done — {successes}/{N_EPISODES} succeeded "
                f"(recovery triggered in {recovery_episodes} episodes)"
            )

    print(f"\nWrote {EPISODE_CSV}")
    print(f"Wrote {STEPS_CSV}")

    # --- Checkpoint: summary table -------------------------------------------
    summary = compute_summary(all_ep_rows)
    print_table(summary)
    write_summary_csv(summary, SUMMARY_CSV)


if __name__ == "__main__":
    main()
