# TAIRO Week 4 — Findings Summary
**Team:** Yves Velasquez Vega & Jachin Choi
**Project:** Trustworthy AI Robotics (TAIRO) — REU @ UMKC

---

## What We Were Trying to Answer

The central question for Week 4 was: **how well does a robot trained with reinforcement learning hold up when an attacker tampers with what it sees or does?**

We tested four types of robot controllers on a simulated robotic arm task (FetchReach-v4 in MuJoCo — the arm has to move its gripper to a floating target in 3D space). We then deliberately broke the robot's inputs in several ways and measured how each controller responded.

---

## The Robot Task

The environment is **FetchReach-v4**: a 7-DOF robotic arm must move its gripper to a randomly placed target position. Each episode runs for up to 50 steps. The robot gets a reward of 0 when it reaches the goal and -1 for every step it hasn't reached it yet, so the best possible score per episode is -1 or -2 (reach goal almost immediately), and the worst is -50 (never reach it).

---

## The Four Controllers We Tested

| Controller | What it is |
|---|---|
| **Random** | Picks random actions every step — simplest baseline |
| **Rule-Based** | A hand-coded proportional controller: computes the direction to the goal and moves toward it |
| **SAC (no HER)** | Neural network trained with Soft Actor-Critic — no goal relabelling. Added as an ablation to isolate HER's contribution |
| **SAC+HER** | Same SAC architecture but trained with Hindsight Experience Replay — our main learned AI policy |

Both SAC models were trained for 20,000 timesteps with identical hyperparameters (lr=1e-3, buffer=100k, batch=256, γ=0.95, τ=0.05). The only difference is the HER replay buffer. SAC+HER was confirmed working at 100% success before the attack sweep. SAC (no HER) achieved 0% success — see findings below for why this is expected and is itself the result.

Each controller was also tested in a **recovery-aware variant** (e.g. `sac_her_recovery`) that applies conservative action damping when an attack is detected.

---

## Attack Conditions Tested

| Condition | What it does | Level |
|---|---|---|
| **Clean** | No attack | — |
| **Sensor Noise** | Gaussian noise injected into the robot's observations (position data) | σ = 0.01 (mild), σ = 0.05 (heavy) |
| **Action Noise** | Gaussian noise added to motor commands after the policy decides | σ = 0.05 |
| **Action Scale** | Motor commands multiplied by a fixed factor (over-actuation) | 1.5× |
| **Target Shift** | Goal position spoofed by a random offset mid-episode | ±0.03 m |

> **Note on action_delay:** This condition was excluded from all results. A confirmed implementation issue — `previous_action` is initialised to zeros at episode start, so the delay mechanism sends zero commands for the entire episode regardless of policy. This makes all policies fail equally for the wrong reason. It is not a meaningful robustness test and will be corrected before future experiments.

---

## Results

### Clean Performance (No Attack)

| Controller | Success Rate | Avg Reward | Avg Steps to Goal |
|---|---|---|---|
| Random | 0% | -50.0 | Never reaches it |
| Rule-Based | 100% | -5.7 | ~5 steps |
| SAC (no HER) | **0%** | **-50.0** | Never reaches it |
| SAC+HER | 100% | **-1.7** | ~1–2 steps |

**HER is load-bearing, not just helpful.** Without goal relabelling, the sparse reward signal means SAC almost never observes a positive reward during training (only 3 of 400 training episodes succeeded). The policy cannot learn. The clean comparison — 0% for SAC (no HER) vs. 100% for SAC+HER — is itself a finding: goal-conditioned sparse-reward tasks require HER or an equivalent curriculum.

---

### Under Sensor Attack

| Controller | Noise σ | Success Rate | Avg Reward | Change vs Clean |
|---|---|---|---|---|
| Rule-Based | 0.01 | 100% | -6.0 | No change |
| Rule-Based | 0.05 | 100% | -9.0 | Slight slowdown |
| **SAC+HER** | **0.01** | **100%** | **-2.0** | Minimal impact |
| **SAC+HER** | **0.05** | **100%** | **-28.7** | **Reward dropped 27 points** |

SAC+HER still succeeds under heavy sensor noise but takes dramatically longer — roughly 27 extra steps — as corrupted observations throw it off course. Rule-Based is essentially unaffected because its proportional controller recomputes each step and small noise only slightly misaims it without accumulating error.

---

### Under Action Attack

| Controller | Condition | Success Rate | Avg Reward | Change vs Clean |
|---|---|---|---|---|
| Rule-Based | Action Noise σ=0.05 | 100% | -6.3 | Minimal |
| Rule-Based | Action Scale 1.5× | 100% | -3.7 | Minimal |
| Rule-Based | Target Shift ±0.03 | 100% | -5.7 | No change |
| **SAC+HER** | **Action Noise σ=0.05** | **100%** | **-1.7** | **No change** |
| **SAC+HER** | **Action Scale 1.5×** | **100%** | **-1.7** | **No change** |
| **SAC+HER** | **Target Shift ±0.03** | **100%** | **-1.7** | **No change** |

At benchmark levels, SAC+HER shows zero degradation across all three action-level and goal-spoofing attacks. The policy has learned to implicitly compensate.

---

### Recovery Variants

Each controller was paired with a recovery-aware variant that applies 0.5× action damping when either the action diverges from the intended command by more than 0.15 or the robot is more than 0.25 m from the goal. Recovery is confirmed to trigger — 914 of 2,700 recovery-method steps fired it.

**The finding is unflattering:** for controllers that were already succeeding (Rule-Based, SAC+HER), recovery damping mostly slows them down without improving outcomes, resulting in lower average reward in most conditions. The current damping strategy is too aggressive — it fires at the start of any episode (distance > 0.25 m) and halves actions that were already on track. Recovery design is a Week 5 item.

---

## Trustworthiness Scores

We computed composite TAIRO trustworthiness scores (0–1) using five component metrics (reliability, robustness, cyber-resilience, safety, recovery) weighted equally at 0.20 each:

| Controller | Clean | Sensor Noise σ=0.01 | Sensor Noise σ=0.05 | Action Noise σ=0.05 |
|---|---|---|---|---|
| Random | 0.41 | 0.34 | 0.16 | 0.11 |
| Rule-Based | **0.996** | 0.64 | 0.59 | 0.64 |
| SAC (no HER) | 0.46 | 0.11 | 0.02 | 0.10 |
| SAC+HER | 0.94 | 0.53 | 0.48 | 0.58 |

Under sensor noise, SAC+HER's score drops more than Rule-Based's — confirming that learned policies are more sensitive to perception corruption than deterministic controllers, even when otherwise superior.

---

## Core Finding

> **SAC+HER is the fastest and most capable controller under clean conditions, but sensor noise is its specific vulnerability. Rule-Based is slower but more robust to perception attacks because it doesn't rely on a learned representation of the world. HER is not an optional enhancement for this task — without it, neural RL fails completely on sparse rewards.**

This is a meaningful research result: learned policies that internalize a model of the environment can be more susceptible to adversarial sensor attacks than simpler reactive controllers, even when the learned policy is otherwise superior. This is precisely the kind of trustworthiness gap the TAIRO framework is designed to expose.

---

## Videos

| Video | Highlights |
|---|---|
| `clean_seed0` | Arm moves directly to the target in ~2 steps, then holds position |
| `sensor_noise_0.05_seed2` | Arm oscillates — keeps approaching then veering away as corrupted readings mislead it |
| `action_noise_0.05_seed0` | Looks almost identical to clean — SAC+HER is robust at this noise level |
| `action_noise_1.5_seed1` | Arm thrashes and fails to reach goal — shows where SAC+HER actually breaks (30× above benchmark) |

---

## Scripts

| Script | What it does |
|---|---|
| `training/train_sac_her.py` | Trains SAC+HER on FetchReach-v4 |
| `training/train_sac_plain.py` | Trains plain SAC (no HER) — ablation baseline |
| `scripts/run_baseline_experiment.py` | Runs the full attack sweep for any method/condition/seed combination |
| `scripts/build_results_table.py` | Regenerates `week4_results_table.csv` from `combined_summary.csv` |
| `scripts/build_results_plot.py` | Regenerates `week4_results_plot.png` from `week4_results_table.csv` |
| `scripts/record_hero_episodes.py` | Records video clips for the three hero episodes |

---

## Files Produced This Week

| File | What it is |
|---|---|
| `tairo_results/sac_her_fetchreach_model.zip` | Trained SAC+HER model (20k timesteps) |
| `tairo_results/sac_plain_fetchreach_model.zip` | Trained SAC (no HER) model (20k timesteps) — ablation |
| `tairo_results/combined_episode_results.csv` | Per-episode results for all runs (150 rows) |
| `tairo_results/combined_step_logs.csv` | Per-step logs including recovery_triggered flag |
| `tairo_results/combined_summary.csv` | Aggregated summary with all trustworthiness score components |
| `tairo_results/week4_results_table.csv` | Slide-ready results table, 44 rows, generated by `build_results_table.py` |
| `tairo_results/week4_results_plot.png` | Three-panel figure: success rate, avg reward, recovery triggered rate |
| `tairo_results/videos/` | Four .mp4 clips (clean, sensor noise, action noise ×2) |
| `tairo_results/run_all-conditions_seed0-1-2_*.csv` | Timestamped snapshots of each benchmark run |
