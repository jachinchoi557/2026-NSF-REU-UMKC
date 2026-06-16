# TAIRO Week 4 — Findings Summary
**Team:** Yves Velasquez Vega & Jachin Choi
**Project:** Trustworthy AI Robotics (TAIRO) — REU @ UMKC

---

## What We Were Trying to Answer

The central question for Week 4 was: **how well does a robot trained with reinforcement learning hold up when an attacker tampers with what it sees or does?**

We tested three types of robot controllers on a simulated robotic arm task (FetchReach-v4 in MuJoCo — the arm has to move its gripper to a floating target in 3D space). We then deliberately broke the robot's inputs in two different ways and measured how each controller responded.

---

## The Robot Task

The environment is **FetchReach-v4**: a 7-DOF robotic arm must move its gripper to a randomly placed target position. Each episode runs for up to 50 steps. The robot gets a reward of 0 when it reaches the goal and -1 for every step it hasn't reached it yet, so the best possible score per episode is -1 or -2 (reach goal almost immediately), and the worst is -50 (never reach it).

---

## The Three Controllers We Tested

| Controller | What it is |
|---|---|
| **Random** | Picks random actions every step — our dumbest baseline |
| **Rule-Based** | A hand-coded proportional controller: computes the direction to the goal and moves toward it |
| **SAC+HER** | A neural network trained with Soft Actor-Critic + Hindsight Experience Replay — our learned AI policy |

The SAC+HER model was trained for 20,000 timesteps and saved. Before running any attack experiments, we confirmed it works: 100% success rate over 5 visual episodes, reaching the goal in as little as 1–2 steps.

---

## The Two Attack Scenarios

### Scenario 1 — Sensor Attack (Observation Corruption)
We injected **Gaussian noise directly into the robot's observations** — the numbers that tell it where its arm and the target are. This simulates a compromised sensor or a man-in-the-middle attack on the robot's perception pipeline.

- Tested at two noise levels: **σ = 0.01** (mild) and **σ = 0.05** (heavy)
- The robot still moves, but it's making decisions based on corrupted position data

### Scenario 2 — Action Attack (Command Corruption)
We **corrupted the action the policy sent to the motors** before it was executed. The robot decides what to do correctly, but the command gets garbled before it reaches the arm.

- Tested at **σ = 0.05** for the benchmark
- Also tested up to σ = 1.5 to find the actual breaking point (used for video)

---

## Results

### Clean Performance (No Attack)

| Controller | Success Rate | Avg Reward | Avg Steps to Goal |
|---|---|---|---|
| Random | 0% | -50.0 | Never reaches it |
| Rule-Based | 100% | -5.7 | ~5 steps |
| SAC+HER | 100% | **-1.7** | ~1–2 steps |

SAC+HER reaches the goal almost instantly — significantly faster than the rule-based controller. This is the baseline we compare everything against.

---

### Under Sensor Attack (Scenario 1)

| Controller | Noise σ | Success Rate | Avg Reward | Change vs Clean |
|---|---|---|---|---|
| Random | 0.01 | 67% | -46.3 | — (random baseline is meaningless) |
| Random | 0.05 | 33% | -49.7 | — |
| Rule-Based | 0.01 | 100% | -6.0 | No change |
| Rule-Based | 0.05 | 100% | -9.0 | Slight slowdown |
| **SAC+HER** | **0.01** | **100%** | **-2.0** | Minimal impact |
| **SAC+HER** | **0.05** | **100%** | **-28.7** | **Reward dropped by 27 points** |

**Key finding:** SAC+HER still succeeds (100% success rate) under heavy sensor noise, but takes dramatically longer — reward drops from -1.7 to -28.7, meaning it takes roughly 27 extra steps to reach the goal. The arm is constantly "finding and losing" the target as corrupted observations throw it off course. You can see this clearly in the `sensor_noise_0.05_seed2` video.

Rule-Based is essentially unaffected — this is expected because it computes a direct vector toward the goal each step, so a bit of noise just slightly misaims it but doesn't accumulate into large errors.

---

### Under Action Attack (Scenario 2)

| Controller | Noise σ | Success Rate | Avg Reward | Change vs Clean |
|---|---|---|---|---|
| Rule-Based | 0.05 | 100% | -6.3 | Minimal |
| **SAC+HER** | **0.05** | **100%** | **-1.7** | **No change at all** |

At benchmark noise (σ = 0.05), SAC+HER shows **zero degradation** from action corruption. It's completely robust at this level.

To find where it actually breaks, we pushed the noise much higher:

| Noise σ | Success Rate | Avg Reward |
|---|---|---|
| 0.05 | 100% | -1.7 |
| 0.50 | 100% | -2.3 |
| 1.00 | 100% | -13.3 |
| **1.50** | **0%** | **-28.3** |
| 2.00 | 67% | -26.0 |

SAC+HER doesn't break until noise is **30× higher** than the benchmark level. This suggests the trained policy has learned to implicitly compensate for small action perturbations — a sign of good robustness in the learned controller.

---

## Trustworthiness Scores

We computed a composite TAIRO trustworthiness score (0–1) for each condition using the Week 3 framework weights (C1–C5: 0.10 / 0.25 / 0.25 / 0.05 / 0.35):

| Controller | Clean | Sensor Noise σ=0.01 | Sensor Noise σ=0.05 | Action Noise σ=0.05 |
|---|---|---|---|---|
| Random | 0.41 | 0.34 | 0.16 | 0.11 |
| Rule-Based | **0.996** | 0.64 | 0.59 | 0.64 |
| SAC+HER | 0.94 | 0.53 | 0.48 | 0.58 |

Under clean conditions, Rule-Based scores nearly perfect (0.996) because it's consistent and safe. SAC+HER scores 0.94 — slightly lower because the trustworthiness formula also factors in action smoothness and the recovery component, which is weighted highest (0.35) and was not triggered in these runs.

Under sensor noise, SAC+HER's score drops more than Rule-Based's — confirming that learned policies are more sensitive to perception corruption than deterministic controllers.

---

## The Core Finding for our Presentation

> **SAC+HER is the fastest and most capable controller under clean conditions, but sensor noise is its specific vulnerability. Rule-Based is slower but more robust to perception attacks because it doesn't rely on a learned representation of the world.**

This is actually a meaningful research insight: learned policies that internalize a model of the environment can be more susceptible to adversarial sensor attacks than simpler reactive controllers — even when the learned policy is otherwise superior. This is precisely the kind of trustworthiness gap the TAIRO framework is designed to expose.

---

## Videos

| Video | Highlights |
|---|---|
| `clean_seed0` | Arm moves directly and smoothly to the target in ~2 steps, then holds position |
| `sensor_noise_0.05_seed2` | Arm oscillates — keeps approaching then veering away as corrupted readings mislead it |
| `action_noise_0.05_seed0` | Looks almost identical to clean — SAC+HER is robust at this noise level |
| `action_noise_1.5_seed1` | Arm thrashes and fails to reach the goal — shows the actual breaking point |

---

## What Was Excluded and Why

- **action_delay**: excluded due to an implementation issue where a zero-action bug at episode start breaks all policies equally — not a meaningful attack result
- **action_scale** and **target_shift**: out of scope for Wednesday per the project plan — future experiment candidates
- **Recovery logic** (`recovery_logic.py`): the module exists but was not wired into a completed benchmark run this week

---

## Files Produced This Week

| File | What it is |
|---|---|
| `tairo_results/sac_her_fetchreach_model.zip` | Trained SAC+HER model |
| `tairo_results/combined_episode_results.csv` | Per-episode results for all runs (117 rows) |
| `tairo_results/combined_summary.csv` | Aggregated summary with trustworthiness scores |
| `tairo_results/week4_results_table.csv` | Slide-ready results table (two-scenario design only) |
| `tairo_results/week4_results_plot.png` | Two-panel figure: avg reward + trustworthiness by condition |
| `tairo_results/videos/` | Four .mp4 clips (clean, sensor noise, action noise ×2) |
| `tairo_results/run_all-conditions_seed0-1-2_20260616-1430_*.csv` | Timestamped snapshot of the full benchmark run |
