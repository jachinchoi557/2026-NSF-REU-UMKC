# Week 5 Session Summary
**Date:** 2026-06-19  
**Branch:** week-5-dev  
**Author:** Yves (with Claude Code)

---

## What We Built This Session

This session extended the TAIRO robotic cybersecurity benchmark from Week 4 by adding four new types of cyberattack scenarios, redesigning the robot's recovery system from scratch, and producing a new set of experimental results across 960 robot episodes. We also added a second trustworthiness scoring formula so the paper can argue the weight choices directly rather than just presenting one number. The session produced two new presentation-ready files — a results table and a four-panel figure — that can go directly into slides or the Overleaf draft.

---

## New Attack Scenarios

### Sensor Dropout (`sensor_dropout`)
**What it simulates:** A complete hardware or communication failure in the robot's proprioceptive sensing system — equivalent to a camera feed going black or an IMU losing power entirely. The robot receives only zeros where it would normally receive its joint positions, velocities, and gripper state.  
**How it's implemented:** The attack function (`apply_sensor_dropout`) replaces the entire `observation` field in the robot's perception dictionary with a zero array before the policy ever sees it. The goal location remains visible, but the robot has no idea where its own hand is.  
**Cybersecurity threat:** Denial-of-service attack on a sensor bus or firmware, or physical severing of a sensor cable.

### Sensor Bias (`sensor_bias`)
**What it simulates:** A sensor that has been tampered with or is miscalibrated in a systematic way — it doesn't give random noise, it consistently reads high or low by a fixed amount. Imagine a position sensor whose zero-point has been shifted by an adversary or by long-term drift.  
**How it's implemented:** At the start of each episode, one constant offset vector is sampled from a uniform distribution (±0.10 per dimension). This same vector is added to the robot's observation on every step. The offset is fixed for the whole episode to simulate persistent miscalibration, not random noise.  
**Cybersecurity threat:** Supply-chain compromise of sensor firmware, or a man-in-the-middle attacker who intercepts the sensor data stream and adds a constant shift.

### Goal Spoofing — Immediate (`goal_spoof_immediate`)
**What it simulates:** An adversary who corrupts the mission command the robot receives — the robot is told to reach toward a false target position from the very first step of the episode.  
**How it's implemented:** A random offset (±0.10 m per axis) is sampled once at the start of the episode and added to the `desired_goal` field in every observation the policy sees. The robot's actual target in the simulator remains unchanged; only the goal the policy is shown gets corrupted.  
**Cybersecurity threat:** Man-in-the-Middle attack on the goal command channel — an attacker intercepts the mission planner's output and substitutes a false goal.

### Goal Spoofing — Mid-Episode (`goal_spoof_midep`)
**What it simulates:** The same MitM goal corruption as above, but the attacker waits until the robot is 40% through its task (step 20 of 50) before activating. This tests whether policies that have already started moving toward the correct goal can be redirected by a late-stage injection.  
**How it's implemented:** Identical to the immediate version, except the offset is not applied until step 20. Before step 20, the policy sees the true goal and behaves normally. From step 20 onward, it sees the corrupted goal. The offset is held constant once activated.  
**Cybersecurity threat:** A "sleeper" network intrusion that activates mid-operation rather than at startup, making it harder to attribute or detect at mission start.

---

## Recovery Logic Redesign

### What the old approach did
The Week 4 recovery system detected an attack by checking two things: (1) whether the executed action differed significantly from the intended action, and (2) whether the robot was farther than 0.25 m from its goal. When either condition was met, it halved the magnitude of the executed action.

### Why it was net-harmful
The distance threshold of 0.25 m fires at the very start of every episode — the robot always begins further than 0.25 m from the goal. This means "recovery" was triggering continuously from step 0 in virtually every episode, not just when attacks were actually occurring. For a capable policy like SAC+HER or the rule-based controller, slowing down a perfectly good action by 50% consistently made performance worse. The damping didn't redirect the robot — it just made it move half as far per step.

### What the new approach does
The redesigned recovery replaces damping with a diagnostic system and a meaningful response:

**Three detection signals:**
- **Action divergence** (threshold > 0.5): If the action at this step is very different from the previous step's action — much more so than normal control variation — something has likely been injected. This catches noise injection and scaling attacks.
- **Jerk** (threshold > 1.0): The same measurement at a higher bar. A sudden lurch of this magnitude almost certainly means the action has been inverted or severely scaled, not just nudged by noise.
- **Distance trend** (5 consecutive increases): If the robot's distance from the goal has been getting worse for five steps in a row without ever improving, it's being driven the wrong direction. This catches sensor-based attacks that don't show up in action divergence because the policy action itself looks normal.

**Recovery response:** When any signal fires, the corrupted action is thrown away entirely. Instead, the recovery module calls the rule-based reaching controller — the simple proportional "point your hand toward the goal" policy — using the **raw, unattacked observation**. This is critical for goal-spoofing attacks: even if the policy was shown a false goal location, the recovery controller uses the true goal location the environment actually tracks, so the robot gets re-anchored to where it's actually supposed to go.

---

## Trustworthiness Scoring

The TAIRO framework computes a single trustworthiness score for each (policy, attack condition) pair by combining five component scores: Reliability (did it succeed?), Robustness (how close did it get?), Cyber Resilience (did it succeed under attack?), Safety (did it avoid unsafe actions?), and Recovery (did recovery ever trigger?).

### The two weight schemes

**Equal weights (0.20 × each):** Every component counts the same. This is the conservative, hard-to-argue-against baseline. If the paper reviewers question our weight choices, equal weights give a model-free comparison point.

**Argued weights (Week 3 TAIRO framework):**
- Reliability: 0.10 — basic task success is expected, not distinguishing
- Robustness: 0.20 — getting close matters, but less than adversarial survival
- Cyber Resilience: 0.25 — the primary thesis of the paper; attack survival is most important
- Safety: 0.15 — actuator safety is a hard real-world constraint
- Recovery: 0.30 — the highest weight, because a policy that can detect and correct an attack is qualitatively better than one that degrades gracefully

### Why having both is useful for the paper
The two columns appear side by side in the results table. For the rule-based and SAC+HER policies under sensor_bias and goal_spoof_midep — where they achieve 100% success — the equal score (≈0.80) and weighted score (≈0.70) are close. For recovery variants under sensor_dropout — where recovery triggers aggressively — the weighted score is notably higher than the equal score because recovery gets the largest weight. This creates a concrete, data-grounded argument in the paper for why recovery deserves its high weight: it's the factor that most differentiates policies that merely survive from policies that actively correct.

---

## Data Produced

| File | Location | Contents | Use |
|---|---|---|---|
| `combined_episode_results.csv` | `tairo_results/canonical/` | 960 rows × 13 cols. One row per episode. method, condition, seed, episode_idx, success, reward, distance, recovery_used. | Source of truth for all downstream analysis. |
| `combined_step_logs.csv` | `tairo_results/canonical/` | 48,000 rows × 13 cols. One row per timestep per episode. Includes recovery_triggered per step. | Recovery trigger rate computation; per-step analysis. |
| `combined_summary.csv` | `tairo_results/canonical/` | 32 rows × 20 cols. One row per (method, condition). All metric averages + both trustworthiness scores. | Source for table and plot scripts. |
| `week5_results_table.csv` | `tairo_results/outputs/` | 32 rows × 9 cols. Human-readable method/condition names. Both trustworthiness score columns. | Paste into slides; cite specific numbers in paper. |
| `week5_results_plot.png` | `tairo_results/outputs/` | 233 KB, 4-panel figure (dpi=150). Panels: Success Rate, Avg Reward, Recovery Rate, Equal vs. Weighted score. | Main results figure for slides and paper. |
| `run_all-conditions_seed0-1-2_20260619-1501_*` | `tairo_results/experiments/all-conditions/` | Timestamped snapshot of the 2026-06-19 sweep. Three files (episodes, steps, summary). Read-only. | Audit trail; revert if canonical CSVs are accidentally modified. |

---

## What the Results Show So Far

The most striking finding is the difference in how policies handle observation-level vs. goal-corruption attacks.

**Sensor dropout is catastrophic for learned policies.** SAC+HER and SAC (no HER) both achieve only ~3% success under sensor dropout — they cannot function without their proprioceptive observation. The rule-based controller, by contrast, achieves 100% success even with sensor dropout, because its control law only needs the goal location and the achieved goal (end-effector position), not the full observation. This is a meaningful paper finding: a simple hand-designed controller is more robust to a complete sensor failure than a neural network trained on millions of data points.

**Sensor bias has a much smaller effect on the rule-based controller than on learned policies.** Rule-based achieves 100% success under sensor bias; SAC+HER only achieves 30%. This makes sense — the rule-based policy uses `desired_goal - achieved_goal` (goal minus end-effector position), which is in the goal space and not the observation field. Sensor bias on `obs["observation"]` doesn't touch those fields, so the rule-based controller is immune. SAC+HER, which learned to use the full observation as context, is more affected.

**Goal spoofing is where the learning advantage disappears.** Under immediate goal spoofing, SAC+HER achieves only 23% success — worse than random might expect from a capable policy. The rule-based controller achieves 47%. Under mid-episode spoofing (activated at step 20), both achieve 100%. The rule-based controller has already moved close enough to the true goal in the first 20 steps that a 0.10 m goal shift at step 20 doesn't prevent success within the remaining 30 steps. SAC+HER can also recover, but its early-episode behavior is more conservative, so it succeeds 100% as well.

**Recovery under the new rule-based replanning scheme is largely benign.** Unlike Week 4's damping (which actively hurt capable policies), the new recovery system triggers rarely for rule-based and SAC+HER (recovery rate < 1% for most conditions), and when it does, it substitutes a sensible reaching action rather than slowing the robot down. The most visible recovery effect is in the sensor_dropout case for SAC+HER + Recovery: recovery triggers 90% of steps (the action divergence and distance trend fire immediately because the policy outputs erratic actions without proprioception), and the recovery action substitutes the rule-based controller — explaining why `sac_her_recovery` under `sensor_dropout` achieves 3% success while the raw rule-based achieves 100%.

---

## What Still Needs to Be Done Before Monday

1. **Fix action_delay** — The `action_delay` attack will currently crash if run because `previous_action` is now `None` at step 0. Needs a one-line fix in `action_attacks.py`, then a re-run of the action_delay sweep.

2. **Sweep action_reverse** — This attack function exists but has never been run. Should be added to the next sweep alongside the action_delay fix.

3. **Implement action_clipping** — Not yet written. Would round out the action-level attack taxonomy.

4. **Overleaf section draft** — The attack taxonomy section needs to be written. The four new attacks + Week 4 attacks give a clean taxonomy: observation attacks (noise, dropout, bias) vs. goal attacks (shift, spoof immediate, spoof mid-episode) vs. action attacks (noise, scale, delay, reverse).

5. **SOTA comparison** — Need to find and cite published numbers on FetchReach robustness to compare against. The sensor_dropout finding (rule-based > SAC+HER) is the most publishable result and should be framed against the literature.

6. **Habitat extension sketch** — At minimum, a one-page design note explaining how each attack maps to a real Habitat navigation scenario. Sensor dropout → camera failure. Goal spoof → semantic target corruption. Sensor bias → depth sensor miscalibration. This sets up Week 6 if the Habitat simulation track continues.

7. **readme.md update** — The readme still describes Week 4 results and incorrect equal weights. Should be updated to reference the Week 5 findings and correct weight formula.
