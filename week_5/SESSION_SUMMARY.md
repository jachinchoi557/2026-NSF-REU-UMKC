# Week 5 Session Summary
**Date:** 2026-06-19 (Session 1) + 2026-06-20 (Session 2)  
**Branch:** week-5-dev  
**Author:** Yves (with Claude Code)

---

## Session 1 — 2026-06-19: Core Week 5 Setup

### What We Built

This session extended the TAIRO robotic cybersecurity benchmark from Week 4 by adding four new types of cyberattack scenarios, redesigning the robot's recovery system from scratch, and producing a new set of experimental results across 960 robot episodes. We also added a second trustworthiness scoring formula so the paper can argue the weight choices directly rather than just presenting one number.

---

### New Attack Scenarios

#### Sensor Dropout (`sensor_dropout`)
**What it simulates:** A complete hardware or communication failure in the robot's proprioceptive sensing system — equivalent to a camera feed going black or an IMU losing power entirely. The robot receives only zeros where it would normally receive its joint positions, velocities, and gripper state.  
**How it's implemented:** `apply_sensor_dropout()` replaces the entire `observation` field with a zero array before the policy sees it. The goal location remains visible, but the robot has no idea where its own hand is.  
**Cybersecurity threat:** Denial-of-service attack on a sensor bus or firmware, or physical severing of a sensor cable.

#### Sensor Bias (`sensor_bias`)
**What it simulates:** A sensor that has been tampered with or is miscalibrated in a systematic way — it doesn't give random noise, it consistently reads high or low by a fixed amount.  
**How it's implemented:** At the start of each episode, one constant offset vector is sampled from a uniform distribution (±0.10 per dimension). This same vector is added to the robot's observation on every step.  
**Cybersecurity threat:** Supply-chain compromise of sensor firmware, or a man-in-the-middle attacker who intercepts the sensor data stream and adds a constant shift.

#### Goal Spoofing — Immediate (`goal_spoof_immediate`)
**What it simulates:** An adversary who corrupts the mission command the robot receives — the robot is told to reach toward a false target position from the very first step.  
**How it's implemented:** A random offset (±0.10 m per axis) is sampled once at episode start and added to `desired_goal` in every observation the policy sees.  
**Cybersecurity threat:** Man-in-the-Middle attack on the goal command channel.

#### Goal Spoofing — Mid-Episode (`goal_spoof_midep`)
**What it simulates:** The same MitM goal corruption as above, but the attacker waits until step 20 of 50 before activating. Tests whether policies already moving toward the correct goal can be redirected by a late-stage injection.  
**Cybersecurity threat:** A "sleeper" network intrusion that activates mid-operation, making it harder to detect at mission start.

---

### Recovery Logic Redesign

**Old approach (Week 4):** Detected an attack by checking action divergence and distance from goal (> 0.25 m). When triggered, halved the magnitude of the executed action.

**Why it was net-harmful:** The 0.25 m distance threshold fires at the very start of every episode. Recovery was triggering continuously from step 0, slowing down capable policies without redirecting them.

**New approach:** Replaces damping with a 3-signal detector and a meaningful response:
- **Action divergence** (> 0.5): Large step-to-step action jump → injected noise or scaling attack
- **Jerk** (> 1.0): Same quantity at higher bar — captures sudden reversals
- **Distance trend** (5 consecutive increases): Robot being driven away from goal → sensor-based attack

When any signal fires, the corrupted action is discarded entirely. The rule-based proportional controller is called using the **raw unattacked observation** — re-anchoring toward the true goal even when the policy was shown a false goal.

---

### Trustworthiness Scoring

Two weight schemes added:
- **Equal weights (0.20 × each):** Conservative, hard-to-argue-against baseline.
- **Argued weights:** Reliability 0.10, Robustness 0.20, Cyber Resilience 0.25, Safety 0.15, Recovery 0.30 (highest — policies that actively correct attacks are qualitatively better than those that degrade gracefully).

Both appear as `trustworthiness_score_equal` and `trustworthiness_score_weighted` in all CSVs and the results table.

---

### Data Produced (Session 1)

| File | Contents |
|---|---|
| `canonical/combined_episode_results.csv` | 960 rows — 4 conditions × 8 methods × 3 seeds × 10 episodes |
| `canonical/combined_step_logs.csv` | 48,000 rows (per-step logs) |
| `canonical/combined_summary.csv` | 32 rows × 20 cols — all metrics + both trustworthiness scores |
| `outputs/week5_results_table.csv` | 32 rows × 9 cols — slide-ready |
| `outputs/week5_results_plot.png` | 4-panel figure, 233 KB |

---

### Key Results (Session 1)

- **Sensor dropout is catastrophic for learned policies.** SAC+HER achieves only ~3% success without proprioception. The rule-based controller, which only uses goal position and end-effector position (not the full observation field), achieves 100%. This is the headline paper finding: a simple hand-designed controller is more robust to complete sensor failure than a neural network trained on millions of data points.
- **Sensor bias affects SAC+HER (30% success) but not Rule-Based (100%)** because the bias is on `obs["observation"]`, not the goal fields that the rule-based controller uses.
- **Goal spoofing mid-episode (step 20):** Both SAC+HER and Rule-Based achieve 100% success. By step 20, the robot is already close enough to the true goal that a ±0.10 m shift can't prevent success in the remaining 30 steps.
- **Recovery is largely benign under the new design** — triggers rarely for capable policies on most conditions. The exception is sensor_dropout: the erratic policy output triggers recovery on ~90% of steps, substituting rule-based replanning and maintaining task progress.

---

## Session 2 — 2026-06-20: New Attacks, Clean Baseline, and Visualization

### What We Built

This session added three new attack conditions (action_delay, action_reverse, action_clipping) to the benchmark, added a clean-condition recovery baseline to interpret recovery trigger rates, expanded all output scripts to cover all 8 conditions, and created three new visualization scripts with progressively focused method sets.

---

### Bug Fix: action_delay Step-0 None Guard

**Problem:** `episode_runner.py` initializes `previous_action = None`. The `action_delay` branch in `manipulate_action()` previously fell through to "return current action unchanged" when `previous_action is None` — meaning the delay attack had no effect at step 0, and CLAUDE.md had flagged it as a potential crash risk.

**Fix:** Added an explicit branch in `attacks/action_attacks.py`:
```python
elif attack_type == ATTACK_DELAY:
    if previous_action is None:
        executed = np.zeros_like(action)   # semantically: nothing to replay
    else:
        executed = np.asarray(previous_action, dtype=np.float32).copy()
```
Verified: `manipulate_action(a, "action_delay", previous_action=None)` returns zeros. `manipulate_action(a, "action_delay", previous_action=prev)` returns `prev`.

---

### New Attack: action_clipping

Added `ATTACK_CLIPPING` branch to `manipulate_action()` (accepts `clip_value` via `**kwargs`, default 0.30). Added `action_clipping` condition branch in `episode_runner.py`. Added to `ALL_CONDITIONS` in the sweep runner with `attack_level=0.30`.

**What it simulates:** An actuator saturation or command throttling attack — each motor command is clipped to ±0.30. The robot can still move but at reduced effective range.

---

### New Attack: action_reverse sweep

`action_reverse` was already implemented but had never been swept. Added to `ALL_CONDITIONS` and swept alongside `action_delay`.

---

### New: --include-clean-recovery Flag

Added `--include-clean-recovery` to `run_baseline_experiment.py`. When set, runs `_recovery` method variants on the `clean` condition in addition to attack conditions. Previously, recovery variants had no clean-condition baseline, making recovery trigger rates uninterpretable.

**Finding:** Recovery trigger rate on clean is very high (71–96% of steps) — the 3-signal detector fires spuriously without attacks, primarily driven by the distance trend signal (robot is still converging in early steps). This is an important calibration data point for the paper: the recovery system as designed needs a specificity adjustment before deployment.

---

### Sweeps Run (2026-06-20)

All sweeps: 4 methods × 3 seeds × 10 episodes, plus recovery variants.

| Sweep | Conditions | New rows |
|---|---|---|
| action_delay + action_reverse | 2 | 480 |
| action_clipping | 1 | 240 |
| clean (with --include-clean-recovery) | 1 | 240 |

**Canonical CSV now:** 1920 episode rows, 8 conditions, 0 duplicate (method, condition, seed, episode_idx) rows.

---

### Key New Results

**action_delay (base policies, 3.3% success):** The step-0 zeros effectively freeze the robot for one step; then the delay replays the previous action. For the proportional rule-based controller this means every action is one step behind, which is enough to prevent reaching within 50 steps most of the time. Recovery (which re-plans from scratch each step) restores 50% success.

**action_reverse (base policies, 0% success):** Negating all actions drives the robot away from the goal. Final distances are very large (>1.2 m). Recovery variants achieve 100% success — the 3-signal jerk detector fires immediately (reversed actions create massive step-to-step divergence) and rule-based replanning takes over entirely.

**action_clipping (base policies, 100% success for Rule-Based and SAC+HER):** Clipping to ±0.30 only reduces speed — the robot still reaches the goal, just more slowly. SAC (no HER) also achieves 100% success under clipping (its failure under clean conditions is due to sparse reward, not action magnitude requirements). Recovery trigger rate for `_recovery` variants is low (~7%) because the actions look smooth even if clipped.

---

### Script and Output Updates

#### `scripts/build_results_table.py`
`INCLUDE_CONDITIONS` expanded from 4 to 8 conditions. `PRETTY_CONDITION` dict updated. Output: `week5_results_table.csv` (64 rows, 9 cols).

#### `scripts/build_results_plot.py`
`COND_LABEL` and `COND_ORDER` expanded to all 8 conditions. Output: `week5_results_plot.png` (regenerated).

#### `scripts/build_focused_plot.py` (new)
3-panel figure showing SAC+HER and SAC (no HER) variants as the main methods, with Rule-Based shown as a slim reference column to the right of Panels A and B. Rationale: for the paper's primary comparison, the SAC vs. rule-based contrast should be visual but not give rule-based equal visual weight as an RL policy.

#### `scripts/build_split_plots.py` (new)
Generates four separate PNGs using pure matplotlib — one per panel. All 8 methods × 8 conditions shown. SAC (no HER) placed as an ablation block with a dashed separator after the main group. Bar layout: `bar_width=0.09`, `offsets = np.linspace(-3.5×bw, 3.5×bw, 8)`. Condition labels use `\n` line breaks and `rotation=0`. Panel D uses paired bars (equal vs. weighted) per method per condition.

#### `scripts/build_seaborn_plots.py` (new)
Four separate PNGs using seaborn. Went through three iterations:
1. All 8 methods
2. Filtered to 4 SAC methods only
3. Final: `FOCUS_METHODS = ["SAC+HER", "SAC+HER + Recovery"]` only

Final design decisions:
- **Panels A & B:** Single `figsize=(14,6)` axes. `fig.suptitle(..., y=1.08)` for the title; legend at `bbox_to_anchor=(0.5, 1.02)`. This separates title and legend vertically without overlap. No two-axes split — SAC (no HER) ablation block removed entirely.
- **Panel C:** SAC+HER + Recovery only. Single method, single color.
- **Panel D:** `FacetGrid` with `col="Method"`, 2 columns (SAC+HER and SAC+HER + Recovery). Each column shows equal vs. weighted bars per condition.
- **Technical fixes:** `ax.xaxis.set_major_locator(plt.FixedLocator(...))` before `set_xticklabels` (eliminates matplotlib FixedLocator warning). `.astype(str)` before Categorical reassignment in Panel D (eliminates Pandas4Warning from leftover categories). `errorbar=None` on all `sns.barplot()` calls (seaborn ≥0.12 API).

#### `readme.md`
Updated weights section: documents both equal (0.20 × 5) and argued (0.10/0.20/0.25/0.15/0.30) weight schemes, matching `metrics.py`. Updated recovery description from "0.5× damping" to "rule-based replanning using unattacked observation."

---

### What Still Needs to Be Done

1. **Overleaf draft — attack taxonomy section.** Full taxonomy is now complete: observation attacks (noise, dropout, bias) + goal attacks (spoof immediate, spoof mid-episode) + action attacks (noise, scale, delay, reverse, clipping). The 8-condition results table and plots are ready to use as figures.

2. **SOTA comparison.** Find and cite published numbers on FetchReach adversarial robustness. The sensor_dropout finding (rule-based: 100%, SAC+HER: 3%) and action_reverse recovery finding (0% → 100% with recovery) are the most publishable results.

3. **Recovery specificity.** Clean-condition recovery trigger rate of 71–96% means the detector fires constantly even without attacks. For the paper, this needs to be addressed: either raise thresholds, add a minimum-steps-since-last-trigger gate, or explicitly frame the current system as a "detect-and-correct" design with known false-positive cost.

4. **Habitat extension design.** Attack-to-Habitat mapping: sensor_dropout → camera failure; goal_spoof → semantic target corruption; sensor_bias → depth sensor drift; action_clipping → actuator saturation. Design note for Week 6.

5. **Notebook update.** `TAIRO_Week5_RL_Attack_SOTA_Habitat_Notebook_executed.ipynb` has not been re-run with the new 8-condition data. Either re-execute or replace with a clean notebook.
