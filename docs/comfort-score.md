# Comfort score

A single passenger-comfort number per scenario, comparable across speed
profiles. Lower is more comfortable.

## Metric

Vertical body acceleration is weighted with the ISO 2631-1 Wk filter
(`src/eval/iso2631.py`) and summarised as

- `a_w`: RMS of the weighted acceleration in m/s². Bands: below 0.315 not
  uncomfortable, 0.315–0.63 a little, 0.63–1.0 fairly, 1.0–1.6 uncomfortable,
  1.6–2.5 very, above 2.5 extremely.
- `VDV`: vibration dose value in m/s^1.75, sensitive to single shocks.

The Wk transfer function combines the ISO 2631-1 band limiting, the
acceleration-velocity transition and the upward step. The standard itself was not available while
implementing; the analytic response matches the tabulated one-third-octave
factors to within 0.2 % at all 24 bands and the digital filter matches the
analytic one.

## Speed profiles

Every profile is replayed open loop through the same standalone half-car model
(`HalfCarODE`, `DT` = 0.02 s) on the same road, so differences come from the
speed only.

| Source | Profile |
|---|---|
| `human` | recorded drive: distance from the wheel odometry in `wheel_excitement.json`, resampled to `DT`, smoothed with a 1 s Savitzky–Golay filter |
| `const` | the scenario speed held constant |
| `mpc` | MPC planner speed (`baseline.mpc`, acados) |
| `ppo` | PPO policy speed |

Real scenarios are scored over the road span the human actually covered.
Beyond the recording the human speed is unknown, so every source is cut at
that distance; a run counts as completed when it reaches it. Synthetic suites
have no human and run to the end of the road.

## Scoring vehicle

The nominal vehicle in `config/gym_env/env.yaml` overpredicted the recorded
IMU (median error 49 %). Scaling the spring rate by 1.06 and the damper by
0.16 (`config/eval/comfort.yaml`) reduces the error of the human replay to a
median of 14 % (11 of 13 scenarios within 30 %). Leave-one-scenario-out,
refitting on the other 12 each time, gives a median of 15 % and a mean of 26 %
(9 of 13 within 25 %).

This is an effective calibration of the scoring model, not a measured damping;
the low damper factor mostly absorbs model error in the nonlinear damper curve.
It is applied only when scoring, the controllers still plan on the nominal
vehicle. `--uncalibrated` scores on the nominal vehicle.

The scenario geometry matters as much as the vehicle. Against the IMU,
regenerating the real scenarios with the perception pipeline set to a bump
width limit of 8 m (was 4 m) and wheel-only height (was the larger of wheel
and LiDAR) cut the mean error from 30 % to 23 % before cross-validation. The
120 mm height limit has no effect. The regenerated files are local; the
published dataset still holds the old geometry.

Known limits:

- `ludwigsburg_sb01` and `stuttgart_sb01` are still off by 70–100 % in
  leave-one-out.
- One calibration is shared by all 13 scenarios, German and Italian.
- Human values are model-based estimates over the recorded span, not direct
  measurements.

## Running

```bash
export CALM_SEQUENCES=/path/to/CALM-Germany/sequences:/path/to/CALM-Italy/sequences
just compare <run-name>                    # PPO_final vs MPC vs human, all suites
just compare <run-name> best_real/best_model.zip   # another checkpoint of the run
just compare <run-name> best_real/best_model.zip --suites real
```

`CALM_SEQUENCES` points at the recorded windows and is only needed for the
human profile. MPC needs acados. The command prints a summary table and writes

- `logs/comfort/comfort_scores.csv`: one row per scenario and source with
  `a_w`, `vdv`, `pitch_rms`, `peak_a_w`, speeds, travel time, `completed`,
  ISO band, `a_w_imu` (measured, human only) and `padded_frac`
- `logs/comfort/profiles/`: speed and acceleration time series and the road
  profile per scenario, for plotting

The 10_potholes suite is not part of the default run. `best_real` is the best
checkpoint on a sample of 20 training-pool real scenarios (not the held-out ones)
and can stall on synthetic ones; check the `done` column of the summary. The MPC
uses the horizon `N` in `config/baseline/mpc.yaml` unless `--horizon` is given.

## Figures and tables

`just sync` copies the CSV and profiles into the folder named by
`MATLAB_FIGURES` in `.env` (see `.env.example`). The MATLAB scripts
`fig_comfort_score` and `fig_comfort_real` read them and are configured by
`comfort_config.m` (sources, suites, ignored scenarios, which figures and LaTeX
tables to write).

Several trainings are compared with `just export-runs`: it reads the per-run
score files in `logs/comfort/runs/` listed in `config/eval/runs.yaml` (one entry
per training, with seed, target and withheld locations) and writes
`ppo_runs.csv`, `ppo_runs_scenarios.csv` and `scenarios.csv`, which `just sync`
copies too. `fig_comfort_research` draws from them the comfort-time frontier,
the gain against bump geometry, RL against MPC per scenario, the operating point
of every training, the leave-location-out comparison, the speed-comfort
relation per bump width and the agreement with the measured acceleration.

## Planner objectives

Both planners use the same form of objective: the comfort terms (body heave,
wheel, pitch), longitudinal acceleration and jerk, and one speed term
`Q_v * track_scale * |v - v_nom| / v_nom`, with `Q_step` 0. Their weights are
configured separately, so tuning one never changes the other.

- PPO: `config/reward/reward.yaml` is the training reward and the environment
  limits; `track_scale` there is replaced during training by the speed-weight
  controller described below.
- MPC: the `weights` and `velocity` blocks of `config/baseline/mpc.yaml`
  override the reward configuration for the MPC only (comfort map, tracking
  cost, and the environment the MPC is evaluated in).

They differ in how the objective is solved.

- MPC picks a crossing speed per bump from a comfort map (the same objective
  evaluated on a bump window, `baseline.mpc.comfort_map`) and tracks it with
  acados; it cruises at the scenario speed between bumps.
- PPO learns a policy from the per-step reward. It does not use the MPC, its
  comfort map or its speeds in any way. Both see the true bump positions and
  sizes of the road ahead (`perception_mode: none` in `config/preview/`), so
  they have the same information.

Speed is limited to at least 8 km/h for both (`velocity.v_min` and
`velocity.v_hard_min`, a clamp in the environment). Episodes are 80 s so slow
comfortable driving can finish the longest roads.

### MPC tuning

The MPC was tuned on its own (the PPO configuration is untouched). On the 13
real scenarios, with the human at a_w 0.289, 14.6 km/h and 9.3 s:

| comfort map | `track_scale` | a_w | speed (km/h) | time (s) | lower a_w than human |
|---|---|---|---|---|---|
| nominal vehicle | 0.25 | 0.281 | 16.4 | 8.1 | 5 of 13 |
| nominal vehicle | 0.1 | 0.269 | 15.8 | 8.4 | 8 of 13 |
| nominal vehicle | 0.02 | 0.265 | 15.6 | 8.9 | 9 of 13 |
| calibrated vehicle | 0.25 | 0.288 | 20.9 | 7.4 | 9 of 13 |
| calibrated vehicle (used) | 0.05 | 0.269 | 18.6 | 7.4 | 9 of 13 |

The planning horizon `N` (steps of 0.02 s) was swept over all 105 real scenarios:
the median a_w against the human is −11.5 % at 20 steps, −12.5 % at 30, −13.3 % at
50 and −13.7 % at 75, 100 and 150, with the same share of scenarios below the human
from 75 steps on (70 of 105; 69 at 150). The braking ramp towards a bump is about 1
to 1.5 s long, so 75 steps (1.5 s) are enough and are used; a longer horizon only
costs solve time. Weighting the
comfort map with the ISO filter instead of plain squared acceleration makes
little difference. Giving the comfort map the calibrated vehicle (the same
spring and damper scales as the scoring vehicle, set separately under
`comfort_map.vehicle` in `config/baseline/mpc.yaml`) lets the planner cross
bumps faster at the same comfort. Below `track_scale` 0.05 the planner saturates
at a_w of about 0.265: it cruises at the scenario speed between bumps and the
8 km/h minimum stops it matching the human's crawls.

The MPC reads all its weights from `config/baseline/mpc.yaml`, none from the PPO's
`reward.yaml`. The `weights` block holds the comfort-map terms (body, wheel and pitch
acceleration, speed tracking scaled by `track_scale`, step cost). The tracking
problem solved by acados has its own block `ocp`: `Q_speed` weights the speed error,
`R_accel` the acceleration command, `S_jerk` its rate of change, and `Q_terminal` the
speed error at the end of the horizon. Comfort enters only through the comfort map,
which picks the crossing speed for each bump; the tracking problem follows it.

### PPO training

A plain PPO on this objective collapsed to one of two extremes, crawling at the
8 km/h minimum or holding the scenario speed, and never slowed selectively for
bumps. The final setup adds six things, each in response to a diagnosed
failure.

1. The action is a recommended slowdown: 0 holds the scenario speed and -1
   goes down to the 8 km/h minimum, tracked by a follower (gain 2 per second, at most 5 m/s²)
   (`action_mode: slowdown` in `config/gym_env/env.yaml`). The speed can
   never exceed the scenario speed.
2. The mean of the Gaussian policy is bounded by a tanh
   (`src/train/policy.py`), and actions that fall outside the allowed range are
   penalised (`action_saturation_penalty`). With an unbounded mean the policy
   sat beyond the limits at every step (mean action between −2.5 and −4.3),
   where every sample is clipped to the same action and no learning signal
   remains.
3. Time is priced directly: every step costs `time_cost * DT` (`time_cost` in
   `config/reward/reward.yaml`, in reward units per second), so slowing down
   costs reward in proportion to the delay and speeding up again is rewarded.
   `time_cost` moves the policy along the comfort/time curve.
4. `gae_lambda` is 0.995. At a 50 Hz control rate the usual 0.95 gives a credit
   horizon of about 0.4 s, while braking has to start 1–2 s before a bump.
5. The preview gives time to the bump and its excitation frequency instead of
   raw distance (`use_dist_obs: false`).
6. Synthetic approach roads are 10–20 m long instead of 30–50 m
   (`flat_start_range`), so most training steps are the relevant approach to a
   bump.

Without the third and fourth points the policy slowed only after the bump
had passed (its braking began after the bump centre); with them the speed
profile depends on bump size. Variants that did not work are listed under
Results.

## Training data

Real scenarios make up 75 % of each batch (`real_fraction` in
`config/curriculum/curriculum.yaml`); the rest are single synthetic bumps whose
height (30–120 mm) and width (1.5–7 m) are sampled from the real range
(`real_range` in `config/train/scenario_gen.yaml`) with 10–20 m approaches and
split into four severity tiers. The real part is the 91 seen scenarios; the 14
held-out ones (see below) are never trained on. The synthetic multi-bump and
hole suites are not trained on.

## Scenarios

The real evaluation covers the 105 usable scenarios (not `low_confidence`) of
the CALM-Scenarios dataset, regenerated with the perception pipeline set to the
8 m width limit and wheel-only height described above. They are split by
location: the held-out locations are the German locations with fewer than five
usable scenarios (sindelfingen, stuttgart) plus one German and one Italian
location drawn with seed 69 (augsburg, nardo), giving 20 scenarios in held-out
locations. The rule was fixed before any result and is listed in `split.yaml` of
the dataset.

The recorded human drive sped up into the bump on one of the 20 (sindelfingen_sb01), and on five more
(augsburg_sb01, sb04, sb05, sb06 and sindelfingen_sb14) the bump starts within 15 m of the
recording start, so the approach speed cannot be read. Those six are used for training. The
evaluation set is the remaining 14 (`real`) against 91 seen in training (`real_train`); `real_all` runs
both. `paper_split` in `split.yaml` lists the 14, and the dataset puller applies it. Seen and unseen are
reported separately where it matters.

The results tables below were computed on the earlier 20-scenario held-out set and
have not been recomputed for the 14.

The IMU calibration was fitted on 13 scenarios; the human replay matches the
IMU to a median of 16 % over all 105 (86 within 30 %).

## Comparing at equal travel time

A trip-level a_w is a root mean square over the whole trip, so a slower drive
spreads the same bump over more time and scores lower: in the scoring model a
constant 8 km/h drive is 34–80 % below 21.6 km/h in every real scenario. The raw
a_w therefore flatters slow driving. Two equal-time measures remove this.

- `a_w_vs_const_pct`, written for every row: the a_w of a profile relative to a
  constant-speed drive of the same duration over the same road (interpolated
  between constant speeds of 8–30 km/h). Negative is better.
- The distance to the MPC frontier (`just frontier build`, then
  `just frontier excess`): the lowest a_w the MPC reaches at each travel time,
  from several comfort weights and cruise speeds down to 8 km/h. Each profile is
  compared with that curve at its own travel time. 0 means on the frontier,
  positive means worse; points outside the time range the frontier covers are
  left out. The MPC is a model-based planner with perfect road knowledge, so it
  is the reference for what is reachable, not a proof of optimality.

## Results

Calibrated scoring, medians over the 105 real scenarios. MPC is the tuned
configuration. PPO is the final setup described above, trained three times with
different seeds (69, 70, 71; 10 M steps each, target speed ratio 0.6). Intervals
are 95 % bootstrap intervals of the median per-scenario change in a_w against
the human; p-values are exact or normal-approximation signed-rank tests.

| | a_w | VDV | speed (km/h) | time (s) | a_w vs human | lower than human | time vs human | a_w vs constant at equal time |
|---|---|---|---|---|---|---|---|---|
| Human | 0.321 | 0.960 | 17.6 | 10.0 | – | – | – | +2.0 % |
| Constant | 0.301 | 0.968 | 21.6 | 8.5 | −7 % | 61 of 105 (p = 0.44) | −18 % | 0 % |
| MPC | 0.279 | 0.896 | 20.5 | 9.5 | −13 % (−17 to −5) | 69 of 105 (p = 0.003) | −9 % | −1.0 % |
| PPO, seed 69 | 0.264 | 0.874 | 21.6 | 9.7 | −15 % (−18 to −5) | 70 of 105 (p < 0.001) | −10 % | −0.7 % |
| PPO, seed 70 | 0.202 | 0.731 | 13.9 | 13.8 | −28 % (−39 to −19) | 85 of 105 (p < 10⁻⁸) | +33 % | −3.4 % |
| PPO, seed 71 | 0.267 | 0.874 | 21.6 | 9.5 | −15 % (−18 to −7) | 72 of 105 (p < 0.001) | −7 % | −0.8 % |

All three PPO trainings beat the human on comfort. Two reach the same
operating point as the MPC (about the scenario speed, 7–10 % shorter trips than
the human); the third settles lower and slower (28 % less a_w, 33 % longer
trips). Against the MPC reference, the PPO is at the frontier on all three
seeds: the distance to the MPC frontier at equal travel time has a median of
+0.0 % (seeds 69 and 71, on or below it in 94 of 99 and 95 of 98 comparable
scenarios) and −0.0 % (seed 70, 85 of 103), where the human is +13 %. Against
the MPC at the same operating point, seeds 69 and 71 differ by +0.0 % in median
a_w (interquartile range −2.3 % to +0.7 % and −2.9 % to +0.2 %, p = 0.43 and
0.17), so the independently trained policy matches the model-based planner.

On the 20 held-out scenarios (never trained on) the a_w change against the
human is −5 % for the MPC and −13 %, −25 % and −10 % for the three PPO seeds,
with trip time +3 %, +68 % and +3 % for the PPO seeds. The intervals are wide
and only seed 70 excludes zero (p = 0.47, 0.09 and 0.5 for the three seeds, so
none is significant on 20 scenarios). At equal time the PPO is 0.5–1.8 % below
the constant-speed line on the held-out scenarios and 0.7–3.8 % on the seen
ones. Twenty scenarios cannot rule out a generalization gap but show none.

Where the gains are (seed 69; medians, a_w against a constant drive of the same
duration):

| bump | n | human | MPC | PPO |
|---|---|---|---|---|
| narrow (≤ 2 m) | 46 | +0.6 % | −0.5 % | −0.5 % |
| medium (2–4.5 m) | 35 | +3.1 % | −21.8 % | −0.7 % |
| wide (> 4.5 m) | 24 | +2.6 % | −11.6 % | −1.4 % |
| tall (> 70 mm) | 10 | +5.9 % | −34.2 % | −20.4 % |

Recommended speed matters only for wider or taller bumps. On narrow bumps
neither planner slows and nothing is lost: speed barely changes a_w there. The
MPC gains most on medium and wide bumps by crossing them slowly; the PPO's
learned rule is cruder (it crawls the whole road when the bump is severe,
reaching a_w 0.130 on the tall bumps against 0.420 for the human, and
otherwise holds the scenario speed).

On the synthetic suites, which have two to four bumps and holes and were not
trained on, the MPC is 12–22 % below a constant drive of the same duration;
the PPO is at 0 %. The learned policy does not transfer to multi-bump roads.

### Validation

**Seeds.** Five trainings of the final setup (seeds 69 to 73, 10 M steps each,
speed-weight target 0.6). Four settle at the scenario speed (21.4–21.6 km/h,
9.0–9.8 s) and reduce a_w by 13 to 15 % against the human (95 % intervals all
below zero, p between 0.0002 and 0.02). Seed 70 settles slower (13.9 km/h,
13.8 s) and reduces it by 28 %. At equal time all five are within 0.3 to
3.4 % of the constant-speed line.

**Speed-weight target.** Targets 0.5 and 0.7 with seed 69 both ended at the
scenario speed (21.6 km/h, 8.5 s, a_w 0.300 and 0.301), which is the
constant-speed drive (0.301) and only 6 to 7 % below the human (p = 0.38 and
0.39). The gain at 0.6 is therefore not a smooth response to the target:
the final speed is either the scenario speed or a slow mode, and which one a
training reaches is partly down to chance.

**Leave-location-out.** Three further trainings each withhold whole locations
(A: taufkirchen and salice_salentino, 27 scenarios; B: munich and ludwigsburg,
25; C: ingolstadt, carmiano and monteroni_di_lecce, 33). Median a_w against the
human on the withheld scenarios, with the MPC and the seed-69 policy that did
train on them for reference:

| fold | MPC | seed 69 (trained on them) | withheld |
|---|---|---|---|
| A | −8 % | −10 % | −9 % |
| B | −13 % | −7 % | −61 % (crawled at 8.4 km/h, 18.4 s trips; −9.1 % against constant at equal time, MPC −9.2 %) |
| C | −18 % | −18 % | −14 % |

Folds A and C show no gap beyond what the intervals (about ±10 points)
allow. In fold B the withheld policy fell into the slow mode, so its a_w is
not comparable; at equal time it is on the frontier like the MPC. Of the eight
trainings of the final setup at target 0.6 (five seeds and three folds), two
crawled or settled slow (seed 70, fold B).

**Camera preview.** A policy trained and evaluated under the camera-noise
perception model (`perception_mode: model`; marking distance only, bump size
hidden until about 0.5 m) holds the scenario speed: a_w 0.301, 8.5 s, −7 %
against the human (p = 0.44), identical to a constant drive. Selective
slowing needs the bump size ahead of the bump, which the camera does not give.

**Measured acceleration.** On the 105 scenarios with IMU data the ranking
holds when a_w is compared with the measured value instead of the replay.
Median change against the human: PPO −14.4 % (p = 0.002; −15.2 % against the
replay), MPC −9.3 % (p = 0.06; −12.6 %), constant −1.0 % (p = 0.95; −6.6 %).

### What did not work

All trained for 10 M steps with the same curriculum.

| variant | outcome |
|---|---|
| fixed speed tracking weight, calibrated or nominal vehicle, old or corrected scenario geometry | mostly crawled at the 8 km/h minimum; the exception (11 km/h) had seen the held-out locations in training |
| constant time cost per step, 0.02 and 0.05 | crawls (0.02); holds the scenario speed and ignores bumps (0.05) |
| speed weight annealed from 1.0 to 0.1 | holds the scenario speed |
| soft penalty below 60 % of the scenario speed, on top of the final setup (seed 70) | no crawl, but the policy holds the scenario speed (a_w 8 % below the human, 21.6 km/h) instead of the slower useful mode |
| recommended-speed action alone | holds the scenario speed |
| speed-weight controller with an unbounded policy mean | weight drops to 0, policy stuck at the speed cap |
| bounded mean, controller (target 0.8) | right average speed but brakes after the bump; about 0 % against constant at equal time |
| lower longitudinal penalties (÷10) | no change |
| final setup, target 0.8 | holds the scenario speed on most roads, crawls on severe ones |

Replaying the MPC and constant-speed profiles under the training reward scored them above a crawl, so the crawl was an optimisation failure, not what the reward asked for.

### Limits

- One reference MPC and one dataset. Human values are model-based replays
  (16 % median error against the IMU), not measurements.
- The PPO sees the true road ahead, as the MPC does. With the camera
  preview, which gives only the distance to a marking (bump height and width
  are hidden until about 0.5 m), selective slowing by bump size is not possible
  and a policy can only slow near markings.
- The held-out set is 14 scenarios from 4 locations (20 before six were moved to training, see Scenarios).
- The operating point is not controlled by the speed-weight target: targets 0.5
  and 0.7 hold the scenario speed, 0.6 does so in most trainings and settles
  slow in some (2 of 8). Results are reported at equal travel time for that
  reason.
- At equal time the PPO is within 1 % of a constant drive on most seeds. Its
  comfort gain over the human comes mainly from average speed and from
  slowing for tall bumps, not from a learned speed profile that is better than
  a cruise.
