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

The policy is trained on the equal-time comfort reward described in
[docs/reward.pdf](reward.pdf) and observes what [docs/observation.pdf](observation.pdf)
lists. In short: the Wk-weighted body acceleration is compared with a constant-speed
drive of the same duration on the same road, and the time lost against the scenario speed
is priced quadratically. The settings are in `config/reward/reward.yaml`,
`config/gym_env/env.yaml` and `config/preview/preview.yaml`.

The MPC reference has two ways of choosing the crossing speed of each bump, then
tracks it with acados:

- `comfort_map` (default): the speed that is best on a window around the bump under
  the MPC weights in `config/baseline/mpc.yaml` (`baseline.mpc.comfort_map`).
- `frontier`: the speed with the best return of the policy's own reward, found by
  replaying the road (`baseline.mpc.frontier_objective`, `--mpc-objective frontier`).

Both planners see the true bump positions and sizes of the road ahead
(`perception_mode: none`). The PPO does not use the MPC in any way.

## Training data

Real scenarios make up 60 % of each batch (`real_fraction` in
`config/curriculum/curriculum.yaml`); the rest are single synthetic bumps whose
height and width are sampled from the real range (`real_range` in
`config/train/scenario_gen.yaml`) and split into four severity tiers. The real part
is the 91 seen scenarios; the 14 held-out ones are never trained on. The
multi-bump and hole suites are not trained on.

## Scenarios

The real evaluation covers the 105 usable scenarios (not `low_confidence`) of the
CALM-Scenarios dataset. They are split by location: the held-out locations are the
German locations with fewer than five usable scenarios (sindelfingen, stuttgart) plus
one German and one Italian location drawn with seed 69 (augsburg, nardo), giving 20
scenarios. The rule was fixed before any result and is listed in `split.yaml` of the dataset.

Six of the 20 are used for training: on sindelfingen_sb01 the recorded human drive sped
up into the bump, and on augsburg_sb01, sb04, sb05, sb06 and sindelfingen_sb14 the bump
starts within 15 m of the recording start, so the approach speed cannot be read. The
evaluation set is the remaining 14 (`real`) against 91 seen in training (`real_train`);
`real_all` runs both.

The recorded human drive is placed on the trimmed road by the trim offset: the scenario
road starts `meta.trim.offset_m` metres into the recorded drive, the replay starts when
the recorded distance reaches that offset, and the scored span ends where the recording
ends. The IMU calibration was fitted on 13 scenarios; the human replay matches the IMU
to a median of 16 % over all 105 (86 within 30 %).

Synthetic roads (`01_single_bump`, `05_one_bump_one_hole`) run 40 m past the last
bump so the recovery is scored, and a flat road (`flat30`, `flat50`) checks that the
policy does nothing without a bump.

## Comparing at equal travel time

A trip-level a_w is a root mean square over the whole trip, so a slower drive spreads
the same bump over more time and scores lower. The raw a_w therefore flatters slow
driving. `a_w_vs_const_pct`, written for every row, is the a_w of a profile relative to a
constant-speed drive of the same duration over the same road (interpolated between
constant speeds). Negative is better. Roads are averaged with equal weight.

## Results

Calibrated scoring on the 105 real roads, change in a_w against a constant drive of
the same duration, mean over roads (negative is better). The MPC with the same
objective uses the frontier choice above. Scored over the full road, without the
span cut by the recorded drive.

| | all 105 | 14 held out | 91 seen | trip time (s) |
|---|---|---|---|---|
| Constant speed | −0.4 % | −0.2 % | −0.5 % | 9.3 |
| MPC, comfort map | −10.3 % | −7.7 % | −10.7 % | 10.5 |
| MPC, same objective | −21.7 % | −15.9 % | −22.6 % | 11.4 |
| PPO (seed 69) | −20.2 % | −22.4 % | −19.8 % | 12.6 |

Five training seeds (68 to 72) give −19.7 to −20.3 % on all roads and −21.8 to −23.3 %
on the held-out ones. On synthetic roads the PPO slows on 12 of 30 roads and gains about
1 % at equal time, where both MPC choices gain 25 to 35 %. Against the measured
acceleration the replay of the recorded drive has a median error of 15.7 %
(95 % interval 11.8 to 19.1), a mean signed error of +3.5 % and a rank correlation of 0.53.

## Limits

- Simulation only. Human values are model-based replays, not measurements, and no
  vehicle test has been run.
- The timing of the slowdown is built into the environment; the policy learns the depth only.
- The policy takes about 38 % longer than a constant drive, so this is a comfort versus
  time result. The time price was chosen with a scan that used some of the held-out roads.
- The policy does not transfer to the synthetic roads; the MPC does.
- The held-out set is 14 scenarios from 4 locations.
- Both planners see the true road ahead. A camera preview that gives only the distance to
  a marking hides the bump size, which selective slowing needs.
- One reference MPC and one dataset.
