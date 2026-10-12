# Clean training results

Five PPO seeds (69 to 73) trained from scratch on the fixed code and scored on the 105 real scenarios against
the recorded human drive, a constant-speed drive and the MPC. Written 2026-10-12.

## Setup

- Training: `train.train --algo PPO --seed <seed>`, 10 M steps, half-car, the settings in
  `config/agent/algo_configs.yaml` (recommended-speed action, tanh policy mean, speed-weight controller
  with target 0.6, gae_lambda 0.995). Real roads are edge-trimmed (`config/gym_env/scenario_trim.yaml`).
- Scoring: `eval.comfort_score --ppo-model <run> --suites real_all`, with the recorded drives from
  `CALM_SEQUENCES`. The weighted acceleration a_w is the ISO 2631-1 Wk value of a replay on the calibrated
  scoring vehicle (`config/eval/comfort.yaml`). MPC at horizon 75, constant drive at the scenario speed.
- Real roads: 20 unseen scenarios (augsburg, nardo, sindelfingen, stuttgart, held out of training) and 85 seen.
- Equal-time measure: a_w of a profile relative to a constant drive covering the same trip in the same
  time, so slow driving is not rewarded. Negative is more comfortable than constant. The constant row is not
  exactly zero because its reference is a constant drive of the same duration, which runs at a slightly
  different speed. The a_w, equal-time and against-the-MPC columns are medians over scenarios; speed and time are means; the interval is a bootstrap of the median.
- Against the human: median over scenarios of the relative change of a_w, and the number of scenarios where
  the profile is more comfortable than the human. The human drove slower than the others (about 19 km/h),
  so this column also rewards speed choices; use the equal-time column for the comparison between planners.

## Real roads

Unseen (20):

| | median a_w (m/s²) [quartiles] | speed (km/h) | time (s) | against the human (%) [95 % interval] | lower than the human | against constant, equal time (%) | against the MPC (%) |
|---|---|---|---|---|---|---|---|
| human | 0.276 [0.220, 0.332] | 18.0 | 9.5 |  |  | +2.5 |  |
| constant | 0.287 [0.240, 0.347] | 21.6 | 8.1 | -1.8 [-14.9, +43.7] | 10/20 | -0.5 | +0.0 |
| MPC | 0.300 [0.223, 0.347] | 19.7 | 9.1 | -5.2 [-14.6, +8.0] | 11/20 | -0.7 |  |
| PPO seed 69 | 0.261 [0.216, 0.347] | 19.8 | 10.1 | -6.6 [-19.4, +43.2] | 11/20 | -1.0 | -0.2 |
| PPO seed 70 | 0.242 [0.187, 0.317] | 19.2 | 10.4 | -5.9 [-21.9, +0.7] | 14/20 | -0.5 | +0.1 |
| PPO seed 71 | 0.245 [0.187, 0.315] | 19.1 | 10.7 | -10.5 [-26.1, +2.1] | 13/20 | -0.6 | +0.0 |
| PPO seed 72 | 0.286 [0.230, 0.346] | 20.9 | 9.0 | -5.2 [-16.4, +43.7] | 11/20 | -0.6 | +0.0 |
| PPO seed 73 | 0.239 [0.168, 0.315] | 18.4 | 11.7 | -15.3 [-28.9, -0.7] | 14/20 | -0.7 | +0.0 |

Seen (85):

| | median a_w (m/s²) [quartiles] | speed (km/h) | time (s) | against the human (%) [95 % interval] | lower than the human | against constant, equal time (%) | against the MPC (%) |
|---|---|---|---|---|---|---|---|
| human | 0.324 [0.271, 0.384] | 19.0 | 10.3 |  |  | +1.8 |  |
| constant | 0.311 [0.245, 0.392] | 21.6 | 8.9 | -5.9 [-11.7, -0.6] | 53/85 | -0.5 | +0.0 |
| MPC | 0.269 [0.200, 0.369] | 19.4 | 10.1 | -15.2 [-17.9, -9.3] | 60/85 | -0.9 |  |
| PPO seed 69 | 0.268 [0.195, 0.347] | 19.4 | 11.0 | -10.3 [-18.0, -4.7] | 57/85 | -1.0 | -0.4 |
| PPO seed 70 | 0.261 [0.185, 0.324] | 18.5 | 12.3 | -16.3 [-22.8, -9.3] | 62/85 | -0.6 | +0.0 |
| PPO seed 71 | 0.267 [0.172, 0.328] | 18.7 | 12.2 | -16.4 [-21.0, -8.0] | 61/85 | -0.7 | +0.0 |
| PPO seed 72 | 0.271 [0.192, 0.348] | 19.6 | 11.1 | -11.9 [-17.9, -2.7] | 56/85 | -0.7 | +0.0 |
| PPO seed 73 | 0.267 [0.187, 0.347] | 19.2 | 11.4 | -15.2 [-19.5, -4.1] | 57/85 | -0.7 | +0.0 |

All real roads (105):

| | median a_w (m/s²) [quartiles] | speed (km/h) | time (s) | against the human (%) [95 % interval] | lower than the human | against constant, equal time (%) | against the MPC (%) |
|---|---|---|---|---|---|---|---|
| human | 0.319 [0.262, 0.379] | 18.8 | 10.2 |  |  | +1.8 |  |
| constant | 0.303 [0.244, 0.386] | 21.6 | 8.8 | -5.7 [-10.4, -0.3] | 63/105 | -0.5 | +0.0 |
| MPC | 0.272 [0.205, 0.359] | 19.5 | 9.9 | -12.8 [-16.4, -5.7] | 71/105 | -0.8 |  |
| PPO seed 69 | 0.265 [0.195, 0.347] | 19.5 | 10.8 | -9.8 [-17.8, -4.7] | 68/105 | -1.0 | -0.4 |
| PPO seed 70 | 0.257 [0.185, 0.324] | 18.7 | 11.9 | -15.9 [-20.7, -6.7] | 76/105 | -0.6 | +0.0 |
| PPO seed 71 | 0.259 [0.179, 0.328] | 18.8 | 11.9 | -16.1 [-20.2, -6.8] | 74/105 | -0.7 | +0.0 |
| PPO seed 72 | 0.275 [0.195, 0.347] | 19.8 | 10.7 | -10.6 [-16.4, -2.7] | 67/105 | -0.7 | +0.0 |
| PPO seed 73 | 0.258 [0.182, 0.347] | 19.1 | 11.5 | -15.2 [-19.2, -5.7] | 71/105 | -0.7 | +0.0 |

At equal time every seed is about 1 % more comfortable than a constant drive, and the median road differs from the MPC by at most 0.4 %.
Against the human, every seed is more comfortable on the median road (-5 to -16 %), and about as many
roads improve as with the MPC.

## Distance to the MPC frontier

The MPC run at several comfort weights and cruise speeds gives a comfort-against-time frontier per road.
a_w above that frontier at equal travel time, median over the roads inside the frontier's time range
(0 is on the frontier, positive is worse): human +15.4 % (interquartile range +2.6 to +29.3 %), MPC +0.0 %,
PPO seed 69 -0.3 % (-0.9 to +0.2 %), constant +0.0 %. The PPO profile sits on the frontier, like the MPC.

## Replay against the IMU

The recorded human drive replayed in the model gives an a_w that is 17 % away from the
IMU value on the median road (81 % of the 105 roads within 30 %, mean bias +3 %).
With the replay or the IMU as the human reference, constant, MPC and PPO (seed 69) are all more comfortable than
the human, but the order of MPC and PPO flips: median change -12.8 % (MPC) and -9.8 % (PPO) against the replay,
-12.3 % and -16.1 % against the IMU, constant -5.7 % and -2.4 %. That matches the tie between the two planners
(`fig_sim2real_gap` in the MATLAB folder).

## Chosen model

Rule: among the seeds that do not crawl (mean speed above 85 % of the constant drive on the unseen real
roads), the lowest median equal-time change on the 20 unseen roads.

Seed 69 (`logs/models/PPO/final_s69`, linked as `final_best`): -1.0 % against constant, -6.6 % against the
human on the unseen roads, 19.8 km/h against 21.6 for the constant drive, every trip completed. The other
seeds are -0.5 to -0.7 % at equal time on the unseen roads. Seed 73 is the most comfortable against the human
(-15.3 % unseen) because it drives slowest, so it ranks lower at equal time.

## Other checks

- Quarter-car: a 100 k-step training run, an eval on one scenario and on a suite, the MPC and the reactive
  driver all ran without errors on `--vehicle quarter`.
- The synthetic suites were scored in the same run but are not part of this study.
- Models and score files stay local: `CALM_final/logs/models/PPO/final_s69` to `final_s73` and
  `CALM_score/logs/comfort/runs/h_final_s69.csv` to `h_final_s73.csv`.

## Limits

- Differences between seeds and against the MPC are around 1 % at equal time, small compared with the spread
  between scenarios; treat the comparison between planners as a tie.
- The replay is within 30 % of the IMU on 81 % of the roads, so single-road conclusions carry that uncertainty.
