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
  different speed. The a_w and equal-time columns are medians over scenarios; speed and time are means.
- Against the human: median over scenarios of the relative change of a_w, and the number of scenarios where
  the profile is more comfortable than the human. The human drove slower than the others (about 19 km/h),
  so this column also rewards speed choices; use the equal-time column for the comparison between planners.

## Real roads

Unseen (20):

| | a_w (m/s²) | speed (km/h) | time (s) | against the human (%) | lower than the human | against constant, equal time (%) |
|---|---|---|---|---|---|---|
| human | 0.276 | 18.0 | 9.5 |  |  | +2.5 |
| constant | 0.287 | 21.6 | 8.1 | -1.8 | 10/20 | -0.5 |
| MPC | 0.300 | 19.7 | 9.1 | -5.2 | 11/20 | -0.7 |
| PPO seed 69 | 0.261 | 19.8 | 10.1 | -6.6 | 11/20 | -1.0 |
| PPO seed 70 | 0.242 | 19.2 | 10.4 | -5.9 | 14/20 | -0.5 |
| PPO seed 71 | 0.245 | 19.1 | 10.7 | -10.5 | 13/20 | -0.6 |
| PPO seed 72 | 0.286 | 20.9 | 9.0 | -5.2 | 11/20 | -0.6 |
| PPO seed 73 | 0.239 | 18.4 | 11.7 | -15.3 | 14/20 | -0.7 |

Seen (85):

| | a_w (m/s²) | speed (km/h) | time (s) | against the human (%) | lower than the human | against constant, equal time (%) |
|---|---|---|---|---|---|---|
| human | 0.324 | 19.0 | 10.3 |  |  | +1.8 |
| constant | 0.311 | 21.6 | 8.9 | -5.9 | 53/85 | -0.5 |
| MPC | 0.269 | 19.4 | 10.1 | -15.2 | 60/85 | -0.9 |
| PPO seed 69 | 0.268 | 19.4 | 11.0 | -10.3 | 57/85 | -1.0 |
| PPO seed 70 | 0.261 | 18.5 | 12.3 | -16.3 | 62/85 | -0.6 |
| PPO seed 71 | 0.267 | 18.7 | 12.2 | -16.4 | 61/85 | -0.7 |
| PPO seed 72 | 0.271 | 19.6 | 11.1 | -11.9 | 56/85 | -0.7 |
| PPO seed 73 | 0.267 | 19.2 | 11.4 | -15.2 | 57/85 | -0.7 |

All real roads (105):

| | a_w (m/s²) | speed (km/h) | time (s) | against the human (%) | lower than the human | against constant, equal time (%) |
|---|---|---|---|---|---|---|
| human | 0.319 | 18.8 | 10.2 |  |  | +1.8 |
| constant | 0.303 | 21.6 | 8.8 | -5.7 | 63/105 | -0.5 |
| MPC | 0.272 | 19.5 | 9.9 | -12.8 | 71/105 | -0.8 |
| PPO seed 69 | 0.265 | 19.5 | 10.8 | -9.8 | 68/105 | -1.0 |
| PPO seed 70 | 0.257 | 18.7 | 11.9 | -15.9 | 76/105 | -0.6 |
| PPO seed 71 | 0.259 | 18.8 | 11.9 | -16.1 | 74/105 | -0.7 |
| PPO seed 72 | 0.275 | 19.8 | 10.7 | -10.6 | 67/105 | -0.7 |
| PPO seed 73 | 0.258 | 19.1 | 11.5 | -15.2 | 71/105 | -0.7 |

At equal time every seed is about 1 % more comfortable than a constant drive and within about 1 % of the MPC.
Against the human, every seed is more comfortable on the median road (-5 to -16 %), and about as many
roads improve as with the MPC.

## Replay against the IMU

The recorded human drive replayed in the model gives an a_w that is 17 % away from the
IMU value on the median road (81 % of the 105 roads within 30 %, mean bias +3 %).
The ranking of the controllers is the same with the replay and with the IMU as the human reference
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
