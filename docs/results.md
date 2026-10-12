# Clean training results

Five PPO seeds (69 to 73) trained from scratch on the fixed code, scored on all 105 real scenarios
(and the bundled synthetic suites, not reported), against a constant-speed drive and the MPC. Written 2026-10-12.

## Setup

- Training: `train.train --algo PPO --seed <seed>`, 10 M steps, half-car, the settings in
  `config/agent/algo_configs.yaml` (recommended-speed action, tanh policy mean, speed-weight controller
  with target 0.6, gae_lambda 0.995). Real roads are edge-trimmed (`config/gym_env/scenario_trim.yaml`).
- Scoring: `eval.comfort_score --ppo-model <run> --suites <all suites> real_all --allow-missing-human`.
  The weighted acceleration a_w is the ISO 2631-1 Wk value of a replay on the calibrated scoring vehicle
  (`config/eval/comfort.yaml`). MPC at horizon 75, constant drive at the scenario speed.
- Real roads: 20 unseen scenarios (augsburg, nardo, sindelfingen, stuttgart, held out of training) and 85 seen.
- Equal-time measure: a_w of a profile relative to a constant drive covering the same trip in the same
  time, so slow driving is not rewarded. Negative is more comfortable than constant. The constant row is not exactly zero because its
  reference is a constant drive of the same duration, which runs at a slightly different speed. The
  equal-time and a_w columns are medians over scenarios; speed and time are means.
- No comparison with the recorded human drive here: the recordings were not available on this machine.

## Real roads

Unseen (20):

| | a_w (m/s²) | speed (km/h) | time (s) | vs constant, equal time (%) | lower than constant (%) | completed (%) |
|---|---|---|---|---|---|---|
| constant | 0.281 | 21.6 | 8.7 | -0.5 | 80 | 100 |
| MPC | 0.295 | 19.7 | 9.7 | -0.8 | 85 | 100 |
| PPO seed 69 | 0.257 | 19.8 | 10.7 | -1.2 | 85 | 100 |
| PPO seed 70 | 0.237 | 19.2 | 11.1 | -0.5 | 70 | 100 |
| PPO seed 71 | 0.240 | 19.1 | 11.3 | -0.6 | 85 | 100 |
| PPO seed 72 | 0.280 | 20.9 | 9.6 | -0.6 | 85 | 100 |
| PPO seed 73 | 0.234 | 18.4 | 12.4 | -0.7 | 90 | 100 |

Seen (85):

| | a_w (m/s²) | speed (km/h) | time (s) | vs constant, equal time (%) | lower than constant (%) | completed (%) |
|---|---|---|---|---|---|---|
| constant | 0.296 | 21.6 | 9.4 | -0.5 | 74 | 100 |
| MPC | 0.262 | 19.5 | 10.7 | -1.1 | 80 | 100 |
| PPO seed 69 | 0.262 | 19.4 | 11.6 | -1.1 | 79 | 100 |
| PPO seed 70 | 0.249 | 18.5 | 13.0 | -0.7 | 74 | 100 |
| PPO seed 71 | 0.255 | 18.7 | 12.9 | -0.7 | 84 | 100 |
| PPO seed 72 | 0.264 | 19.6 | 11.7 | -0.8 | 84 | 100 |
| PPO seed 73 | 0.252 | 19.2 | 12.1 | -0.6 | 84 | 100 |

All real roads (105):

| | a_w (m/s²) | speed (km/h) | time (s) | vs constant, equal time (%) | lower than constant (%) | completed (%) |
|---|---|---|---|---|---|---|
| constant | 0.290 | 21.6 | 9.3 | -0.5 | 75 | 100 |
| MPC | 0.264 | 19.5 | 10.5 | -1.0 | 81 | 100 |
| PPO seed 69 | 0.258 | 19.5 | 11.5 | -1.2 | 80 | 100 |
| PPO seed 70 | 0.246 | 18.6 | 12.6 | -0.6 | 73 | 100 |
| PPO seed 71 | 0.252 | 18.8 | 12.6 | -0.7 | 84 | 100 |
| PPO seed 72 | 0.267 | 19.8 | 11.3 | -0.7 | 84 | 100 |
| PPO seed 73 | 0.249 | 19.1 | 12.1 | -0.7 | 85 | 100 |

At equal time every seed is about 1 % more comfortable than a constant drive (bootstrap interval of the
median below zero for all five), and within about 1 % of the MPC. The raw a_w column is lower for the
PPO seeds because they drive slower on average (18 to 21 km/h against 21.6), which the equal-time measure removes.

## Synthetic suites

Not part of the paper's scope (the study is on real roads), so no tables here. They were scored in the
same run (180 scenarios, per-seed files above). In short, the MPC gains 11 to 19 % at equal time on the
multi-bump suites and the PPO seeds show no gain there, because the PPO is trained on real roads.

## Chosen model

Rule: among the seeds that do not crawl (mean speed above 85 % of the constant drive on the unseen real roads),
the lowest median equal-time change on the 20 unseen real roads.

Seed 69 (`logs/models/PPO/final_s69`): -1.2 % against constant (interval -2.25 to -0.75), -0.7 % against the
MPC (interval -1.55 to +0.50, so not distinguishable), 100 % of trips completed, 19.8 km/h against 21.6.
The other seeds are -0.5 to -0.7 % on the unseen roads.

## Other checks

- Quarter-car: a 100 k-step training run, an eval on one scenario and on a suite, the MPC and the reactive
  driver all ran without errors on `--vehicle quarter`.
- Trained models and the per-seed score files are under `CALM_final/logs/models/PPO/final_s69` to `final_s73`
  and `CALM_score/logs/comfort/runs/final_s69.csv` to `final_s73.csv` (local, not committed).

## Limits

- Differences between seeds and against the MPC on real roads are around 1 %, small compared with the
  spread between scenarios; treat the real-road result as a tie, not a win.
- No human-drive comparison in this run.
