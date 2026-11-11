# Figures

Selected figures of the simulator results, grouped by topic. Each group has at most six figures. The road names follow the dataset, for example Nardò SB10 is `nardo_sb10`.

| Folder | Content |
|---|---|
| `comfort/` | Change of the weighted body acceleration at equal trip time, on all roads, and on the held-out roads; comfort against mean speed on two roads |
| `speed/` | Speed over distance of the recorded drive, the MPC, the PPO policy and the reactive driver on six real roads |
| `vibration/` | Power spectral density, vertical jerk and acceleration command of the MPC, the PPO policy and the reactive driver on two roads |
| `vehicle/` | Suspension travel and dynamic tire force against their limits; quarter-car and half-car response |
| `validation/` | Closeness of the simulated to the measured acceleration, and three recorded crossings against their replay |
| `dataset/` | Height and width of the bumps of the qualified crossings in Germany and Italy |

The weighted acceleration is the ISO 2631-1 Wk-weighted vertical body acceleration of the half-car model.
