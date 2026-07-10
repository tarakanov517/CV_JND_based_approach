| Группа     | Эксперимент                            |                      Activation σ |        Parameter σ | Clean acc | Val loss | FGSM 8/255 | PGD 8/255 |
| ---------- | -------------------------------------- | --------------------------------: | -----------------: | --------: | -------: | ---------: | --------: |
| Activation | Baseline                               |                         `(0,0,0)` |                  — |    81.29% |    0.541 |          — |         — |
| Activation | Early                                  |                      `(0.03,0,0)` |                  — |    81.74% |    0.550 |          — |         — |
| Activation | Late                                   |                      `(0,0.03,0)` |                  — |    82.96% |    0.506 |          — |         — |
| Activation | Classifier                             |                      `(0,0,0.03)` |                  — |    83.19% |    0.505 |          — |         — |
| Activation | All                                    |                `(0.03,0.03,0.03)` |                  — |    82.52% |    0.542 |          — |         — |
| Activation | Decreasing                             |                `(0.05,0.03,0.01)` |                  — |    82.01% |    0.549 |          — |         — |
| Activation | Increasing                             |                `(0.01,0.03,0.05)` |                  — |    83.13% |    0.512 |          — |         — |
| Activation | Classifier small                       |                      `(0,0,0.01)` |                  — |    82.88% |    0.507 |          — |         — |
| Activation | Classifier larger                      |                      `(0,0,0.05)` |                  — |    82.57% |    0.516 |          — |         — |
| Parameter  | All small-med                          |                         `(0,0,0)` | `(1e-3,1e-3,1e-3)` |    81.83% |    0.539 |          — |         — |
| Parameter  | Early                                  |                         `(0,0,0)` |       `(1e-3,0,0)` |    83.09% |    0.502 |          — |         — |
| Parameter  | Late                                   |                         `(0,0,0)` |       `(0,1e-3,0)` |    82.45% |    0.534 |          — |         — |
| Parameter  | Classifier                             |                         `(0,0,0)` |       `(0,0,1e-3)` |    83.53% |    0.489 |          — |         — |
| Parameter  | All large                              |                         `(0,0,0)` | `(1e-2,1e-2,1e-2)` |    73.83% |    0.755 |          — |         — |
| Parameter  | Early large                            |                         `(0,0,0)` |       `(1e-2,0,0)` |    81.24% |    0.559 |          — |         — |
| Parameter  | Late large                             |                         `(0,0,0)` |       `(0,1e-2,0)` |    78.77% |    0.633 |          — |         — |
| Parameter  | Classifier large                       |                         `(0,0,0)` |       `(0,0,1e-2)` |    78.23% |    0.636 |          — |         — |
| Parameter  | Classifier small                       |                         `(0,0,0)` |       `(0,0,1e-4)` |    82.81% |    0.498 |          — |         — |
| Parameter  | All small                              |                         `(0,0,0)` | `(1e-4,1e-4,1e-4)` |    82.21% |    0.525 |          — |         — |
| Combined   | Classifier act + classifier param      |                      `(0,0,0.03)` |       `(0,0,1e-3)` |    81.48% |    0.542 |          — |         — |
| Combined   | Weak classifier act + classifier param |                      `(0,0,0.01)` |       `(0,0,1e-3)` |    82.77% |    0.509 |          — |         — |
| Combined   | Increasing act + classifier param      |                `(0.01,0.03,0.05)` |       `(0,0,1e-3)` |    83.17% |    0.509 |          — |         — |
| Combined   | All act + weak all param               |                `(0.03,0.03,0.03)` | `(1e-4,1e-4,1e-4)` |    83.37% |    0.507 |          — |         — |
| Combined   | All act + all param                    |                `(0.03,0.03,0.03)` | `(1e-3,1e-3,1e-3)` |    82.12% |    0.539 |          — |         — |
| Retest     | Baseline                               |                         `(0,0,0)` |          `(0,0,0)` |    82.41% |    0.524 |          — |         — |
| Retest     | Activation classifier                  |                      `(0,0,0.03)` |          `(0,0,0)` |    82.32% |    0.530 |          — |         — |
| Retest     | Parameter classifier                   |                         `(0,0,0)` |       `(0,0,1e-3)` |    83.18% |    0.506 |          — |         — |
| Retest     | Combined                               |                `(0.03,0.03,0.03)` | `(1e-4,1e-4,1e-4)` |    82.63% |    0.508 |          — |         — |
| No dropout | Baseline                               |                         `(0,0,0)` |          `(0,0,0)` |    84.04% |    0.490 |          — |         — |
| No dropout | Activation classifier                  |                      `(0,0,0.03)` |          `(0,0,0)` |    83.68% |    0.500 |          — |         — |
| No dropout | Parameter classifier                   |                         `(0,0,0)` |       `(0,0,1e-3)` |    83.86% |    0.480 |          — |         — |
| No dropout | Combined                               |                `(0.03,0.03,0.03)` | `(1e-4,1e-4,1e-4)` |    84.56% |    0.476 |          — |         — |
| Attacks    | Baseline                               |                         `(0,0,0)` |          `(0,0,0)` |    84.04% |    0.490 |     19.02% |     0.55% |
| Attacks    | Activation classifier                  |                      `(0,0,0.03)` |          `(0,0,0)` |    84.28% |    0.487 |     18.66% |     0.96% |
| Attacks    | Parameter classifier                   |                         `(0,0,0)` |       `(0,0,1e-3)` |    84.64% |    0.463 |     18.54% |     0.74% |
| Attacks    | Combined                               |                `(0.03,0.03,0.03)` | `(1e-4,1e-4,1e-4)` |    84.59% |    0.467 |     19.05% |     0.92% |
| Группа    | Эксперимент        | Параметры                                   | Clean acc | Val loss | FGSM 8/255 | PGD 8/255 |
| Bio-noise | Bio baseline       | `sigma=0, sigma_prop=0, sigma_m=0`          |    83.68% |    0.507 |     18.27% |     0.65% |
| Bio-noise | Lateral inhibition | `sigma=0.03`                                |    83.85% |    0.517 |     20.52% |     0.90% |
| Bio-noise | Contrast adaptive  | `sigma_prop=0.03, sigma_add=0.01`           |    85.41% |    0.450 |     20.88% |     0.88% |
| Bio-noise | Magno/Parvo        | `sigma_m=0.03, ratio_m=0.1`                 |    84.41% |    0.476 |     17.04% |     0.65% |
| Bio-noise | All bio-noises     | `sigma=0.03, sigma_prop=0.03, sigma_m=0.03` |    83.99% |    0.513 |     19.06% |     0.94% |
| Bio-noise | Strong Magno/Parvo | `sigma_m=0.05, ratio_m=0.1`                 |    84.21% |    0.496 |     19.54% |     0.66% |
