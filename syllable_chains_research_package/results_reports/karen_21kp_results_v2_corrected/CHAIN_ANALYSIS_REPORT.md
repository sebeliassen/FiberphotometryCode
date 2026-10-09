# Project B: Syllable Chain Dynamics Report — Karen

**Cohort:** Karen  
**Sessions Analyzed:** 110  
**Total Chain Executions Detected:** 76,542  

## 1. Canonical Behavioral Chains Analyzed

| Chain ID | Name | Category | Syllables | Description |
|---|---|---|---|---|
| `K1_Rest_Micro_Rest` | Rest -> Micro -> Rest | Quiescence/Micro | `syll_29 -> syll_01 -> syll_29` | Low-speed rest -> small head/limb micro-movement -> rest |
| `K2_Rest_Shift_Rest` | Rest -> Shift -> Rest | Quiescence/Shift | `syll_29 -> syll_30 -> syll_29` | Stationary rest -> posture shift -> stationary rest |
| `K3_Dash_Reorient` | Dash -> Reorient -> Dash | Locomotion/Darting | `syll_15 -> syll_25 -> syll_15` | Rapid dash forward -> directional reorient -> repeat dash |
| `K4_Turn_Loop` | Turn Left <-> Turn Right | Steering/Turning | `syll_56 -> syll_58 -> syll_56` | Steering turn -> angular counter-turn -> re-align |
| `K5_Sprint_Burst` | Sprint Burst Loop | Locomotion | `syll_15 -> syll_53 -> syll_15` | Sprint stride -> gait acceleration -> sustained sprint |
| `K6_Bound_GroundSniff` | Bound -> Sniff -> Bound | Exploration | `syll_42 -> syll_71 -> syll_42` | Locomotion bound -> ground sniff -> continued bound |

---

## 2. Inter-Chain Organization Summary

![Inter-Chain Organization](inter_chain_organization_summary.png)

### 2.1 Perseveration Index & Chain Repeat Rates
| genotype   |   count |     mean |      std |
|:-----------|--------:|---------:|---------:|
| Monomer    |      43 | 0.373754 | 0.144066 |
| PFF        |      56 | 0.340404 | 0.15392  |
| Pilot      |       4 | 0.5      | 0.412393 |
| Unknown    |       6 | 0.2223   | 0.151948 |

### 2.2 Continuation Hazard Rates Across Successive Cycles
| genotype   | chain_id             |   total_bouts |   mean_cycles_per_bout |   median_bout_duration_s |   p_continue_1_to_2 |   p_continue_2_to_3 |   p_continue_3_to_4 |
|:-----------|:---------------------|--------------:|-----------------------:|-------------------------:|--------------------:|--------------------:|--------------------:|
| Monomer    | K1_Rest_Micro_Rest   |           296 |                1.01351 |                  1.14167 |          0.0135135  |                 0   |                   0 |
| Monomer    | K2_Rest_Shift_Rest   |           296 |                1.01351 |                  1.16667 |          0.0135135  |                 0   |                   0 |
| Monomer    | K3_Dash_Reorient     |           240 |                1.025   |                  5.15833 |          0.025      |                 0   |                   0 |
| Monomer    | K4_Turn_Loop         |           158 |                1.01266 |                  4.775   |          0.0126582  |                 0   |                   0 |
| Monomer    | K5_Sprint_Burst      |           151 |                1.00662 |                  2.4     |          0.00662252 |                 0   |                   0 |
| Monomer    | K6_Bound_GroundSniff |           195 |                1.00513 |                  3.4     |          0.00512821 |                 0   |                   0 |
| PFF        | K1_Rest_Micro_Rest   |           340 |                1.02353 |                  1.08333 |          0.0235294  |                 0   |                   0 |
| PFF        | K2_Rest_Shift_Rest   |           317 |                1.00946 |                  1.18333 |          0.00630915 |                 0.5 |                   0 |
| PFF        | K3_Dash_Reorient     |           223 |                1.00897 |                  5.06667 |          0.00896861 |                 0   |                   0 |
| PFF        | K4_Turn_Loop         |           221 |                1.0181  |                  4.21667 |          0.0180995  |                 0   |                   0 |
| PFF        | K5_Sprint_Burst      |           226 |                1.00442 |                  2.7     |          0.00442478 |                 0   |                   0 |
| PFF        | K6_Bound_GroundSniff |           198 |                1       |                  3.55    |          0          |                 0   |                   0 |
| Pilot      | K1_Rest_Micro_Rest   |             6 |                1       |                  0.975   |          0          |                 0   |                   0 |
| Pilot      | K2_Rest_Shift_Rest   |             4 |                1       |                  1.30833 |          0          |                 0   |                   0 |
| Pilot      | K3_Dash_Reorient     |            16 |                1.0625  |                  6.14167 |          0.0625     |                 0   |                   0 |
| Pilot      | K4_Turn_Loop         |             1 |                1       |                  3.48333 |          0          |                 0   |                   0 |
| Pilot      | K5_Sprint_Burst      |             5 |                1       |                  2.56667 |          0          |                 0   |                   0 |
| Pilot      | K6_Bound_GroundSniff |             4 |                1       |                  3.68333 |          0          |                 0   |                   0 |
| Unknown    | K1_Rest_Micro_Rest   |            40 |                1.025   |                  1.075   |          0.025      |                 0   |                   0 |
| Unknown    | K2_Rest_Shift_Rest   |            47 |                1.02128 |                  1.21667 |          0.0212766  |                 0   |                   0 |
| Unknown    | K3_Dash_Reorient     |            19 |                1       |                  4.01667 |          0          |                 0   |                   0 |
| Unknown    | K4_Turn_Loop         |            16 |                1       |                  3.875   |          0          |                 0   |                   0 |
| Unknown    | K5_Sprint_Burst      |            22 |                1       |                  2.23333 |          0          |                 0   |                   0 |
| Unknown    | K6_Bound_GroundSniff |            16 |                1       |                  3.39167 |          0          |                 0   |                   0 |

### 2.3 Syntax Transition Entropy Rate
| genotype   |   count |     mean |      std |
|:-----------|--------:|---------:|---------:|
| Monomer    |      43 | 1.34754  | 0.321713 |
| PFF        |      56 | 1.33326  | 0.279812 |
| Pilot      |       4 | 0.619638 | 0.427036 |
| Unknown    |       5 | 1.53509  | 0.228993 |

### 2.4 Initiations: Lead vs. Repetition Kinematics
| chain_id           |   n_multi_bouts |   n_animals_tested |   lead_mean_speed |   repeat_mean_speed |   speed_ratio_lead_to_repeat |   p_speed_diff_paired_mouse |   lead_mean_duration_s |   repeat_mean_duration_s |   duration_ratio_lead_to_repeat |
|:-------------------|----------------:|-------------------:|------------------:|--------------------:|-----------------------------:|----------------------------:|-----------------------:|-------------------------:|--------------------------------:|
| K1_Rest_Micro_Rest |              13 |                 11 |           42.0356 |             42.6984 |                     0.984476 |                    0.873361 |                1.49231 |                  1.36923 |                        1.08989  |
| K2_Rest_Shift_Rest |               7 |                  6 |           54.1525 |             56.8886 |                     0.951904 |                    0.660796 |                1.0619  |                  1.14375 |                        0.928441 |
| K3_Dash_Reorient   |               9 |                  5 |           33.9348 |             35.1268 |                     0.966067 |                    0.050276 |                8.77778 |                  5.71296 |                        1.53647  |
| K4_Turn_Loop       |               6 |                  4 |           33.4424 |             34.741  |                     0.962621 |                    0.188253 |               12.6694  |                 11.1472  |                        1.13656  |

---

## 3. Intra-Chain Execution Fidelity & Dynamics

![Intra-Chain Execution](intra_chain_execution_summary.png)

### 3.1 Execution Consistency (DTW Distance & Profile Correlation)
| chain_id             | genotype   |   n_complete_occurrences |   n_animals_evaluated |   mean_pairwise_dtw_dist |   mean_velocity_profile_correlation |   duration_cv |   mean_duration_s |   std_duration_s |
|:---------------------|:-----------|-------------------------:|----------------------:|-------------------------:|------------------------------------:|--------------:|------------------:|-----------------:|
| K1_Rest_Micro_Rest   | Monomer    |                      300 |                     9 |                  9.34595 |                          0.0516213  |      0.503034 |           1.30167 |         0.654782 |
| K1_Rest_Micro_Rest   | PFF        |                      348 |                    12 |                  9.47412 |                          0.0415332  |      0.509587 |           1.22763 |         0.625587 |
| K1_Rest_Micro_Rest   | Pilot      |                        6 |                     1 |                  7.00427 |                         -0.0221406  |      0.460159 |           1.11389 |         0.512566 |
| K1_Rest_Micro_Rest   | Unknown    |                       41 |                     1 |                  8.33634 |                          0.0791959  |      0.456963 |           1.15772 |         0.529036 |
| K2_Rest_Shift_Rest   | Monomer    |                      300 |                     9 |                 10.1558  |                          0.193738   |      0.358364 |           1.23661 |         0.443157 |
| K2_Rest_Shift_Rest   | PFF        |                      320 |                    12 |                 10.0808  |                          0.170167   |      0.376975 |           1.26365 |         0.476363 |
| K2_Rest_Shift_Rest   | Pilot      |                        4 |                     1 |                  7.47462 |                          0.0847298  |      0.138256 |           1.35417 |         0.187222 |
| K2_Rest_Shift_Rest   | Unknown    |                       48 |                     1 |                  9.2922  |                          0.122587   |      0.334276 |           1.20799 |         0.403801 |
| K3_Dash_Reorient     | Monomer    |                      246 |                     9 |                  5.57729 |                          0.0162499  |      0.69299  |           6.07263 |         4.20827  |
| K3_Dash_Reorient     | PFF        |                      225 |                    12 |                  5.5586  |                         -0.0137681  |      0.591119 |           5.79326 |         3.4245   |
| K3_Dash_Reorient     | Pilot      |                       17 |                     2 |                  5.6913  |                          0.0150831  |      1.07186  |           9.78824 |        10.4916   |
| K3_Dash_Reorient     | Unknown    |                       19 |                     1 |                  5.93924 |                         -0.00978214 |      0.959298 |           5.54825 |         5.32242  |
| K4_Turn_Loop         | Monomer    |                      160 |                     7 |                  5.62646 |                         -0.00395514 |      0.938366 |           7.00615 |         6.57433  |
| K4_Turn_Loop         | PFF        |                      225 |                    12 |                  5.63173 |                          0.00719488 |      0.906699 |           5.97007 |         5.41306  |
| K4_Turn_Loop         | Unknown    |                       16 |                     1 |                  5.79643 |                         -0.00700543 |      0.478738 |           4.28333 |         2.05059  |
| K5_Sprint_Burst      | Monomer    |                      152 |                     9 |                  6.57784 |                          0.0136756  |      0.6388   |           2.91513 |         1.86219  |
| K5_Sprint_Burst      | PFF        |                      227 |                    12 |                  6.696   |                          0.0249001  |      0.50515  |           2.93913 |         1.4847   |
| K5_Sprint_Burst      | Pilot      |                        5 |                     2 |                  9.17229 |                          0.101187   |      0.379915 |           2.89333 |         1.09922  |
| K5_Sprint_Burst      | Unknown    |                       22 |                     1 |                  7.53185 |                          0.0192317  |      0.464864 |           2.62273 |         1.21921  |
| K6_Bound_GroundSniff | Monomer    |                      196 |                     9 |                 16.9722  |                          0.146342   |      0.340413 |           3.5148  |         1.19648  |
| K6_Bound_GroundSniff | PFF        |                      198 |                    12 |                 17.1584  |                          0.106755   |      0.412891 |           3.8537  |         1.59116  |
| K6_Bound_GroundSniff | Pilot      |                        4 |                     1 |                 16.3219  |                         -0.0442649  |      0.25032  |           3.69167 |         0.924099 |
| K6_Bound_GroundSniff | Unknown    |                       16 |                     1 |                 19.9191  |                          0.0499041  |      0.367502 |           3.78229 |         1.39     |

### 3.2 Completion Rates
| chain_id             | genotype   |   total_attempts |   completed_count |   aborted_count |   completion_rate_pct |
|:---------------------|:-----------|-----------------:|------------------:|----------------:|----------------------:|
| K1_Rest_Micro_Rest   | Monomer    |             4704 |               300 |            4404 |               6.37755 |
| K1_Rest_Micro_Rest   | PFF        |             5730 |               348 |            5382 |               6.0733  |
| K1_Rest_Micro_Rest   | Pilot      |              135 |                 6 |             129 |               4.44444 |
| K1_Rest_Micro_Rest   | Unknown    |              720 |                41 |             679 |               5.69444 |
| K2_Rest_Shift_Rest   | Monomer    |             4704 |               300 |            4404 |               6.37755 |
| K2_Rest_Shift_Rest   | PFF        |             5730 |               320 |            5410 |               5.58464 |
| K2_Rest_Shift_Rest   | Pilot      |              135 |                 4 |             131 |               2.96296 |
| K2_Rest_Shift_Rest   | Unknown    |              720 |                48 |             672 |               6.66667 |
| K3_Dash_Reorient     | Monomer    |             5677 |               246 |            5431 |               4.33327 |
| K3_Dash_Reorient     | PFF        |             7857 |               225 |            7632 |               2.86369 |
| K3_Dash_Reorient     | Pilot      |              264 |                17 |             247 |               6.43939 |
| K3_Dash_Reorient     | Unknown    |              887 |                19 |             868 |               2.14205 |
| K4_Turn_Loop         | Monomer    |             2721 |               160 |            2561 |               5.88019 |
| K4_Turn_Loop         | PFF        |             3757 |               225 |            3532 |               5.98882 |
| K4_Turn_Loop         | Pilot      |               74 |                 1 |              73 |               1.35135 |
| K4_Turn_Loop         | Unknown    |              430 |                16 |             414 |               3.72093 |
| K5_Sprint_Burst      | Monomer    |             5677 |               152 |            5525 |               2.67747 |
| K5_Sprint_Burst      | PFF        |             7857 |               227 |            7630 |               2.88914 |
| K5_Sprint_Burst      | Pilot      |              264 |                 5 |             259 |               1.89394 |
| K5_Sprint_Burst      | Unknown    |              887 |                22 |             865 |               2.48027 |
| K6_Bound_GroundSniff | Monomer    |             7372 |               196 |            7176 |               2.65871 |
| K6_Bound_GroundSniff | PFF        |             9184 |               198 |            8986 |               2.15592 |
| K6_Bound_GroundSniff | Pilot      |              146 |                 4 |             142 |               2.73973 |
| K6_Bound_GroundSniff | Unknown    |              910 |                16 |             894 |               1.75824 |

### 3.3 Structural Variations (Insertions, Deletions, Stutters)
| chain_id             | genotype   |   total_attempts |   exact_execution_pct |   mean_edit_distance |   mean_insertions |   mean_deletions |   mean_repetitions_stutter |
|:---------------------|:-----------|-----------------:|----------------------:|---------------------:|------------------:|-----------------:|---------------------------:|
| K1_Rest_Micro_Rest   | Monomer    |             4704 |              5.7398   |              1.77976 |         0.0195578 |          1.7602  |                          0 |
| K1_Rest_Micro_Rest   | PFF        |             5730 |              5.77661  |              1.76003 |         0.0190227 |          1.74101 |                          0 |
| K1_Rest_Micro_Rest   | Pilot      |              135 |              4.44444  |              1.78519 |         0.0814815 |          1.7037  |                          0 |
| K1_Rest_Micro_Rest   | Unknown    |              720 |              5.55556  |              1.77917 |         0.0152778 |          1.76389 |                          0 |
| K2_Rest_Shift_Rest   | Monomer    |             4704 |              5.95238  |              1.82228 |         0.0110544 |          1.81122 |                          0 |
| K2_Rest_Shift_Rest   | PFF        |             5730 |              5.02618  |              1.85497 |         0.0120419 |          1.84293 |                          0 |
| K2_Rest_Shift_Rest   | Pilot      |              135 |              2.96296  |              1.88889 |         0         |          1.88889 |                          0 |
| K2_Rest_Shift_Rest   | Unknown    |              720 |              6.25     |              1.81528 |         0.0138889 |          1.80139 |                          0 |
| K3_Dash_Reorient     | Monomer    |             5677 |              3.61106  |              1.88022 |         0.0269509 |          1.85327 |                          0 |
| K3_Dash_Reorient     | PFF        |             7857 |              2.44368  |              1.90187 |         0.0264732 |          1.8754  |                          0 |
| K3_Dash_Reorient     | Pilot      |              264 |              6.06061  |              1.80682 |         0.0151515 |          1.79167 |                          0 |
| K3_Dash_Reorient     | Unknown    |              887 |              1.57835  |              1.92108 |         0.0417136 |          1.87937 |                          0 |
| K4_Turn_Loop         | Monomer    |             2721 |              5.21867  |              1.77178 |         0.0338111 |          1.73796 |                          0 |
| K4_Turn_Loop         | PFF        |             3757 |              5.08384  |              1.75779 |         0.0367314 |          1.72105 |                          0 |
| K4_Turn_Loop         | Pilot      |               74 |              1.35135  |              1.97297 |         0         |          1.97297 |                          0 |
| K4_Turn_Loop         | Unknown    |              430 |              3.02326  |              1.84884 |         0.0209302 |          1.82791 |                          0 |
| K5_Sprint_Burst      | Monomer    |             5677 |              2.30756  |              1.89079 |         0.0170865 |          1.8737  |                          0 |
| K5_Sprint_Burst      | PFF        |             7857 |              2.44368  |              1.89182 |         0.0206186 |          1.8712  |                          0 |
| K5_Sprint_Burst      | Pilot      |              264 |              1.89394  |              1.93939 |         0.0151515 |          1.92424 |                          0 |
| K5_Sprint_Burst      | Unknown    |              887 |              2.02931  |              1.89064 |         0.0180383 |          1.8726  |                          0 |
| K6_Bound_GroundSniff | Monomer    |             7372 |              1.79056  |              1.87507 |         0.0435431 |          1.83152 |                          0 |
| K6_Bound_GroundSniff | PFF        |             9184 |              1.42639  |              1.89253 |         0.0399608 |          1.85257 |                          0 |
| K6_Bound_GroundSniff | Pilot      |              146 |              2.73973  |              1.89041 |         0.0273973 |          1.86301 |                          0 |
| K6_Bound_GroundSniff | Unknown    |              910 |              0.989011 |              1.93407 |         0.0362637 |          1.8978  |                          0 |

### 3.4 Temporal Stage Pacing Profiles
| chain_id             | genotype   |   stage_index |   stage_token |   stage_mean_duration_s |   stage_std_duration_s |   stage_relative_fraction |
|:---------------------|:-----------|--------------:|--------------:|------------------------:|-----------------------:|--------------------------:|
| K1_Rest_Micro_Rest   | Monomer    |             0 |            29 |                0.273944 |              0.226503  |                  0.221518 |
| K1_Rest_Micro_Rest   | Monomer    |             1 |             1 |                0.6275   |              0.435093  |                  0.507412 |
| K1_Rest_Micro_Rest   | Monomer    |             2 |            29 |                0.335222 |              0.236017  |                  0.271069 |
| K1_Rest_Micro_Rest   | PFF        |             0 |            29 |                0.259243 |              0.201059  |                  0.218999 |
| K1_Rest_Micro_Rest   | PFF        |             1 |             1 |                0.573228 |              0.365167  |                  0.484242 |
| K1_Rest_Micro_Rest   | PFF        |             2 |            29 |                0.351293 |              0.288569  |                  0.296759 |
| K1_Rest_Micro_Rest   | Pilot      |             0 |            29 |                0.161111 |              0.0895806 |                  0.144638 |
| K1_Rest_Micro_Rest   | Pilot      |             1 |             1 |                0.797222 |              0.492764  |                  0.715711 |
| K1_Rest_Micro_Rest   | Pilot      |             2 |            29 |                0.155556 |              0.0797372 |                  0.139651 |
| K1_Rest_Micro_Rest   | Unknown    |             0 |            29 |                0.235366 |              0.158465  |                  0.20583  |
| K1_Rest_Micro_Rest   | Unknown    |             1 |             1 |                0.603659 |              0.376414  |                  0.527906 |
| K1_Rest_Micro_Rest   | Unknown    |             2 |            29 |                0.304472 |              0.183057  |                  0.266264 |
| K2_Rest_Shift_Rest   | Monomer    |             0 |            29 |                0.234778 |              0.151455  |                  0.195485 |
| K2_Rest_Shift_Rest   | Monomer    |             1 |            30 |                0.661889 |              0.32167   |                  0.551115 |
| K2_Rest_Shift_Rest   | Monomer    |             2 |            29 |                0.304333 |              0.199985  |                  0.2534   |
| K2_Rest_Shift_Rest   | PFF        |             0 |            29 |                0.255573 |              0.170353  |                  0.212277 |
| K2_Rest_Shift_Rest   | PFF        |             1 |            30 |                0.628542 |              0.28134   |                  0.522063 |
| K2_Rest_Shift_Rest   | PFF        |             2 |            29 |                0.319844 |              0.225118  |                  0.26566  |
| K2_Rest_Shift_Rest   | Pilot      |             0 |            29 |                0.220833 |              0.129301  |                  0.163077 |
| K2_Rest_Shift_Rest   | Pilot      |             1 |            30 |                0.945833 |              0.250382  |                  0.698462 |
| K2_Rest_Shift_Rest   | Pilot      |             2 |            29 |                0.1875   |              0.0660124 |                  0.138462 |
| K2_Rest_Shift_Rest   | Unknown    |             0 |            29 |                0.246528 |              0.158186  |                  0.209316 |
| K2_Rest_Shift_Rest   | Unknown    |             1 |            30 |                0.660417 |              0.262569  |                  0.560731 |
| K2_Rest_Shift_Rest   | Unknown    |             2 |            29 |                0.270833 |              0.150366  |                  0.229953 |
| K3_Dash_Reorient     | Monomer    |             0 |            15 |                1.46463  |              1.05178   |                  0.251759 |
| K3_Dash_Reorient     | Monomer    |             1 |            25 |                2.98808  |              3.40189   |                  0.513626 |
| K3_Dash_Reorient     | Monomer    |             2 |            15 |                1.36491  |              1.09173   |                  0.234616 |
| K3_Dash_Reorient     | PFF        |             0 |            15 |                1.64933  |              1.58205   |                  0.2927   |
| K3_Dash_Reorient     | PFF        |             1 |            25 |                2.41104  |              2.40495   |                  0.427877 |
| K3_Dash_Reorient     | PFF        |             2 |            15 |                1.57452  |              1.50564   |                  0.279423 |
| K3_Dash_Reorient     | Pilot      |             0 |            15 |                1.90882  |              1.09352   |                  0.195737 |
| K3_Dash_Reorient     | Pilot      |             1 |            25 |                6.4598   |             10.8177    |                  0.662411 |
| K3_Dash_Reorient     | Pilot      |             2 |            15 |                1.38333  |              0.883047  |                  0.141852 |
| K3_Dash_Reorient     | Unknown    |             0 |            15 |                1.09737  |              0.887243  |                  0.204278 |
| K3_Dash_Reorient     | Unknown    |             1 |            25 |                3.12193  |              5.63924   |                  0.581156 |
| K3_Dash_Reorient     | Unknown    |             2 |            15 |                1.15263  |              0.902049  |                  0.214566 |
| K4_Turn_Loop         | Monomer    |             0 |            56 |                2.94302  |              4.04329   |                  0.429808 |
| K4_Turn_Loop         | Monomer    |             1 |            58 |                0.785417 |              0.514555  |                  0.114705 |
| K4_Turn_Loop         | Monomer    |             2 |            56 |                3.11885  |              4.12982   |                  0.455487 |
| K4_Turn_Loop         | PFF        |             0 |            56 |                2.35252  |              2.99011   |                  0.419515 |
| K4_Turn_Loop         | PFF        |             1 |            58 |                0.803407 |              0.661882  |                  0.143269 |
| K4_Turn_Loop         | PFF        |             2 |            56 |                2.45178  |              2.99263   |                  0.437216 |
| K4_Turn_Loop         | Unknown    |             0 |            56 |                1.56667  |              0.947438  |                  0.412507 |
| K4_Turn_Loop         | Unknown    |             1 |            58 |                0.595833 |              0.605774  |                  0.156884 |
| K4_Turn_Loop         | Unknown    |             2 |            56 |                1.63542  |              0.564207  |                  0.430609 |
| K5_Sprint_Burst      | Monomer    |             0 |            15 |                1.01711  |              0.973751  |                  0.376232 |
| K5_Sprint_Burst      | Monomer    |             1 |            53 |                0.763377 |              0.662727  |                  0.282377 |
| K5_Sprint_Burst      | Monomer    |             2 |            15 |                0.922917 |              1.02834   |                  0.341391 |
| K5_Sprint_Burst      | PFF        |             0 |            15 |                0.98069  |              0.866434  |                  0.358308 |
| K5_Sprint_Burst      | PFF        |             1 |            53 |                0.748678 |              0.673352  |                  0.273539 |
| K5_Sprint_Burst      | PFF        |             2 |            15 |                1.00764  |              0.850264  |                  0.368153 |
| K5_Sprint_Burst      | Pilot      |             0 |            15 |                1.09333  |              1.12015   |                  0.37788  |
| K5_Sprint_Burst      | Pilot      |             1 |            53 |                0.876667 |              0.639149  |                  0.302995 |
| K5_Sprint_Burst      | Pilot      |             2 |            15 |                0.923333 |              0.639149  |                  0.319124 |
| K5_Sprint_Burst      | Unknown    |             0 |            15 |                0.847727 |              0.768842  |                  0.344626 |
| K5_Sprint_Burst      | Unknown    |             1 |            53 |                0.768182 |              0.610243  |                  0.312288 |
| K5_Sprint_Burst      | Unknown    |             2 |            15 |                0.843939 |              0.513247  |                  0.343086 |
| K6_Bound_GroundSniff | Monomer    |             0 |            42 |                1.15094  |              0.670191  |                  0.352805 |
| K6_Bound_GroundSniff | Monomer    |             1 |            71 |                0.906463 |              0.519411  |                  0.277865 |
| K6_Bound_GroundSniff | Monomer    |             2 |            42 |                1.20485  |              0.721643  |                  0.369331 |
| K6_Bound_GroundSniff | PFF        |             0 |            42 |                1.29941  |              0.962935  |                  0.361649 |
| K6_Bound_GroundSniff | PFF        |             1 |            71 |                0.935017 |              0.54041   |                  0.260232 |
| K6_Bound_GroundSniff | PFF        |             2 |            42 |                1.35859  |              0.853692  |                  0.378119 |
| K6_Bound_GroundSniff | Pilot      |             0 |            42 |                1.60417  |              1.06858   |                  0.434537 |
| K6_Bound_GroundSniff | Pilot      |             1 |            71 |                1.19167  |              0.464504  |                  0.322799 |
| K6_Bound_GroundSniff | Pilot      |             2 |            42 |                0.895833 |              0.530903  |                  0.242664 |
| K6_Bound_GroundSniff | Unknown    |             0 |            42 |                1.37083  |              1.01969   |                  0.40455  |
| K6_Bound_GroundSniff | Unknown    |             1 |            71 |                0.8625   |              0.423506  |                  0.254534 |
| K6_Bound_GroundSniff | Unknown    |             2 |            42 |                1.15521  |              0.519906  |                  0.340916 |

---

## 4. Context-Dependent Movement

![Context Dependent Movement](context_dependent_syllable_kinematics.png)

|   syllable_token | genotype   |   n_in_chain_samples |   n_outside_chain_samples |   in_chain_mean_duration_s |   outside_chain_mean_duration_s |   duration_modulation_ratio |   in_chain_mean_speed |   outside_chain_mean_speed |   speed_modulation_ratio |   in_chain_speed_std |
|-----------------:|:-----------|---------------------:|--------------------------:|---------------------------:|--------------------------------:|----------------------------:|----------------------:|---------------------------:|-------------------------:|---------------------:|
|                1 | Monomer    |                  300 |                      4520 |                   0.6275   |                        1.02532  |                    0.612006 |               42.0586 |                   nan      |               nan        |              9.20851 |
|                1 | PFF        |                  348 |                      6671 |                   0.573228 |                        1.0111   |                    0.566933 |               41.4686 |                   nan      |               nan        |              9.33119 |
|                1 | Pilot      |                    6 |                       263 |                   0.797222 |                        0.974144 |                    0.818382 |               36.107  |                   nan      |               nan        |              9.1497  |
|                1 | Unknown    |                   41 |                       671 |                   0.603659 |                        0.933085 |                    0.646949 |               40.0071 |                    45.7428 |                 0.87461  |              8.68514 |
|               15 | Monomer    |                  796 |                      4935 |                   1.24491  |                        1.00937  |                    1.23335  |               35.5351 |                    35.6676 |                 0.996285 |              4.0355  |
|               15 | PFF        |                  904 |                      7011 |                   1.30168  |                        1.12307  |                    1.15904  |               35.2483 |                    35.8003 |                 0.984582 |              4.25298 |
|               15 | Pilot      |                   44 |                       229 |                   1.50114  |                        1.1492   |                    1.30625  |               36.3369 |                    35.6207 |                 1.02011  |              4.38636 |
|               15 | Unknown    |                   82 |                       806 |                   0.975203 |                        0.876778 |                    1.11226  |               35.8004 |                    37.4452 |                 0.956074 |              5.62042 |
|               25 | Monomer    |                  246 |                      1802 |                   2.98808  |                        2.49709  |                    1.19662  |               34.1863 |                   nan      |               nan        |              2.39222 |
|               25 | PFF        |                  225 |                      2929 |                   2.41104  |                        2.25217  |                    1.07054  |               34.5131 |                   nan      |               nan        |              3.46372 |
|               25 | Pilot      |                   17 |                       189 |                   6.4598   |                        6.14674  |                    1.05093  |               32.8641 |                    34.3173 |                 0.957655 |              1.86467 |
|               25 | Unknown    |                   19 |                       386 |                   3.12193  |                        2.85941  |                    1.09181  |               35.7069 |                   nan      |               nan        |              3.54447 |
|               29 | Monomer    |                 1200 |                      3521 |                   0.287069 |                        0.33873  |                    0.847489 |               61.3946 |                   nan      |               nan        |             29.128   |
|               29 | PFF        |                 1336 |                      4427 |                   0.296856 |                        0.327795 |                    0.905615 |               60.4086 |                   nan      |               nan        |             27.3132  |
|               29 | Pilot      |                   20 |                       115 |                   0.176667 |                        0.284638 |                    0.620672 |               43.7727 |                    46.2383 |                 0.946676 |             13.4192  |
|               29 | Unknown    |                  178 |                       547 |                   0.263858 |                        0.320689 |                    0.822785 |               58.1834 |                    56.9249 |                 1.02211  |             22.6098  |
|               30 | Monomer    |                  300 |                      2152 |                   0.661889 |                        0.81813  |                    0.809026 |               48.3306 |                    47.1341 |                 1.02538  |             11.2769  |
|               30 | PFF        |                  320 |                      2437 |                   0.628542 |                        0.837279 |                    0.750695 |               46.7677 |                    45.6691 |                 1.02406  |             10.0781  |
|               30 | Pilot      |                    4 |                        30 |                   0.945833 |                        0.770556 |                    1.22747  |               45.3743 |                    46.5732 |                 0.974258 |              4.48304 |
|               30 | Unknown    |                   48 |                       357 |                   0.660417 |                        0.830486 |                    0.795218 |               49.3287 |                    48.2471 |                 1.02242  |             11.8157  |
|               42 | Monomer    |                  392 |                      6992 |                   1.17789  |                        1.04242  |                    1.12996  |              125.525  |                   124.483  |                 1.00838  |             33.3747  |
|               42 | PFF        |                  396 |                      8794 |                   1.329    |                        1.12173  |                    1.18478  |              119.165  |                   nan      |               nan        |             30.4258  |
|               42 | Pilot      |                    8 |                       138 |                   1.25     |                        1.08394  |                    1.1532   |              114.375  |                   113.325  |                 1.00926  |             27.3787  |
|               42 | Unknown    |                   32 |                       878 |                   1.26302  |                        1.01625  |                    1.24283  |              129.87   |                   134.71   |                 0.964074 |             36.5024  |
|               53 | Monomer    |                  152 |                      3031 |                   0.763377 |                        1.10699  |                    0.689594 |               46.6766 |                    59.7194 |                 0.781599 |             10.5432  |
|               53 | PFF        |                  227 |                      4106 |                   0.748678 |                        1.11817  |                    0.669555 |               45.9751 |                    58.8726 |                 0.780925 |             11.7205  |
|               53 | Pilot      |                    5 |                        94 |                   0.876667 |                        1.36135  |                    0.64397  |               44.9208 |                    62.1071 |                 0.72328  |             20.447   |
|               53 | Unknown    |                   22 |                       455 |                   0.768182 |                        1.0456   |                    0.734677 |               47.6567 |                    62.7864 |                 0.759029 |             11.2223  |
|               56 | Monomer    |                  320 |                      2418 |                   3.03094  |                        2.19376  |                    1.38162  |               34.5234 |                   nan      |               nan        |              2.75926 |
|               56 | PFF        |                  450 |                      3337 |                   2.40215  |                        2.08116  |                    1.15423  |               34.234  |                    34.4769 |                 0.992955 |              3.83888 |
|               56 | Unknown    |                   32 |                       398 |                   1.60104  |                        1.86114  |                    0.860248 |               34.2566 |                    36.0305 |                 0.950766 |              3.12689 |
|               58 | Monomer    |                  160 |                      3545 |                   0.785417 |                        0.94149  |                    0.834227 |               36.529  |                    39.5931 |                 0.92261  |              5.3623  |
|               58 | PFF        |                  225 |                      5247 |                   0.803407 |                        0.963595 |                    0.83376  |               36.3222 |                    39.1661 |                 0.927389 |              5.26807 |
|               58 | Unknown    |                   16 |                       436 |                   0.595833 |                        0.844304 |                    0.705709 |               36.683  |                    40.2798 |                 0.910705 |              4.32283 |
|               71 | Monomer    |                  196 |                      6092 |                   0.906463 |                        0.934611 |                    0.969882 |               52.4142 |                    50.436  |                 1.03922  |              9.8472  |
|               71 | PFF        |                  198 |                      7463 |                   0.935017 |                        0.941241 |                    0.993387 |               51.5389 |                    47.912  |                 1.0757   |              8.60686 |
|               71 | Pilot      |                    4 |                       180 |                   1.19167  |                        0.938889 |                    1.26923  |               46.776  |                    45.0694 |                 1.03787  |              7.52244 |
|               71 | Unknown    |                   16 |                       668 |                   0.8625   |                        0.858957 |                    1.00412  |               53.2812 |                    51.2597 |                 1.03944  |              5.24154 |

