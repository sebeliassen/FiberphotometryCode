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
| genotype   |   count |      mean |       std |
|:-----------|--------:|----------:|----------:|
| Monomer    |      43 | 0.147099  | 0.0865903 |
| PFF        |      56 | 0.141484  | 0.0887975 |
| Pilot      |       4 | 0.0663118 | 0.0568792 |
| Unknown    |       6 | 0.143862  | 0.116415  |

### 2.2 Continuation Hazard Rates Across Successive Cycles
| genotype   | chain_id             |   total_bouts |   mean_cycles_per_bout |   median_bout_duration_s |   p_continue_1_to_2 |   p_continue_2_to_3 |   p_continue_3_to_4 |
|:-----------|:---------------------|--------------:|-----------------------:|-------------------------:|--------------------:|--------------------:|--------------------:|
| Monomer    | K1_Rest_Micro_Rest   |          4704 |                1       |                 0.316667 |           0         |           0         |            0        |
| Monomer    | K2_Rest_Shift_Rest   |          4704 |                1       |                 0.3      |           0         |           0         |            0        |
| Monomer    | K3_Dash_Reorient     |          5677 |                1       |                 0.833333 |           0         |           0         |            0        |
| Monomer    | K4_Turn_Loop         |          2417 |                1.12578 |                 1.7      |           0.102607  |           0.169355  |            0.190476 |
| Monomer    | K5_Sprint_Burst      |          5677 |                1       |                 0.85     |           0         |           0         |            0        |
| Monomer    | K6_Bound_GroundSniff |          6500 |                1.13415 |                 1.16667  |           0.117538  |           0.111257  |            0.223529 |
| PFF        | K1_Rest_Micro_Rest   |          5730 |                1       |                 0.316667 |           0         |           0         |            0        |
| PFF        | K2_Rest_Shift_Rest   |          5730 |                1       |                 0.283333 |           0         |           0         |            0        |
| PFF        | K3_Dash_Reorient     |          7857 |                1       |                 0.866667 |           0         |           0         |            0        |
| PFF        | K4_Turn_Loop         |          3288 |                1.14264 |                 1.68333  |           0.113747  |           0.160428  |            0.283333 |
| PFF        | K5_Sprint_Burst      |          7857 |                1       |                 0.883333 |           0         |           0         |            0        |
| PFF        | K6_Bound_GroundSniff |          8179 |                1.12288 |                 1.2      |           0.109549  |           0.104911  |            0.138298 |
| Pilot      | K1_Rest_Micro_Rest   |           135 |                1       |                 0.3      |           0         |           0         |            0        |
| Pilot      | K2_Rest_Shift_Rest   |           135 |                1       |                 0.233333 |           0         |           0         |            0        |
| Pilot      | K3_Dash_Reorient     |           264 |                1       |                 0.916667 |           0         |           0         |            0        |
| Pilot      | K4_Turn_Loop         |            67 |                1.10448 |                 1.35     |           0.0597015 |           0.75      |            0        |
| Pilot      | K5_Sprint_Burst      |           264 |                1       |                 0.875    |           0         |           0         |            0        |
| Pilot      | K6_Bound_GroundSniff |           133 |                1.09774 |                 1.15     |           0.0902256 |           0.0833333 |            0        |
| Unknown    | K1_Rest_Micro_Rest   |           720 |                1       |                 0.316667 |           0         |           0         |            0        |
| Unknown    | K2_Rest_Shift_Rest   |           720 |                1       |                 0.3      |           0         |           0         |            0        |
| Unknown    | K3_Dash_Reorient     |           887 |                1       |                 0.75     |           0         |           0         |            0        |
| Unknown    | K4_Turn_Loop         |           375 |                1.14667 |                 1.35     |           0.122667  |           0.173913  |            0.125    |
| Unknown    | K5_Sprint_Burst      |           887 |                1       |                 0.733333 |           0         |           0         |            0        |
| Unknown    | K6_Bound_GroundSniff |           824 |                1.10437 |                 1.05     |           0.100728  |           0.0361446 |            0        |

### 2.3 Syntax Transition Entropy Rate
| genotype   |   count |     mean |       std |
|:-----------|--------:|---------:|----------:|
| Monomer    |      43 | 1.08233  | 0.120976  |
| PFF        |      56 | 1.07231  | 0.160253  |
| Pilot      |       4 | 0.885154 | 0.209854  |
| Unknown    |       6 | 1.07896  | 0.0989691 |

### 2.4 Initiations: Lead vs. Repetition Kinematics
| chain_id             |   n_multi_bouts |   lead_mean_speed |   repeat_mean_speed |   speed_ratio_lead_to_repeat |   p_speed_diff |   lead_mean_duration_s |   repeat_mean_duration_s |   duration_ratio_lead_to_repeat |   p_duration_diff |
|:---------------------|----------------:|------------------:|--------------------:|-----------------------------:|---------------:|-----------------------:|-------------------------:|--------------------------------:|------------------:|
| K4_Turn_Loop         |             672 |           34.8114 |             34.6281 |                      1.00529 |       0.334963 |                4.14187 |                  3.63802 |                         1.13849 |       0.067352    |
| K6_Bound_GroundSniff |            1755 |          121.127  |            123.192  |                      0.98324 |       0.126533 |                1.53301 |                  1.19976 |                         1.27777 |       2.99104e-17 |

---

## 3. Intra-Chain Execution Fidelity & Dynamics

![Intra-Chain Execution](intra_chain_execution_summary.png)

### 3.1 Execution Consistency (DTW Distance & Profile Correlation)
| chain_id             | genotype   |   n_complete_occurrences |   mean_pairwise_dtw_dist |   mean_velocity_profile_correlation |   duration_cv |   mean_duration_s |   std_duration_s |
|:---------------------|:-----------|-------------------------:|-------------------------:|------------------------------------:|--------------:|------------------:|-----------------:|
| K1_Rest_Micro_Rest   | Monomer    |                      300 |                 10.1331  |                         0.0611626   |      0.503034 |           1.30167 |         0.654782 |
| K1_Rest_Micro_Rest   | PFF        |                      348 |                 10.2083  |                         0.0520613   |      0.509587 |           1.22763 |         0.625587 |
| K1_Rest_Micro_Rest   | Pilot      |                        6 |                  7.00427 |                        -0.0221406   |      0.460159 |           1.11389 |         0.512566 |
| K1_Rest_Micro_Rest   | Unknown    |                       41 |                  8.33634 |                         0.0791959   |      0.456963 |           1.15772 |         0.529036 |
| K2_Rest_Shift_Rest   | Monomer    |                      300 |                 11.114   |                         0.189658    |      0.358364 |           1.23661 |         0.443157 |
| K2_Rest_Shift_Rest   | PFF        |                      320 |                  9.45836 |                         0.164048    |      0.376975 |           1.26365 |         0.476363 |
| K2_Rest_Shift_Rest   | Pilot      |                        4 |                  7.47462 |                         0.0847298   |      0.138256 |           1.35417 |         0.187222 |
| K2_Rest_Shift_Rest   | Unknown    |                       48 |                  9.2922  |                         0.122587    |      0.334276 |           1.20799 |         0.403801 |
| K3_Dash_Reorient     | Monomer    |                      246 |                  5.66134 |                         0.0058084   |      0.69299  |           6.07263 |         4.20827  |
| K3_Dash_Reorient     | PFF        |                      225 |                  5.58122 |                         0.000726113 |      0.591119 |           5.79326 |         3.4245   |
| K3_Dash_Reorient     | Pilot      |                       17 |                  5.59858 |                         0.0259422   |      1.07186  |           9.78824 |        10.4916   |
| K3_Dash_Reorient     | Unknown    |                       19 |                  5.93924 |                        -0.00978214  |      0.959298 |           5.54825 |         5.32242  |
| K4_Turn_Loop         | Monomer    |                      160 |                  5.59045 |                         0.0100156   |      0.938366 |           7.00615 |         6.57433  |
| K4_Turn_Loop         | PFF        |                      225 |                  5.62433 |                        -0.0026277   |      0.906699 |           5.97007 |         5.41306  |
| K4_Turn_Loop         | Unknown    |                       16 |                  5.79643 |                        -0.00700543  |      0.478738 |           4.28333 |         2.05059  |
| K5_Sprint_Burst      | Monomer    |                      152 |                  6.79019 |                         0.0200427   |      0.6388   |           2.91513 |         1.86219  |
| K5_Sprint_Burst      | PFF        |                      227 |                  6.66459 |                         0.0278705   |      0.50515  |           2.93913 |         1.4847   |
| K5_Sprint_Burst      | Pilot      |                        5 |                  8.9085  |                         0.0458582   |      0.379915 |           2.89333 |         1.09922  |
| K5_Sprint_Burst      | Unknown    |                       22 |                  7.53185 |                         0.0192317   |      0.464864 |           2.62273 |         1.21921  |
| K6_Bound_GroundSniff | Monomer    |                      196 |                 17.0912  |                         0.151321    |      0.340413 |           3.5148  |         1.19648  |
| K6_Bound_GroundSniff | PFF        |                      198 |                 18.011   |                         0.129955    |      0.412891 |           3.8537  |         1.59116  |
| K6_Bound_GroundSniff | Pilot      |                        4 |                 16.3219  |                        -0.0442649   |      0.25032  |           3.69167 |         0.924099 |
| K6_Bound_GroundSniff | Unknown    |                       16 |                 19.9191  |                         0.0499041   |      0.367502 |           3.78229 |         1.39     |

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

|   syllable_token | genotype   |   n_in_chain_samples |   in_chain_mean_duration_s |   in_chain_mean_speed |   in_chain_speed_std |
|-----------------:|:-----------|---------------------:|---------------------------:|----------------------:|---------------------:|
|                1 | Monomer    |                  300 |                   0.6275   |               48.6464 |            12.2352   |
|                1 | PFF        |                  348 |                   0.573228 |               49.1217 |            12.8704   |
|                1 | Pilot      |                    6 |                   0.797222 |               37.8268 |             7.43518  |
|                1 | Unknown    |                   41 |                   0.603659 |               46.7435 |             9.16415  |
|               15 | Monomer    |                  398 |                   1.29372  |               37.832  |             9.14905  |
|               15 | PFF        |                  452 |                   1.31353  |               38.7069 |            10.5842   |
|               15 | Pilot      |                   22 |                   1.72348  |               36.3229 |             6.52321  |
|               15 | Unknown    |                   41 |                   0.963415 |               38.4914 |             5.22687  |
|               25 | Monomer    |                  246 |                   2.98808  |               35.1905 |             2.90538  |
|               25 | PFF        |                  225 |                   2.41104  |               34.9032 |             2.2765   |
|               25 | Pilot      |                   17 |                   6.4598   |               34.6186 |             0.861282 |
|               25 | Unknown    |                   19 |                   3.12193  |               35.5768 |             2.21058  |
|               29 | Monomer    |                  600 |                   0.254361 |               54.022  |            13.9632   |
|               29 | PFF        |                  668 |                   0.257485 |               52.6195 |            11.9032   |
|               29 | Pilot      |                   10 |                   0.185    |               41.2531 |             7.26667  |
|               29 | Unknown    |                   89 |                   0.241386 |               51.7967 |            11.0375   |
|               30 | Monomer    |                  300 |                   0.661889 |               58.7452 |            12.3397   |
|               30 | PFF        |                  320 |                   0.628542 |               56.5759 |            10.0012   |
|               30 | Pilot      |                    4 |                   0.945833 |               46.3924 |             2.2509   |
|               30 | Unknown    |                   48 |                   0.660417 |               55.8795 |            10.6774   |
|               42 | Monomer    |                  196 |                   1.15094  |              103.138  |            23.2885   |
|               42 | PFF        |                  198 |                   1.29941  |               98.6975 |            22.2304   |
|               42 | Pilot      |                    4 |                   1.60417  |               92.8494 |             4.18471  |
|               42 | Unknown    |                   16 |                   1.37083  |              111.407  |            26.1488   |
|               53 | Monomer    |                  152 |                   0.763377 |               39.5227 |             6.42116  |
|               53 | PFF        |                  227 |                   0.748678 |               40.2882 |             8.47851  |
|               53 | Pilot      |                    5 |                   0.876667 |               42.1174 |            11.8851   |
|               53 | Unknown    |                   22 |                   0.768182 |               41.1588 |             5.72035  |
|               56 | Monomer    |                  160 |                   2.94302  |               35.3099 |             4.09524  |
|               56 | PFF        |                  225 |                   2.35252  |               35.2319 |             5.8146   |
|               56 | Unknown    |                   16 |                   1.56667  |               35.172  |             3.10205  |
|               58 | Monomer    |                  160 |                   0.785417 |               35.1353 |             2.27746  |
|               58 | PFF        |                  225 |                   0.803407 |               34.9921 |             2.957    |
|               58 | Unknown    |                   16 |                   0.595833 |               35.172  |             3.10205  |
|               71 | Monomer    |                  196 |                   0.906463 |              102.612  |            23.7573   |
|               71 | PFF        |                  198 |                   0.935017 |               97.6455 |            23.5226   |
|               71 | Pilot      |                    4 |                   1.19167  |               92.8494 |             4.18471  |
|               71 | Unknown    |                   16 |                   0.8625   |              111.407  |            26.1488   |

