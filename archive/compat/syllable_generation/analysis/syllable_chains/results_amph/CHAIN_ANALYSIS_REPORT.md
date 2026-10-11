# Project B: Syllable Chain Dynamics Report — Amph

**Cohort:** Amph  
**Sessions Analyzed:** 30  
**Total Chain Executions Detected:** 32,060  

## 1. Canonical Behavioral Chains Analyzed

| Chain ID | Name | Category | Syllables | Description |
|---|---|---|---|---|
| `M1_Rear_Descent` | Rear -> Descent | Vertical Rearing | `syll_14 -> syll_04 -> syll_03` | Rearing up -> sniffing/extending -> descending back down |
| `M5_Slalom_Sprint` | Slalom Sprint | Locomotion | `syll_06 -> syll_17 -> syll_02` | High-speed progression -> darting -> slalom sprint |
| `M8_Turn_Wall_Turn` | Turn -> Wall -> Turn | Wall Exploration | `syll_11 -> syll_01 -> syll_30` | Right turn -> running along wall -> right turn |
| `M10_Groom_Rest_Groom` | Groom -> Rest -> Groom | Quiescence/Groom | `syll_20 -> syll_34 -> syll_49` | Grooming burst -> pause/rest -> face-wash groom |
| `M6_Dash_Redirect` | Dash & Redirect | Locomotion | `syll_21 -> syll_02 -> syll_00` | Rapid forward dart -> steer -> settling orient |
| `M7_Wall_Exploration` | Wall Exploration | Exploration | `syll_16 -> syll_14 -> syll_07` | Wall sniff -> unsupported rear -> explore perimeter |

---

## 2. Inter-Chain Organization Summary

![Inter-Chain Organization](inter_chain_organization_summary.png)

### 2.1 Perseveration Index & Chain Repeat Rates
| genotype   |   count |     mean |       std |
|:-----------|--------:|---------:|----------:|
| KI         |      14 | 0.260237 | 0.0473192 |
| WT         |      16 | 0.326548 | 0.0903211 |

### 2.2 Continuation Hazard Rates Across Successive Cycles
| genotype   | chain_id             |   total_bouts |   mean_cycles_per_bout |   median_bout_duration_s |   p_continue_1_to_2 |   p_continue_2_to_3 |   p_continue_3_to_4 |
|:-----------|:---------------------|--------------:|-----------------------:|-------------------------:|--------------------:|--------------------:|--------------------:|
| KI         | M1_Rear_Descent      |          2894 |                1.07153 |                   0.6125 |           0.0646164 |           0.101604  |           0.0526316 |
| KI         | M5_Slalom_Sprint     |          3446 |                1.16512 |                   0.8    |           0.142484  |           0.138493  |           0.132353  |
| KI         | M8_Turn_Wall_Turn    |          2545 |                1.21257 |                   1.075  |           0.180747  |           0.158696  |           0.0821918 |
| KI         | M10_Groom_Rest_Groom |           905 |                1.0895  |                   0.425  |           0.080663  |           0.0410959 |           0.333333  |
| KI         | M6_Dash_Redirect     |          1993 |                1.07827 |                   0.675  |           0.0752634 |           0.04      |           0         |
| KI         | M7_Wall_Exploration  |          3128 |                1.07801 |                   0.85   |           0.0703325 |           0.109091  |           0         |
| WT         | M1_Rear_Descent      |          2105 |                1.08124 |                   0.6    |           0.072209  |           0.125     |           0         |
| WT         | M5_Slalom_Sprint     |          2923 |                1.13514 |                   0.675  |           0.114266  |           0.161677  |           0.12963   |
| WT         | M8_Turn_Wall_Turn    |          2004 |                1.22305 |                   1.15   |           0.185629  |           0.177419  |           0.136364  |
| WT         | M10_Groom_Rest_Groom |          3145 |                1.12528 |                   0.525  |           0.1062    |           0.158683  |           0.113208  |
| WT         | M6_Dash_Redirect     |          1937 |                1.05472 |                   0.8    |           0.0536913 |           0.0192308 |           0         |
| WT         | M7_Wall_Exploration  |          1667 |                1.03419 |                   0.775  |           0.0323935 |           0.0555556 |           0         |

### 2.3 Syntax Transition Entropy Rate
| genotype   |   count |    mean |       std |
|:-----------|--------:|--------:|----------:|
| KI         |      14 | 2.27867 | 0.0665888 |
| WT         |      16 | 2.181   | 0.229858  |

### 2.4 Initiations: Lead vs. Repetition Kinematics
| chain_id             |   n_multi_bouts |   lead_mean_speed |   repeat_mean_speed |   speed_ratio_lead_to_repeat |   p_speed_diff |   lead_mean_duration_s |   repeat_mean_duration_s |   duration_ratio_lead_to_repeat |   p_duration_diff |
|:---------------------|----------------:|------------------:|--------------------:|-----------------------------:|---------------:|-----------------------:|-------------------------:|--------------------------------:|------------------:|
| M1_Rear_Descent      |             339 |           62.8375 |             64.4237 |                     0.975378 |    0.27864     |               0.581932 |                 0.695304 |                        0.836946 |       0.00589842  |
| M5_Slalom_Sprint     |             825 |           98.285  |             94.1882 |                     1.0435   |    0.0512168   |               0.870121 |                 0.904927 |                        0.961537 |       0.313161    |
| M8_Turn_Wall_Turn    |             832 |          127.671  |            114.817  |                     1.11195  |    2.20161e-08 |               1.15856  |                 1.1961   |                        0.968615 |       0.333307    |
| M10_Groom_Rest_Groom |             407 |           57.3647 |             55.8864 |                     1.02645  |    0.166845    |               1.01959  |                 0.864842 |                        1.17894  |       0.0465697   |
| M6_Dash_Redirect     |             254 |          134.507  |            124.933  |                     1.07663  |    0.0718121   |               0.820571 |                 0.753626 |                        1.08883  |       0.197676    |
| M7_Wall_Exploration  |             274 |           67.2943 |             70.8099 |                     0.950352 |    0.0298925   |               0.729471 |                 0.920266 |                        0.792674 |       3.13844e-05 |

---

## 3. Intra-Chain Execution Fidelity & Dynamics

![Intra-Chain Execution](intra_chain_execution_summary.png)

### 3.1 Execution Consistency (DTW Distance & Profile Correlation)
| chain_id             | genotype   |   n_complete_occurrences |   mean_pairwise_dtw_dist |   mean_velocity_profile_correlation |   duration_cv |   mean_duration_s |   std_duration_s |
|:---------------------|:-----------|-------------------------:|-------------------------:|------------------------------------:|--------------:|------------------:|-----------------:|
| M1_Rear_Descent      | KI         |                      298 |                 15.101   |                          0.119581   |      0.594151 |           1.35596 |         0.805643 |
| M1_Rear_Descent      | WT         |                      173 |                 11.6413  |                          0.102545   |      0.501051 |           1.14971 |         0.576064 |
| M5_Slalom_Sprint     | KI         |                      651 |                 24.5075  |                          0.0301986  |      0.53618  |           1.5578  |         0.835259 |
| M5_Slalom_Sprint     | WT         |                      322 |                 19.3636  |                          0.0572102  |      0.574594 |           1.47686 |         0.848597 |
| M8_Turn_Wall_Turn    | KI         |                      364 |                 28.8263  |                          0.0702713  |      0.458513 |           2.116   |         0.970214 |
| M8_Turn_Wall_Turn    | WT         |                      251 |                 23.1346  |                          0.0145304  |      0.445658 |           2.3245  |         1.03593  |
| M10_Groom_Rest_Groom | KI         |                       25 |                  8.76019 |                          0.00113836 |      0.543392 |           1.867   |         1.01451  |
| M10_Groom_Rest_Groom | WT         |                      293 |                  8.38618 |                          0.025067   |      0.638565 |           2.73746 |         1.74804  |
| M6_Dash_Redirect     | KI         |                      250 |                 23.4552  |                          0.107067   |      0.560984 |           1.4391  |         0.807313 |
| M6_Dash_Redirect     | WT         |                      122 |                 23.4083  |                          0.0773684  |      0.513168 |           1.33627 |         0.685731 |
| M7_Wall_Exploration  | KI         |                      265 |                 12.578   |                          0.0590455  |      0.453147 |           1.41764 |         0.642399 |
| M7_Wall_Exploration  | WT         |                      138 |                 12.3355  |                          0.0825592  |      0.455559 |           1.38496 |         0.630932 |

### 3.2 Completion Rates
| chain_id             | genotype   |   total_attempts |   completed_count |   aborted_count |   completion_rate_pct |
|:---------------------|:-----------|-----------------:|------------------:|----------------:|----------------------:|
| M1_Rear_Descent      | KI         |             3101 |               298 |            2803 |               9.6098  |
| M1_Rear_Descent      | WT         |             2276 |               173 |            2103 |               7.60105 |
| M5_Slalom_Sprint     | KI         |             4015 |               651 |            3364 |              16.2142  |
| M5_Slalom_Sprint     | WT         |             3318 |               322 |            2996 |               9.70464 |
| M8_Turn_Wall_Turn    | KI         |             3086 |               364 |            2722 |              11.7952  |
| M8_Turn_Wall_Turn    | WT         |             2451 |               251 |            2200 |              10.2407  |
| M10_Groom_Rest_Groom | KI         |              986 |                25 |             961 |               2.5355  |
| M10_Groom_Rest_Groom | WT         |             3539 |               293 |            3246 |               8.27917 |
| M6_Dash_Redirect     | KI         |             2149 |               250 |            1899 |              11.6333  |
| M6_Dash_Redirect     | WT         |             2043 |               122 |            1921 |               5.97161 |
| M7_Wall_Exploration  | KI         |             3372 |               265 |            3107 |               7.85884 |
| M7_Wall_Exploration  | WT         |             1724 |               138 |            1586 |               8.00464 |

### 3.3 Structural Variations (Insertions, Deletions, Stutters)
| chain_id             | genotype   |   total_attempts |   exact_execution_pct |   mean_edit_distance |   mean_insertions |   mean_deletions |   mean_repetitions_stutter |
|:---------------------|:-----------|-----------------:|----------------------:|---------------------:|------------------:|-----------------:|---------------------------:|
| M1_Rear_Descent      | KI         |             3101 |               8.35214 |              1.49274 |       0.0532086   |          1.43954 |                          0 |
| M1_Rear_Descent      | WT         |             2276 |               6.76626 |              1.558   |       0.0342707   |          1.52373 |                          0 |
| M5_Slalom_Sprint     | KI         |             4015 |              16.1893  |              1.40299 |       0.000996264 |          1.40199 |                          0 |
| M5_Slalom_Sprint     | WT         |             3318 |               9.6745  |              1.5868  |       0.000904159 |          1.5859  |                          0 |
| M8_Turn_Wall_Turn    | KI         |             3086 |              11.1471  |              1.29067 |       0.0288399   |          1.26183 |                          0 |
| M8_Turn_Wall_Turn    | WT         |             2451 |               9.79192 |              1.32354 |       0.0281518   |          1.29539 |                          0 |
| M10_Groom_Rest_Groom | KI         |              986 |               2.43408 |              1.81237 |       0.00811359  |          1.80426 |                          0 |
| M10_Groom_Rest_Groom | WT         |             3539 |               7.77056 |              1.69511 |       0.0138457   |          1.68127 |                          0 |
| M6_Dash_Redirect     | KI         |             2149 |              10.7027  |              1.6631  |       0.0134946   |          1.6496  |                          0 |
| M6_Dash_Redirect     | WT         |             2043 |               5.58003 |              1.75869 |       0.0132159   |          1.74547 |                          0 |
| M7_Wall_Exploration  | KI         |             3372 |               7.35469 |              1.71975 |       0.0204626   |          1.69929 |                          0 |
| M7_Wall_Exploration  | WT         |             1724 |               7.65661 |              1.71288 |       0.012761    |          1.70012 |                          0 |

### 3.4 Temporal Stage Pacing Profiles
| chain_id             | genotype   |   stage_index |   stage_token |   stage_mean_duration_s |   stage_std_duration_s |   stage_relative_fraction |
|:---------------------|:-----------|--------------:|--------------:|------------------------:|-----------------------:|--------------------------:|
| M1_Rear_Descent      | KI         |             0 |            14 |                0.6      |               0.60272  |                  0.463934 |
| M1_Rear_Descent      | KI         |             1 |             4 |                0.243624 |               0.177006 |                  0.188376 |
| M1_Rear_Descent      | KI         |             2 |             3 |                0.449664 |               0.406661 |                  0.347691 |
| M1_Rear_Descent      | WT         |             0 |            14 |                0.501156 |               0.440956 |                  0.457098 |
| M1_Rear_Descent      | WT         |             1 |             4 |                0.21763  |               0.144003 |                  0.198497 |
| M1_Rear_Descent      | WT         |             2 |             3 |                0.377601 |               0.303925 |                  0.344405 |
| M5_Slalom_Sprint     | KI         |             0 |             6 |                0.539132 |               0.483201 |                  0.34618  |
| M5_Slalom_Sprint     | KI         |             1 |            17 |                0.640899 |               0.540375 |                  0.411525 |
| M5_Slalom_Sprint     | KI         |             2 |             2 |                0.377343 |               0.323086 |                  0.242294 |
| M5_Slalom_Sprint     | WT         |             0 |             6 |                0.501242 |               0.468518 |                  0.339593 |
| M5_Slalom_Sprint     | WT         |             1 |            17 |                0.632997 |               0.555593 |                  0.428857 |
| M5_Slalom_Sprint     | WT         |             2 |             2 |                0.34177  |               0.306133 |                  0.23155  |
| M8_Turn_Wall_Turn    | KI         |             0 |            11 |                0.826648 |               0.627755 |                  0.398358 |
| M8_Turn_Wall_Turn    | KI         |             1 |             1 |                0.450618 |               0.354283 |                  0.217151 |
| M8_Turn_Wall_Turn    | KI         |             2 |            30 |                0.797871 |               0.583817 |                  0.384491 |
| M8_Turn_Wall_Turn    | WT         |             0 |            11 |                0.860259 |               0.642964 |                  0.375163 |
| M8_Turn_Wall_Turn    | WT         |             1 |             1 |                0.537251 |               0.38151  |                  0.234298 |
| M8_Turn_Wall_Turn    | WT         |             2 |            30 |                0.895518 |               0.60845  |                  0.390539 |
| M10_Groom_Rest_Groom | KI         |             0 |            20 |                0.333    |               0.311827 |                  0.178744 |
| M10_Groom_Rest_Groom | KI         |             1 |            34 |                0.784    |               0.849732 |                  0.420827 |
| M10_Groom_Rest_Groom | KI         |             2 |            49 |                0.746    |               0.797016 |                  0.400429 |
| M10_Groom_Rest_Groom | WT         |             0 |            20 |                0.46058  |               0.386823 |                  0.170531 |
| M10_Groom_Rest_Groom | WT         |             1 |            34 |                0.871843 |               0.878109 |                  0.322803 |
| M10_Groom_Rest_Groom | WT         |             2 |            49 |                1.36843  |               1.52115  |                  0.506666 |
| M6_Dash_Redirect     | KI         |             0 |            21 |                0.7785   |               0.626714 |                  0.558145 |
| M6_Dash_Redirect     | KI         |             1 |             2 |                0.3165   |               0.451603 |                  0.226914 |
| M6_Dash_Redirect     | KI         |             2 |             0 |                0.2998   |               0.261964 |                  0.214941 |
| M6_Dash_Redirect     | WT         |             0 |            21 |                0.707377 |               0.534187 |                  0.536692 |
| M6_Dash_Redirect     | WT         |             1 |             2 |                0.326025 |               0.315391 |                  0.247357 |
| M6_Dash_Redirect     | WT         |             2 |             0 |                0.284631 |               0.292065 |                  0.215951 |
| M7_Wall_Exploration  | KI         |             0 |            16 |                0.592358 |               0.407374 |                  0.423313 |
| M7_Wall_Exploration  | KI         |             1 |            14 |                0.525094 |               0.471777 |                  0.375244 |
| M7_Wall_Exploration  | KI         |             2 |             7 |                0.281887 |               0.184882 |                  0.201443 |
| M7_Wall_Exploration  | WT         |             0 |            16 |                0.575906 |               0.442074 |                  0.423133 |
| M7_Wall_Exploration  | WT         |             1 |            14 |                0.528623 |               0.421883 |                  0.388393 |
| M7_Wall_Exploration  | WT         |             2 |             7 |                0.256522 |               0.173553 |                  0.188473 |

---

## 4. Context-Dependent Movement

![Context Dependent Movement](context_dependent_syllable_kinematics.png)

|   syllable_token | genotype   |   n_in_chain_samples |   in_chain_mean_duration_s |   in_chain_mean_speed |   in_chain_speed_std |
|-----------------:|:-----------|---------------------:|---------------------------:|----------------------:|---------------------:|
|                0 | KI         |                  250 |                   0.2998   |              125.546  |             40.9973  |
|                0 | WT         |                  122 |                   0.284631 |              112.268  |             39.7018  |
|                1 | KI         |                  364 |                   0.450618 |              144.825  |             45.7969  |
|                1 | WT         |                  251 |                   0.537251 |              133.487  |             37.0121  |
|                2 | KI         |                  901 |                   0.360461 |              118.815  |             43.3395  |
|                2 | WT         |                  444 |                   0.337444 |              100.223  |             37.5095  |
|                3 | KI         |                  298 |                   0.449664 |               79.6336 |             23.6132  |
|                3 | WT         |                  173 |                   0.377601 |               71.3571 |             17.0467  |
|                4 | KI         |                  298 |                   0.243624 |               79.3665 |             23.2028  |
|                4 | WT         |                  173 |                   0.21763  |               70.4637 |             15.6204  |
|                6 | KI         |                  651 |                   0.539132 |              116.181  |             43.9618  |
|                6 | WT         |                  322 |                   0.501242 |               95.8772 |             35.8399  |
|                7 | KI         |                  265 |                   0.281887 |               71.2675 |             14.3615  |
|                7 | WT         |                  138 |                   0.256522 |               66.897  |             12.0363  |
|               11 | KI         |                  364 |                   0.826648 |              144.825  |             45.7969  |
|               11 | WT         |                  251 |                   0.860259 |              133.487  |             37.0121  |
|               14 | KI         |                  563 |                   0.564742 |               75.5275 |             19.9683  |
|               14 | WT         |                  311 |                   0.513344 |               68.8983 |             14.2793  |
|               16 | KI         |                  265 |                   0.592358 |               71.5442 |             14.6827  |
|               16 | WT         |                  138 |                   0.575906 |               66.897  |             12.0363  |
|               17 | KI         |                  651 |                   0.640899 |              116.181  |             43.9618  |
|               17 | WT         |                  322 |                   0.632997 |               96.1168 |             36.0418  |
|               20 | KI         |                   25 |                   0.333    |               54.0484 |              8.9765  |
|               20 | WT         |                  293 |                   0.46058  |               52.1971 |              4.17497 |
|               21 | KI         |                  250 |                   0.7785   |              125.755  |             40.9456  |
|               21 | WT         |                  122 |                   0.707377 |              111.694  |             39.3541  |
|               30 | KI         |                  364 |                   0.797871 |              144.825  |             45.7969  |
|               30 | WT         |                  251 |                   0.895518 |              133.487  |             37.0121  |
|               34 | KI         |                   25 |                   0.784    |               54.0484 |              8.9765  |
|               34 | WT         |                  293 |                   0.871843 |               52.1971 |              4.17497 |
|               49 | KI         |                   25 |                   0.746    |               54.0484 |              8.9765  |
|               49 | WT         |                  293 |                   1.36843  |               52.1971 |              4.17497 |

