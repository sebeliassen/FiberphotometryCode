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
| `M10_Pause_Scan` | Sniff -> Look -> Pause | Investigative / Scan | `syll_20 -> syll_34 -> syll_49` | Sniff Right (20) -> Look Left Subtle (34) -> Stationary Pause (49) |
| `M6_Dash_Redirect` | Dash & Redirect | Locomotion | `syll_21 -> syll_02 -> syll_00` | Rapid forward dart -> steer -> settling orient |
| `M7_Wall_Exploration` | Wall Exploration | Exploration | `syll_16 -> syll_14 -> syll_07` | Wall sniff -> unsupported rear -> explore perimeter |

---

## 2. Inter-Chain Organization Summary

![Inter-Chain Organization](inter_chain_organization_summary.png)

### 2.1 Perseveration Index & Chain Repeat Rates
| genotype   |   count |     mean |       std |
|:-----------|--------:|---------:|----------:|
| KI         |      14 | 0.271693 | 0.0539424 |
| WT         |      16 | 0.316859 | 0.142695  |

### 2.2 Continuation Hazard Rates Across Successive Cycles
| genotype   | chain_id            |   total_bouts |   mean_cycles_per_bout |   median_bout_duration_s |   p_continue_1_to_2 |   p_continue_2_to_3 |   p_continue_3_to_4 |
|:-----------|:--------------------|--------------:|-----------------------:|-------------------------:|--------------------:|--------------------:|--------------------:|
| KI         | M1_Rear_Descent     |           295 |                1.01017 |                   1.225  |          0.0101695  |            0        |                   0 |
| KI         | M5_Slalom_Sprint    |           641 |                1.0156  |                   1.425  |          0.0156006  |            0        |                   0 |
| KI         | M8_Turn_Wall_Turn   |           360 |                1.01111 |                   2.05   |          0.0111111  |            0        |                   0 |
| KI         | M10_Pause_Scan      |            25 |                1       |                   1.625  |          0          |            0        |                   0 |
| KI         | M6_Dash_Redirect    |           245 |                1.02041 |                   1.35   |          0.0204082  |            0        |                   0 |
| KI         | M7_Wall_Exploration |           263 |                1.0076  |                   1.35   |          0.00760456 |            0        |                   0 |
| WT         | M1_Rear_Descent     |           172 |                1.00581 |                   1.0125 |          0.00581395 |            0        |                   0 |
| WT         | M5_Slalom_Sprint    |           316 |                1.01899 |                   1.4    |          0.0189873  |            0        |                   0 |
| WT         | M8_Turn_Wall_Turn   |           250 |                1.004   |                   2.2375 |          0.004      |            0        |                   0 |
| WT         | M10_Pause_Scan      |           283 |                1.03534 |                   2.35   |          0.0318021  |            0.111111 |                   0 |
| WT         | M6_Dash_Redirect    |           120 |                1.01667 |                   1.2875 |          0.0166667  |            0        |                   0 |
| WT         | M7_Wall_Exploration |           138 |                1       |                   1.275  |          0          |            0        |                   0 |

### 2.3 Syntax Transition Entropy Rate
| genotype   |   count |    mean |      std |
|:-----------|--------:|--------:|---------:|
| KI         |      14 | 1.91863 | 0.240726 |
| WT         |      16 | 1.78588 | 0.32244  |

### 2.4 Initiations: Lead vs. Repetition Kinematics
| chain_id          |   n_multi_bouts |   n_animals_tested |   lead_mean_speed |   repeat_mean_speed |   speed_ratio_lead_to_repeat |   p_speed_diff_paired_mouse |   lead_mean_duration_s |   repeat_mean_duration_s |   duration_ratio_lead_to_repeat |
|:------------------|----------------:|-------------------:|------------------:|--------------------:|-----------------------------:|----------------------------:|-----------------------:|-------------------------:|--------------------------------:|
| M1_Rear_Descent   |               4 |                  4 |           74.0319 |             95.1728 |                     0.777868 |                    0.239029 |               1.225    |                  1.51875 |                        0.806584 |
| M5_Slalom_Sprint  |              16 |                  7 |           98.8482 |            100.931  |                     0.979364 |                    0.544239 |               1.07812  |                  1.7375  |                        0.620504 |
| M8_Turn_Wall_Turn |               5 |                  3 |          143.383  |            129.994  |                     1.10299  |                    0.765824 |               1.905    |                  1.575   |                        1.20952  |
| M10_Pause_Scan    |               9 |                  1 |           52.1251 |             49.9685 |                     1.04316  |                  nan        |               3.94722  |                  4.5175  |                        0.873763 |
| M6_Dash_Redirect  |               7 |                  6 |          127.873  |            119.199  |                     1.07276  |                    0.791641 |               0.882143 |                  1.575   |                        0.560091 |

---

## 3. Intra-Chain Execution Fidelity & Dynamics

![Intra-Chain Execution](intra_chain_execution_summary.png)

### 3.1 Execution Consistency (DTW Distance & Profile Correlation)
| chain_id            | genotype   |   n_complete_occurrences |   n_animals_evaluated |   mean_pairwise_dtw_dist |   mean_velocity_profile_correlation |   duration_cv |   mean_duration_s |   std_duration_s |
|:--------------------|:-----------|-------------------------:|----------------------:|-------------------------:|------------------------------------:|--------------:|------------------:|-----------------:|
| M1_Rear_Descent     | KI         |                      298 |                     7 |                 15.1296  |                          0.127799   |      0.594151 |           1.35596 |         0.805643 |
| M1_Rear_Descent     | WT         |                      173 |                     8 |                 11.6209  |                          0.080728   |      0.501051 |           1.14971 |         0.576064 |
| M5_Slalom_Sprint    | KI         |                      651 |                     7 |                 23.0352  |                          0.0610684  |      0.53618  |           1.5578  |         0.835259 |
| M5_Slalom_Sprint    | WT         |                      322 |                     8 |                 18.98    |                          0.0262723  |      0.574594 |           1.47686 |         0.848597 |
| M8_Turn_Wall_Turn   | KI         |                      364 |                     7 |                 26.7431  |                          0.0305755  |      0.458513 |           2.116   |         0.970214 |
| M8_Turn_Wall_Turn   | WT         |                      251 |                     7 |                 20.5614  |                          0.00517458 |      0.445658 |           2.3245  |         1.03593  |
| M10_Pause_Scan      | KI         |                       25 |                     5 |                  8.25922 |                          0.0513741  |      0.543392 |           1.867   |         1.01451  |
| M10_Pause_Scan      | WT         |                      293 |                     8 |                  7.92387 |                         -0.0194452  |      0.638565 |           2.73746 |         1.74804  |
| M6_Dash_Redirect    | KI         |                      250 |                     7 |                 23.6102  |                          0.0916153  |      0.560984 |           1.4391  |         0.807313 |
| M6_Dash_Redirect    | WT         |                      122 |                     8 |                 25.2728  |                         -0.014231   |      0.513168 |           1.33627 |         0.685731 |
| M7_Wall_Exploration | KI         |                      265 |                     7 |                 12.3538  |                          0.0838815  |      0.453147 |           1.41764 |         0.642399 |
| M7_Wall_Exploration | WT         |                      138 |                     8 |                 11.1905  |                          0.0858695  |      0.455559 |           1.38496 |         0.630932 |

### 3.2 Completion Rates
| chain_id            | genotype   |   total_attempts |   completed_count |   aborted_count |   completion_rate_pct |
|:--------------------|:-----------|-----------------:|------------------:|----------------:|----------------------:|
| M1_Rear_Descent     | KI         |             3101 |               298 |            2803 |               9.6098  |
| M1_Rear_Descent     | WT         |             2276 |               173 |            2103 |               7.60105 |
| M5_Slalom_Sprint    | KI         |             4015 |               651 |            3364 |              16.2142  |
| M5_Slalom_Sprint    | WT         |             3318 |               322 |            2996 |               9.70464 |
| M8_Turn_Wall_Turn   | KI         |             3086 |               364 |            2722 |              11.7952  |
| M8_Turn_Wall_Turn   | WT         |             2451 |               251 |            2200 |              10.2407  |
| M10_Pause_Scan      | KI         |              986 |                25 |             961 |               2.5355  |
| M10_Pause_Scan      | WT         |             3539 |               293 |            3246 |               8.27917 |
| M6_Dash_Redirect    | KI         |             2149 |               250 |            1899 |              11.6333  |
| M6_Dash_Redirect    | WT         |             2043 |               122 |            1921 |               5.97161 |
| M7_Wall_Exploration | KI         |             3372 |               265 |            3107 |               7.85884 |
| M7_Wall_Exploration | WT         |             1724 |               138 |            1586 |               8.00464 |

### 3.3 Structural Variations (Insertions, Deletions, Stutters)
| chain_id            | genotype   |   total_attempts |   exact_execution_pct |   mean_edit_distance |   mean_insertions |   mean_deletions |   mean_repetitions_stutter |
|:--------------------|:-----------|-----------------:|----------------------:|---------------------:|------------------:|-----------------:|---------------------------:|
| M1_Rear_Descent     | KI         |             3101 |               8.35214 |              1.49274 |       0.0532086   |          1.43954 |                          0 |
| M1_Rear_Descent     | WT         |             2276 |               6.76626 |              1.558   |       0.0342707   |          1.52373 |                          0 |
| M5_Slalom_Sprint    | KI         |             4015 |              16.1893  |              1.40299 |       0.000996264 |          1.40199 |                          0 |
| M5_Slalom_Sprint    | WT         |             3318 |               9.6745  |              1.5868  |       0.000904159 |          1.5859  |                          0 |
| M8_Turn_Wall_Turn   | KI         |             3086 |              11.1471  |              1.29067 |       0.0288399   |          1.26183 |                          0 |
| M8_Turn_Wall_Turn   | WT         |             2451 |               9.79192 |              1.32354 |       0.0281518   |          1.29539 |                          0 |
| M10_Pause_Scan      | KI         |              986 |               2.43408 |              1.81237 |       0.00811359  |          1.80426 |                          0 |
| M10_Pause_Scan      | WT         |             3539 |               7.77056 |              1.69511 |       0.0138457   |          1.68127 |                          0 |
| M6_Dash_Redirect    | KI         |             2149 |              10.7027  |              1.6631  |       0.0134946   |          1.6496  |                          0 |
| M6_Dash_Redirect    | WT         |             2043 |               5.58003 |              1.75869 |       0.0132159   |          1.74547 |                          0 |
| M7_Wall_Exploration | KI         |             3372 |               7.35469 |              1.71975 |       0.0204626   |          1.69929 |                          0 |
| M7_Wall_Exploration | WT         |             1724 |               7.65661 |              1.71288 |       0.012761    |          1.70012 |                          0 |

### 3.4 Temporal Stage Pacing Profiles
| chain_id            | genotype   |   stage_index |   stage_token |   stage_mean_duration_s |   stage_std_duration_s |   stage_relative_fraction |
|:--------------------|:-----------|--------------:|--------------:|------------------------:|-----------------------:|--------------------------:|
| M1_Rear_Descent     | KI         |             0 |            14 |                0.6      |               0.60272  |                  0.463934 |
| M1_Rear_Descent     | KI         |             1 |             4 |                0.243624 |               0.177006 |                  0.188376 |
| M1_Rear_Descent     | KI         |             2 |             3 |                0.449664 |               0.406661 |                  0.347691 |
| M1_Rear_Descent     | WT         |             0 |            14 |                0.501156 |               0.440956 |                  0.457098 |
| M1_Rear_Descent     | WT         |             1 |             4 |                0.21763  |               0.144003 |                  0.198497 |
| M1_Rear_Descent     | WT         |             2 |             3 |                0.377601 |               0.303925 |                  0.344405 |
| M5_Slalom_Sprint    | KI         |             0 |             6 |                0.539132 |               0.483201 |                  0.34618  |
| M5_Slalom_Sprint    | KI         |             1 |            17 |                0.640899 |               0.540375 |                  0.411525 |
| M5_Slalom_Sprint    | KI         |             2 |             2 |                0.377343 |               0.323086 |                  0.242294 |
| M5_Slalom_Sprint    | WT         |             0 |             6 |                0.501242 |               0.468518 |                  0.339593 |
| M5_Slalom_Sprint    | WT         |             1 |            17 |                0.632997 |               0.555593 |                  0.428857 |
| M5_Slalom_Sprint    | WT         |             2 |             2 |                0.34177  |               0.306133 |                  0.23155  |
| M8_Turn_Wall_Turn   | KI         |             0 |            11 |                0.826648 |               0.627755 |                  0.398358 |
| M8_Turn_Wall_Turn   | KI         |             1 |             1 |                0.450618 |               0.354283 |                  0.217151 |
| M8_Turn_Wall_Turn   | KI         |             2 |            30 |                0.797871 |               0.583817 |                  0.384491 |
| M8_Turn_Wall_Turn   | WT         |             0 |            11 |                0.860259 |               0.642964 |                  0.375163 |
| M8_Turn_Wall_Turn   | WT         |             1 |             1 |                0.537251 |               0.38151  |                  0.234298 |
| M8_Turn_Wall_Turn   | WT         |             2 |            30 |                0.895518 |               0.60845  |                  0.390539 |
| M10_Pause_Scan      | KI         |             0 |            20 |                0.333    |               0.311827 |                  0.178744 |
| M10_Pause_Scan      | KI         |             1 |            34 |                0.784    |               0.849732 |                  0.420827 |
| M10_Pause_Scan      | KI         |             2 |            49 |                0.746    |               0.797016 |                  0.400429 |
| M10_Pause_Scan      | WT         |             0 |            20 |                0.46058  |               0.386823 |                  0.170531 |
| M10_Pause_Scan      | WT         |             1 |            34 |                0.871843 |               0.878109 |                  0.322803 |
| M10_Pause_Scan      | WT         |             2 |            49 |                1.36843  |               1.52115  |                  0.506666 |
| M6_Dash_Redirect    | KI         |             0 |            21 |                0.7785   |               0.626714 |                  0.558145 |
| M6_Dash_Redirect    | KI         |             1 |             2 |                0.3165   |               0.451603 |                  0.226914 |
| M6_Dash_Redirect    | KI         |             2 |             0 |                0.2998   |               0.261964 |                  0.214941 |
| M6_Dash_Redirect    | WT         |             0 |            21 |                0.707377 |               0.534187 |                  0.536692 |
| M6_Dash_Redirect    | WT         |             1 |             2 |                0.326025 |               0.315391 |                  0.247357 |
| M6_Dash_Redirect    | WT         |             2 |             0 |                0.284631 |               0.292065 |                  0.215951 |
| M7_Wall_Exploration | KI         |             0 |            16 |                0.592358 |               0.407374 |                  0.423313 |
| M7_Wall_Exploration | KI         |             1 |            14 |                0.525094 |               0.471777 |                  0.375244 |
| M7_Wall_Exploration | KI         |             2 |             7 |                0.281887 |               0.184882 |                  0.201443 |
| M7_Wall_Exploration | WT         |             0 |            16 |                0.575906 |               0.442074 |                  0.423133 |
| M7_Wall_Exploration | WT         |             1 |            14 |                0.528623 |               0.421883 |                  0.388393 |
| M7_Wall_Exploration | WT         |             2 |             7 |                0.256522 |               0.173553 |                  0.188473 |

---

## 4. Context-Dependent Movement

![Context Dependent Movement](context_dependent_syllable_kinematics.png)

|   syllable_token | genotype   |   n_in_chain_samples |   n_outside_chain_samples |   in_chain_mean_duration_s |   outside_chain_mean_duration_s |   duration_modulation_ratio |   in_chain_mean_speed |   outside_chain_mean_speed |   speed_modulation_ratio |   in_chain_speed_std |
|-----------------:|:-----------|---------------------:|--------------------------:|---------------------------:|--------------------------------:|----------------------------:|----------------------:|---------------------------:|-------------------------:|---------------------:|
|                0 | KI         |                  250 |                      7808 |                   0.2998   |                        0.288617 |                    1.03875  |               97.8119 |                   106.195  |                 0.921064 |             43.334   |
|                0 | WT         |                  122 |                      6194 |                   0.284631 |                        0.283436 |                    1.00422  |               89.0412 |                    93.1464 |                 0.955927 |             40.519   |
|                1 | KI         |                  364 |                      6138 |                   0.450618 |                        0.389777 |                    1.15609  |               94.2364 |                    94.2969 |                 0.999359 |             45.9814  |
|                1 | WT         |                  251 |                      4565 |                   0.537251 |                        0.405159 |                    1.32603  |               97.8971 |                    84.8655 |                 1.15356  |             41.5909  |
|                2 | KI         |                  901 |                      4653 |                   0.360461 |                        0.313212 |                    1.15085  |               93.5336 |                    85.3693 |                 1.09564  |             49.3072  |
|                2 | WT         |                  444 |                      3333 |                   0.337444 |                        0.32488  |                    1.03867  |               81.2517 |                    72.072  |                 1.12737  |             39.0525  |
|                3 | KI         |                  298 |                      4736 |                   0.449664 |                        0.454128 |                    0.990171 |              107.172  |                    98.1955 |                 1.09142  |             53.5936  |
|                3 | WT         |                  173 |                      4106 |                   0.377601 |                        0.441165 |                    0.855917 |               84.2643 |                    80.2338 |                 1.05023  |             34.0062  |
|                4 | KI         |                  298 |                      4966 |                   0.243624 |                        0.240646 |                    1.01237  |               79.1489 |                    86.4141 |                 0.915925 |             28.6775  |
|                4 | WT         |                  173 |                      3519 |                   0.21763  |                        0.231884 |                    0.93853  |               71.4808 |                    77.0571 |                 0.927635 |             23.8187  |
|                6 | KI         |                  651 |                      3364 |                   0.539132 |                        0.581674 |                    0.926864 |               94.8576 |                    87.1371 |                 1.0886   |             47.2266  |
|                6 | WT         |                  322 |                      2996 |                   0.501242 |                        0.559296 |                    0.896203 |               80.8245 |                    76.5286 |                 1.05613  |             37.9472  |
|                7 | KI         |                  265 |                      4264 |                   0.281887 |                        0.256279 |                    1.09992  |               94.372  |                    89.3685 |                 1.05599  |             32.8682  |
|                7 | WT         |                  138 |                      2614 |                   0.256522 |                        0.226655 |                    1.13177  |               87.3125 |                    78.6331 |                 1.11038  |             29.4825  |
|               11 | KI         |                  364 |                      2722 |                   0.826648 |                        0.706686 |                    1.16975  |              156.018  |                   139.795  |                 1.11605  |             68.9411  |
|               11 | WT         |                  251 |                      2200 |                   0.860259 |                        0.797125 |                    1.0792   |              140.906  |                   124.991  |                 1.12733  |             54.0526  |
|               14 | KI         |                  563 |                      2538 |                   0.564742 |                        0.552059 |                    1.02298  |               61.0773 |                    61.6282 |                 0.991061 |             16.6433  |
|               14 | WT         |                  311 |                      1965 |                   0.513344 |                        0.553766 |                    0.927006 |               60.8619 |                    59.8385 |                 1.0171   |             14.7985  |
|               16 | KI         |                  265 |                      3104 |                   0.592358 |                        0.714997 |                    0.828477 |               67.1137 |                    68.0498 |                 0.986245 |             21.7432  |
|               16 | WT         |                  138 |                      1586 |                   0.575906 |                        0.638635 |                    0.901776 |               65.9424 |                    65.6675 |                 1.00419  |             28.2273  |
|               17 | KI         |                  651 |                      2313 |                   0.640899 |                        0.591342 |                    1.0838   |              146.485  |                   161.149  |                 0.909005 |             68.5976  |
|               17 | WT         |                  322 |                      1428 |                   0.632997 |                        0.581653 |                    1.08827  |              114.754  |                   133.858  |                 0.85728  |             52.6784  |
|               20 | KI         |                   25 |                       961 |                   0.333    |                        0.443652 |                    0.750588 |               60.2549 |                    61.8027 |                 0.974955 |             24.6325  |
|               20 | WT         |                  293 |                      3246 |                   0.46058  |                        0.463124 |                    0.994508 |               55.3157 |                    57.9621 |                 0.954342 |             11.6627  |
|               21 | KI         |                  250 |                      1899 |                   0.7785   |                        0.662033 |                    1.17592  |              151.876  |                   145.302  |                 1.04525  |             63.6299  |
|               21 | WT         |                  122 |                      1921 |                   0.707377 |                        0.779906 |                    0.907003 |              133.088  |                   128.411  |                 1.03642  |             54.3978  |
|               30 | KI         |                  364 |                      1557 |                   0.797871 |                        0.782739 |                    1.01933  |              156.278  |                   168.227  |                 0.928971 |             59.5877  |
|               30 | WT         |                  251 |                       598 |                   0.895518 |                        0.681982 |                    1.31311  |              145.344  |                   130.059  |                 1.11752  |             50.3733  |
|               34 | KI         |                   25 |                       294 |                   0.784    |                        0.809269 |                    0.968776 |               51.4661 |                    53.9861 |                 0.953321 |              8.85356 |
|               34 | WT         |                  293 |                      1518 |                   0.871843 |                        0.921443 |                    0.946172 |               51.7389 |                    52.1038 |                 0.992996 |              7.97583 |
|               49 | KI         |                   25 |                        67 |                   0.746    |                        0.954851 |                    0.781274 |               52.6085 |                    50.2364 |                 1.04722  |              8.25393 |
|               49 | WT         |                  293 |                       748 |                   1.36843  |                        1.56126  |                    0.876489 |               50.6355 |                    51.4283 |                 0.984583 |              7.03285 |

