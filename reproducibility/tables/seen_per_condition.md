# Formal Seen Degradation Per-Condition mAP50

Values are mean +/- std when seeds 0/1/2 are available; otherwise the seed is marked explicitly.

| Condition | YOLO11n Clean | YOLO11n DART | SPhantomNet Clean | SPhantomNet Clean Repeat | SPhantomNet DART | SPhantomNet SDCT | Best (SPhantomNet only) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_light_L1 | 0.884889 +/- 0.006316 | 0.910538 (seed0) | 0.884563 +/- 0.004217 | 0.915585 +/- 0.003454 | 0.908306 +/- 0.002330 | 0.905161 +/- 0.005685 | SPhantomNet Clean Repeat | evaluation result files |
| low_light_L2 | 0.862191 +/- 0.007746 | 0.910113 (seed0) | 0.869731 +/- 0.005921 | 0.893564 +/- 0.006300 | 0.906101 +/- 0.001424 | 0.903805 +/- 0.005450 | SPhantomNet DART | evaluation result files |
| low_light_L3 | 0.806408 +/- 0.013162 | 0.904714 (seed0) | 0.819853 +/- 0.004076 | 0.841981 +/- 0.011555 | 0.901147 +/- 0.002260 | 0.901097 +/- 0.006027 | SPhantomNet DART | evaluation result files |
| low_contrast_L1 | 0.883150 +/- 0.001858 | 0.911156 (seed0) | 0.875927 +/- 0.005040 | 0.911238 +/- 0.004486 | 0.908326 +/- 0.003179 | 0.904838 +/- 0.004413 | SPhantomNet Clean Repeat | evaluation result files |
| low_contrast_L2 | 0.843811 +/- 0.002351 | 0.912322 (seed0) | 0.840867 +/- 0.013001 | 0.864979 +/- 0.010801 | 0.908314 +/- 0.003583 | 0.905255 +/- 0.004396 | SPhantomNet DART | evaluation result files |
| low_contrast_L3 | 0.758932 +/- 0.008519 | 0.903767 (seed0) | 0.758051 +/- 0.015298 | 0.769213 +/- 0.027954 | 0.898782 +/- 0.004775 | 0.897978 +/- 0.003489 | SPhantomNet DART | evaluation result files |
| noise_L1 | 0.803481 +/- 0.009395 | 0.897983 (seed0) | 0.786947 +/- 0.007914 | 0.841995 +/- 0.006295 | 0.895278 +/- 0.000666 | 0.893868 +/- 0.003372 | SPhantomNet DART | evaluation result files |
| noise_L2 | 0.657793 +/- 0.007601 | 0.866592 (seed0) | 0.629374 +/- 0.005437 | 0.694972 +/- 0.019997 | 0.866216 +/- 0.004519 | 0.874888 +/- 0.005137 | SPhantomNet SDCT | evaluation result files |
| noise_L3 | 0.448901 +/- 0.023893 | 0.745872 (seed0) | 0.431371 +/- 0.013762 | 0.483143 +/- 0.009832 | 0.748726 +/- 0.023344 | 0.805567 +/- 0.001905 | SPhantomNet SDCT | evaluation result files |
| motion_blur_L1 | 0.881632 +/- 0.004368 | 0.906718 (seed0) | 0.880266 +/- 0.001233 | 0.910379 +/- 0.002435 | 0.904993 +/- 0.002590 | 0.903743 +/- 0.004042 | SPhantomNet Clean Repeat | evaluation result files |
| motion_blur_L2 | 0.843482 +/- 0.001849 | 0.886561 (seed0) | 0.841752 +/- 0.001597 | 0.884528 +/- 0.005536 | 0.892625 +/- 0.001664 | 0.895436 +/- 0.001481 | SPhantomNet SDCT | evaluation result files |
| motion_blur_L3 | 0.786249 +/- 0.002937 | 0.864777 (seed0) | 0.792662 +/- 0.008134 | 0.832505 +/- 0.009111 | 0.870639 +/- 0.003101 | 0.876755 +/- 0.003307 | SPhantomNet SDCT | evaluation result files |
| mixed_L1 | 0.807480 +/- 0.009510 | 0.895218 (seed0) | 0.801064 +/- 0.004573 | 0.839192 +/- 0.006895 | 0.894982 +/- 0.000520 | 0.894146 +/- 0.001192 | SPhantomNet DART | evaluation result files |
| mixed_L2 | 0.221852 +/- 0.026895 | 0.699514 (seed0) | 0.222138 +/- 0.024285 | 0.370853 +/- 0.019928 | 0.720449 +/- 0.005733 | 0.790762 +/- 0.005025 | SPhantomNet SDCT | evaluation result files |
| mixed_L3 | 0.000000 +/- 0.000000 | 0.041314 (seed0) | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.204271 +/- 0.101948 | 0.275049 +/- 0.006037 | SPhantomNet SDCT | evaluation result files |
