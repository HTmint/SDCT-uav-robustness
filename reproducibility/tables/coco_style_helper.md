# COCO-Style Helper Metrics

Metrics are computed by `tools/eval_coco_small_ap.py` / `tools/eval_coco_small_ap_suite.py` using pycocotools. They are seed0 helper metrics, not Ultralytics default val metrics and not three-seed mean/std.

| Method | Scope | Clean AP | Clean AP50 | Clean AP75 | Clean AP-S | Avg-Degraded AP | Avg-Degraded AP50 | Avg-Degraded AP75 | Avg-Degraded AP-S | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n-Gray Clean | seed0 helper | 0.582488 | 0.871698 | 0.658912 | 0.427934 | 0.416826 | 0.672792 | 0.443524 | 0.270036 | evaluation result files |
| YOLO11n-Gray DART | seed0 helper | 0.617359 | 0.892314 | 0.698000 | 0.474924 | 0.529170 | 0.795366 | 0.583654 | 0.378562 | evaluation result files |
| SPhantomNet-Gray Clean | seed0 helper | 0.582965 | 0.869759 | 0.654343 | 0.412739 | 0.418674 | 0.671112 | 0.444839 | 0.255198 | evaluation result files |
| SPhantomNet-Gray Clean Repeat | seed0 helper | 0.625309 | 0.908348 | 0.703834 | 0.471135 | 0.447807 | 0.691165 | 0.484104 | 0.300098 | evaluation result files |
| SPhantomNet-Gray DART | seed0 helper | 0.617919 | 0.889811 | 0.692300 | 0.463911 | 0.535501 | 0.800157 | 0.593671 | 0.376958 | evaluation result files |
| SPhantomNet-Gray SDCT / Random-Aug | seed0 helper | 0.606489 | 0.888552 | 0.671591 | 0.447828 | 0.544570 | 0.826009 | 0.591390 | 0.376244 | evaluation result files |
