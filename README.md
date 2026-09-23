# UAV Grayscale Robustness Experiments

Code for grayscale UAV detection experiments with deterministic degradation training (DART), stochastic degradation coverage training (SDCT), and target-domain evaluation.

## Setup

Use Python 3.10 and install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your hardware separately. Training and detector evaluation require the project-modified Ultralytics 8.3.138 code used in the experiments. It is not bundled here; pass its directory with `--repo`. Dataset images and model weights are also not included.

## Basic workflow

1. Place the source YOLO dataset locally, then build the grayscale data and fixed degradation sets:

```bash
python tools/build_dut_gray_degradation.py \
  --src data/RGBanti1280/RGBanti1280.yaml \
  --output-root data/dart_dut \
  --yaml-dir data/dart_dut/yamls \
  --dataset-name DUT \
  --classes uav \
  --seed 2026
```

2. Build the SDCT training set:

```bash
python tools/build_random_aug_train.py \
  --data-root data/dart_dut \
  --yaml-dir data/dart_dut/yamls \
  --seed 2026
```

3. Train a model from its YAML configuration:

```bash
python tools/train_yolo_single.py \
  --repo ultralytics_repo \
  --model configs/dart_models/sphantomnet_gray.yaml \
  --data data/dart_dut/yamls/dut_gray_random_aug.yaml \
  --project runs \
  --name sdct_seed0 \
  --epochs 200 \
  --batch 16 \
  --device 0 \
  --seed 0
```

For a DART run, use `data/dart_dut/yamls/dut_gray_dart.yaml` in place of the SDCT YAML and change the run name.

4. Evaluate on the clean and seen degradation tests:

```bash
python tools/eval_degradation_suite.py \
  --repo ultralytics_repo \
  --weights runs/sdct_seed0/weights/best.pt \
  --test-yaml-dir data/dart_dut/yamls \
  --clean-yaml data/dart_dut/yamls/dut_gray_test_clean.yaml \
  --output-csv outputs/sdct_seed0.csv
```

The remaining scripts build ablations and held-out test suites, evaluate the supplied DUT-Adv test split, summarize metrics, and benchmark existing TensorRT engines. Pass `--help` to any script for its options. Generated data, runs, weights, and outputs are excluded from version control.

## Reproducibility files

Fixed split lists, dataset YAMLs, offline generation manifests, and the paper-facing result tables are collected under `reproducibility/`. YAML dataset roots resolve to the repository root, and split-list entries are relative to that root. Dataset images and labels, model weights, and raw deployment logs are not included; obtain the source data separately and follow `reproducibility/README.md` for the expected layout.

For the target-domain test, first create a grayscale copy of the supplied test split:

```bash
python tools/build_dut_dve_gray_test.py \
  --source-root data/DUT-Dve \
  --output-root outputs/dut_dve_zero_shot/gray_test
```

Then evaluate checkpoints with one or more `--model METHOD|SEED|WEIGHTS` arguments:

```bash
python tools/eval_dut_dve_zero_shot.py \
  --repo ultralytics_repo \
  --data outputs/dut_dve_zero_shot/gray_test/dut_dve_gray_test.yaml \
  --model 'SPhantomNet-Gray DART|seed0|runs/dart_seed0/weights/best.pt'
```
