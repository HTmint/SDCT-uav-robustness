#!/usr/bin/env python3
"""Train RandAugment-Gray and AugMix-Input-Gray baselines.

This runner is intentionally separate from the older baseline runner so the new
generic robustness baselines cannot overwrite existing DART/SDCT runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path("ultralytics_repo")
DEFAULT_MODEL = ROOT / "configs/dart_models/sphantomnet_gray.yaml"
DEFAULT_DATA_DIR = ROOT / "data/dart_dut/yamls"
DEFAULT_PROJECT = ROOT / "runs/robust_aug_baselines_200"


@dataclass(frozen=True)
class TrainSpec:
    method: str
    key: str
    data_yaml: Path
    name_prefix: str


SPECS = {
    "randaugment_gray": TrainSpec(
        method="SPhantomNet-Gray RandAugment-Gray",
        key="randaugment_gray",
        data_yaml=DEFAULT_DATA_DIR / "dut_gray_randaugment_gray.yaml",
        name_prefix="sphantomnet_gray_randaugment_gray",
    ),
    "augmix_input_gray": TrainSpec(
        method="SPhantomNet-Gray AugMix-Input-Gray",
        key="augmix_input_gray",
        data_yaml=DEFAULT_DATA_DIR / "dut_gray_augmix_input_gray.yaml",
        name_prefix="sphantomnet_gray_augmix_input_gray",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generic gray robustness baselines.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--method", choices=["randaugment_gray", "augmix_input_gray", "all"], default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run", action="store_true", help="Actually train. Default only prints status.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose last epoch is already target.")
    return parser.parse_args()


def read_yaml_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def validate_model_yaml(path: Path) -> None:
    if path.suffix.lower() != ".yaml":
        raise ValueError(f"Model must be a YAML for random initialization, got: {path}")
    text = read_yaml_text(path)
    if "ch: 1" not in text:
        raise ValueError(f"Model YAML does not explicitly contain ch: 1: {path}")


def validate_data_yaml(path: Path) -> None:
    text = read_yaml_text(path)
    if "ch: 1" not in text or "channels: 1" not in text:
        raise ValueError(f"Data YAML must contain ch: 1 and channels: 1: {path}")


def seeds_from_text(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def last_epoch(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    for row in reversed(rows):
        if not row or row[0].strip().lower() == "epoch":
            continue
        try:
            return int(float(row[0]))
        except ValueError:
            return 0
    return 0


def selected_specs(args: argparse.Namespace) -> list[TrainSpec]:
    keys = list(SPECS) if args.method == "all" else [args.method]
    specs: list[TrainSpec] = []
    for key in keys:
        spec = SPECS[key]
        specs.append(
            TrainSpec(
                method=spec.method,
                key=spec.key,
                data_yaml=args.data_dir / spec.data_yaml.name,
                name_prefix=spec.name_prefix,
            )
        )
    return specs


def validate(args: argparse.Namespace, specs: list[TrainSpec]) -> None:
    if not args.repo.exists():
        raise FileNotFoundError(args.repo)
    validate_model_yaml(args.model)
    for spec in specs:
        validate_data_yaml(spec.data_yaml)
    args.project.mkdir(parents=True, exist_ok=True)


def train_one(seed: int, spec: TrainSpec, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    run_name = f"{spec.name_prefix}_seed{seed}"
    run_dir = args.project / run_name
    results_csv = run_dir / "results.csv"
    last_pt = run_dir / "weights/last.pt"
    epoch = last_epoch(results_csv)
    if epoch >= args.epochs:
        print(f"SKIP complete: {run_name} epoch={epoch}", flush=True)
        return
    if last_pt.exists():
        raise RuntimeError(
            "Partial run exists with weights/last.pt, but this baseline protocol forbids loading last.pt. "
            f"Inspect manually and choose a new run name or remove the incomplete run only if appropriate: {run_dir}"
        )
    if run_dir.exists():
        allowed = {"args.yaml", "train_batch0.jpg", "weights"}
        existing = {p.name for p in run_dir.iterdir()}
        weights_dir = run_dir / "weights"
        bootstrap_only = (
            results_csv.exists() is False
            and existing.issubset(allowed)
            and (not weights_dir.exists() or not any(weights_dir.iterdir()))
        )
        if not bootstrap_only:
            raise RuntimeError(f"Run directory exists without weights/last.pt; inspect before reusing: {run_dir}")
        print(f"REUSE_BOOTSTRAP: {run_name} existing startup directory will be reused", flush=True)
    else:
        bootstrap_only = False

    print(f"START: {run_name}", flush=True)
    model = YOLO(str(args.model))
    model.train(
        data=str(spec.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        amp=False,
        project=str(args.project),
        name=run_name,
        device=args.device,
        seed=seed,
        deterministic=True,
        nwdloss=False,
        pretrained=False,
        exist_ok=bootstrap_only,
    )


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.model = args.model.resolve()
    args.data_dir = args.data_dir.resolve()
    args.project = args.project.resolve()
    specs = selected_specs(args)
    validate(args, specs)
    sys.path.insert(0, str(args.repo))

    print("Robust augmentation training plan")
    print(f"repo:       {args.repo}")
    print(f"model:      {args.model}")
    print(f"project:    {args.project}")
    print(f"epochs:     {args.epochs}")
    print(f"batch:      {args.batch}")
    print(f"imgsz:      {args.imgsz}")
    print("init:       YAML random initialization, pretrained=False")
    for seed in seeds_from_text(args.seeds):
        for spec in specs:
            run_name = f"{spec.name_prefix}_seed{seed}"
            epoch = last_epoch(args.project / run_name / "results.csv")
            last_pt = args.project / run_name / "weights/last.pt"
            state = "complete" if epoch >= args.epochs else "partial" if last_pt.exists() else "missing"
            print(f"{state.upper()}: {run_name} epoch={epoch} data={spec.data_yaml}")
            if args.skip_existing and state == "complete":
                continue
            if args.run:
                train_one(seed, spec, args)


if __name__ == "__main__":
    main()
