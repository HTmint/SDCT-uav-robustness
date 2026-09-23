#!/usr/bin/env python3
"""Evaluate the generic gray robustness baselines on existing fixed suites."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path("ultralytics_repo")
DEFAULT_RUNS = ROOT / "runs/robust_aug_baselines_200"
DEFAULT_AUGMIX_RUNS = ROOT / "runs/robust_aug_baselines_200_augmix_gpu1"
DEFAULT_OUT = ROOT / "outputs/robust_aug_baselines"
DEFAULT_YAML_DIR = ROOT / "data/dart_dut/yamls"
DEFAULT_STRONG_YAML_DIR = ROOT / "data/dart_dut/yamls_strong_unseen"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    training: str
    prefix: str


SPECS = {
    "randaugment_gray": MethodSpec(
        method="SPhantomNet-Gray RandAugment-Gray",
        training="RandAugment-Gray",
        prefix="sphantomnet_gray_randaugment_gray",
    ),
    "augmix_input_gray": MethodSpec(
        method="SPhantomNet-Gray AugMix-Input-Gray",
        training="AugMix-Input-Gray",
        prefix="sphantomnet_gray_augmix_input_gray",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean/seen/held-out/strong evaluations for new baselines.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--augmix-runs-dir", type=Path, default=DEFAULT_AUGMIX_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--method", choices=["randaugment_gray", "augmix_input_gray", "all"], default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--seen-yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--unseen-yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--strong-yaml-dir", type=Path, default=DEFAULT_STRONG_YAML_DIR)
    parser.add_argument("--clean-yaml", type=Path, default=DEFAULT_YAML_DIR / "dut_gray_test_clean.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run", action="store_true", help="Actually run evaluations. Default only prints status.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--coco-seed0", action="store_true", help="Also run existing COCO AP/AP-S helper for seed0.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Evaluate runs before results.csv reaches --epochs.")
    parser.add_argument("--epochs", type=int, default=200, help="Target training epoch for completion checks.")
    return parser.parse_args()


def seeds_from_text(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def selected_specs(method: str) -> list[MethodSpec]:
    keys = list(SPECS) if method == "all" else [method]
    return [SPECS[key] for key in keys]


def run_dir_for_spec(args: argparse.Namespace, spec: MethodSpec) -> Path:
    if spec.prefix == "sphantomnet_gray_augmix_input_gray" and args.augmix_runs_dir.exists():
        return args.augmix_runs_dir
    return args.runs_dir


def run_or_print(cmd: list[str], cwd: Path, do_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if do_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def should_skip(path: Path, args: argparse.Namespace) -> bool:
    return args.skip_existing and not args.overwrite and path.exists()


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


def main() -> None:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.runs_dir = args.runs_dir.resolve()
    args.augmix_runs_dir = args.augmix_runs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_dir = args.output_dir / "formal_seen"
    seen_unseen_dir = args.output_dir / "ordinary_heldout"
    strong_dir = args.output_dir / "strong_heldout"
    coco_dir = args.output_dir / "coco_small_ap"
    for path in (formal_dir, seen_unseen_dir, strong_dir, coco_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("Robust augmentation evaluation plan")
    print(f"runs:       {args.runs_dir}")
    print(f"augmix:     {args.augmix_runs_dir}")
    print(f"output:     {args.output_dir}")
    print(f"imgsz/batch:{args.imgsz}/{args.batch}")

    for spec in selected_specs(args.method):
        runs_dir = run_dir_for_spec(args, spec)
        for seed in seeds_from_text(args.seeds):
            run_name = f"{spec.prefix}_seed{seed}"
            weights = runs_dir / run_name / "weights/best.pt"
            if not weights.exists():
                print(f"MISSING weights: {weights}", flush=True)
                continue
            epoch = last_epoch(runs_dir / run_name / "results.csv")
            if epoch < args.epochs and not args.allow_incomplete:
                print(f"SKIP incomplete: {run_name} epoch={epoch}/{args.epochs}", flush=True)
                continue

            formal_csv = formal_dir / f"{run_name}.csv"
            if should_skip(formal_csv, args):
                print(f"SKIP existing formal seen: {formal_csv}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(ROOT / "tools/eval_degradation_suite.py"),
                    "--repo",
                    str(args.repo),
                    "--weights",
                    str(weights),
                    "--test-yaml-dir",
                    str(args.seen_yaml_dir),
                    "--clean-yaml",
                    str(args.clean_yaml),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--device",
                    args.device,
                    "--output-csv",
                    str(formal_csv),
                    "--project",
                    str(args.output_dir / "formal_seen_val_runs"),
                    "--name-prefix",
                    run_name,
                ]
                run_or_print(cmd, ROOT, args.run)

            ordinary_csv = seen_unseen_dir / f"{run_name}.csv"
            if should_skip(ordinary_csv, args):
                print(f"SKIP existing ordinary held-out: {ordinary_csv}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(ROOT / "tools/eval_seen_unseen_suite.py"),
                    "--repo",
                    str(args.repo),
                    "--weights",
                    str(weights),
                    "--method",
                    spec.method,
                    "--training",
                    spec.training,
                    "--seen-yaml-dir",
                    str(args.seen_yaml_dir),
                    "--unseen-yaml-dir",
                    str(args.unseen_yaml_dir),
                    "--clean-yaml",
                    str(args.clean_yaml),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--device",
                    args.device,
                    "--output-csv",
                    str(ordinary_csv),
                    "--project",
                    str(args.output_dir / "ordinary_heldout_val_runs"),
                    "--name-prefix",
                    run_name,
                ]
                run_or_print(cmd, ROOT, args.run)

            strong_csv = strong_dir / f"{run_name}.csv"
            if should_skip(strong_csv, args):
                print(f"SKIP existing strong held-out: {strong_csv}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(ROOT / "tools/eval_strong_unseen_suite.py"),
                    "--repo",
                    str(args.repo),
                    "--weights",
                    str(weights),
                    "--method",
                    spec.method,
                    "--training",
                    spec.training,
                    "--strong-yaml-dir",
                    str(args.strong_yaml_dir),
                    "--clean-yaml",
                    str(args.clean_yaml),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--device",
                    args.device,
                    "--output-csv",
                    str(strong_csv),
                    "--project",
                    str(args.output_dir / "strong_heldout_val_runs"),
                    "--name-prefix",
                    run_name,
                ]
                run_or_print(cmd, ROOT, args.run)

            if args.coco_seed0 and seed == 0:
                coco_summary = coco_dir / f"{spec.prefix}_suite_summary.csv"
                if should_skip(coco_summary, args):
                    print(f"SKIP existing COCO helper: {coco_summary}", flush=True)
                else:
                    cmd = [
                        sys.executable,
                        str(ROOT / "tools/eval_coco_small_ap_suite.py"),
                        "--repo",
                        str(args.repo),
                        "--weights",
                        str(weights),
                        "--name-prefix",
                        spec.prefix,
                        "--yaml-dir",
                        str(args.seen_yaml_dir),
                        "--annotation-dir",
                        str(ROOT / "data/dart_dut/DUT-Gray-Degraded-Test/annotations"),
                        "--output-dir",
                        str(coco_dir),
                        "--imgsz",
                        str(args.imgsz),
                        "--batch",
                        "4",
                        "--device",
                        args.device,
                        "--skip-existing",
                    ]
                    run_or_print(cmd, ROOT, args.run)

    print(f"Run summary after evaluations with: {sys.executable} {ROOT / 'tools/summarize_robust_aug_gray_results.py'}")


if __name__ == "__main__":
    main()
