#!/usr/bin/env python3
"""Evaluate existing grayscale detectors on the DUT-Dve test split only."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


DEFAULT_DATA = Path("outputs/dut_dve_zero_shot/gray_test/dut_dve_gray_test.yaml")
DEFAULT_OUTPUT = Path("outputs/dut_dve_zero_shot/eval")
DEFAULT_REPO = Path("ultralytics_repo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="METHOD|SEED|WEIGHTS",
        help="repeatable model specification, e.g. 'SPhantomNet-Gray DART|seed0|runs/dart_seed0/weights/best.pt'",
    )
    return parser.parse_args()


def parse_models(values: list[str] | None) -> list[tuple[str, str, Path]]:
    if values is None:
        raise ValueError("Provide at least one --model METHOD|SEED|WEIGHTS.")
    models: list[tuple[str, str, Path]] = []
    for value in values:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid --model {value!r}; expected METHOD|SEED|WEIGHTS")
        models.append((parts[0], parts[1], Path(parts[2])))
    return models


def metric(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "nan"


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "seed", "mAP50", "mAP50-95", "precision", "recall", "status"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(row[column] for column in columns) + " |\n")


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)
    models = parse_models(args.model)
    for _, _, weights in models:
        if not weights.expanduser().resolve().exists():
            raise FileNotFoundError(weights)

    sys.path.insert(0, str(args.repo.resolve()))
    from ultralytics import YOLO  # noqa: WPS433

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project = args.project.resolve() if args.project else None
    rows: list[dict[str, str]] = []
    for method, seed, weights in models:
        print(f"\nEvaluating {method} {seed}: {weights}", flush=True)
        model = YOLO(str(weights.resolve()))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split="test",
            plots=False,
            save_json=False,
            verbose=False,
            project=str(project) if project else None,
            name=f"dut_dve_zero_shot_{method.lower().replace(' ', '_').replace('-', '_')}_{seed}",
        )
        box = metrics.box
        rows.append(
            {
                "method": method,
                "seed": seed,
                "weights": str(weights.resolve()),
                "mAP50": metric(getattr(box, "map50", "nan")),
                "mAP50-95": metric(getattr(box, "map", "nan")),
                "precision": metric(getattr(box, "mp", "nan")),
                "recall": metric(getattr(box, "mr", "nan")),
                "status": "completed",
            }
        )

    csv_path = output_dir / "dut_dve_zero_shot_per_seed.csv"
    fields = ["method", "seed", "weights", "mAP50", "mAP50-95", "precision", "recall", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, str]] = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary = {"method": method, "seed": "mean +/- std", "weights": ""}
        for key in ("mAP50", "mAP50-95", "precision", "recall"):
            values = [float(row[key]) for row in method_rows]
            summary[key] = f"{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}" if len(values) > 1 else f"{values[0]:.6f}"
        summary["status"] = "completed"
        summary_rows.append(summary)

    summary_csv = output_dir / "dut_dve_zero_shot_mean_std.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown(output_dir / "dut_dve_zero_shot_mean_std.md", summary_rows)
    print(f"\nPer-seed CSV: {csv_path}")
    print(f"Mean/std CSV: {summary_csv}")


if __name__ == "__main__":
    main()
