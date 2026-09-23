#!/usr/bin/env python3
"""Evaluate a detector on clean plus the stronger synthetic unseen suite."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean


CONDITIONS = [
    "jpeg_compression_L4",
    "gaussian_blur_L4",
    "defocus_blur_L4",
    "salt_pepper_noise_L4",
    "diagonal_motion_blur_L4",
    "gamma_darkening_L4",
    "mixed_unseen_L4",
    "compound_blur_noise_L4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean and strong-unseen degradation conditions.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--training", default="")
    parser.add_argument("--strong-yaml-dir", default="./data/dart_dut/yamls_strong_unseen")
    parser.add_argument("--clean-yaml", default="./data/dart_dut/yamls/dut_gray_test_clean.yaml")
    parser.add_argument("--repo", default="ultralytics_repo")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name-prefix", default="strong_unseen")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    return parser.parse_args()


def split_conditions(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def condition_type(condition: str) -> str:
    prefixes = [
        "diagonal_motion_blur",
        "salt_pepper_noise",
        "jpeg_compression",
        "gaussian_blur",
        "defocus_blur",
        "gamma_darkening",
        "mixed_unseen",
        "compound_blur_noise",
    ]
    if condition == "clean":
        return "clean"
    for prefix in prefixes:
        if condition.startswith(prefix):
            return prefix
    return "unknown"


def condition_level(condition: str) -> str:
    return condition.rsplit("_", 1)[-1] if condition.rsplit("_", 1)[-1].startswith("L") else ""


def metric_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def format_metric(value: float | str) -> str:
    return f"{float(value):.6f}" if value != "" else ""


def yaml_for_condition(condition: str, yaml_dir: Path) -> Path:
    path = yaml_dir / f"dut_gray_test_{condition}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_markdown(csv_path: Path, rows: list[dict[str, str]]) -> Path:
    md_path = csv_path.with_suffix(".md")
    headers = ["method", "training", "suite", "condition", "type", "level", "mAP50", "mAP50-95", "precision", "recall"]
    keys = ["method", "training", "suite", "condition", "type", "level", "map50", "map5095", "precision", "recall"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[k] for k in keys) + " |\n")
    return md_path


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from ultralytics import YOLO  # noqa: WPS433

    yaml_dir = Path(args.strong_yaml_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_plan = [("clean", "clean", Path(args.clean_yaml).resolve())]
    eval_plan.extend(("strong_unseen", c, yaml_for_condition(c, yaml_dir)) for c in split_conditions(args.conditions))

    model = YOLO(str(Path(args.weights).resolve()))
    rows: list[dict[str, str]] = []
    for suite, condition, data_yaml in eval_plan:
        print(f"\nEvaluating {args.method} / {suite} / {condition}: {data_yaml}")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split="test",
            plots=False,
            save_json=False,
            verbose=False,
            project=args.project,
            name=f"{args.name_prefix}_{condition}",
        )
        box = metrics.box
        values = {
            "method": args.method,
            "training": args.training,
            "suite": suite,
            "condition": condition,
            "type": condition_type(condition),
            "level": condition_level(condition),
            "precision": metric_float(getattr(box, "mp", float("nan"))),
            "recall": metric_float(getattr(box, "mr", float("nan"))),
            "map50": metric_float(getattr(box, "map50", float("nan"))),
            "map5095": metric_float(getattr(box, "map", float("nan"))),
        }
        rows.append({**values, **{k: format_metric(values[k]) for k in ("precision", "recall", "map50", "map5095")}})

    clean = next(row for row in rows if row["condition"] == "clean")
    clean_map50 = float(clean["map50"])
    strong = [row for row in rows if row["suite"] == "strong_unseen"]
    avg_map50 = mean(float(row["map50"]) for row in strong)
    avg_map5095 = mean(float(row["map5095"]) for row in strong)
    avg_recall = mean(float(row["recall"]) for row in strong)
    rows.extend(
        [
            {
                "method": args.method,
                "training": args.training,
                "suite": "strong_unseen",
                "condition": "STRONG_UNSEEN_AVG",
                "type": "summary",
                "level": "",
                "precision": "",
                "recall": f"{avg_recall:.6f}",
                "map50": f"{avg_map50:.6f}",
                "map5095": f"{avg_map5095:.6f}",
            },
            {
                "method": args.method,
                "training": args.training,
                "suite": "strong_unseen",
                "condition": "STRONG_UNSEEN_ROBUSTNESS_RATIO",
                "type": "summary",
                "level": "",
                "precision": "",
                "recall": "",
                "map50": f"{avg_map50 / clean_map50:.6f}" if clean_map50 else "nan",
                "map5095": "",
            },
        ]
    )

    fieldnames = ["method", "training", "suite", "condition", "type", "level", "precision", "recall", "map50", "map5095"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = write_markdown(output_csv, rows)
    print(f"\nCSV: {output_csv}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
