#!/usr/bin/env python3
"""Evaluate a detector on clean, seen degradation, and unseen degradation suites."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List


SEEN_CONDITIONS = [
    "low_light_L1",
    "low_light_L2",
    "low_light_L3",
    "low_contrast_L1",
    "low_contrast_L2",
    "low_contrast_L3",
    "noise_L1",
    "noise_L2",
    "noise_L3",
    "motion_blur_L1",
    "motion_blur_L2",
    "motion_blur_L3",
    "mixed_L1",
    "mixed_L2",
    "mixed_L3",
]

UNSEEN_CONDITIONS = [
    "jpeg_compression_L1",
    "jpeg_compression_L2",
    "jpeg_compression_L3",
    "gaussian_blur_L1",
    "gaussian_blur_L2",
    "gaussian_blur_L3",
    "defocus_blur_L1",
    "defocus_blur_L2",
    "defocus_blur_L3",
    "salt_pepper_noise_L1",
    "salt_pepper_noise_L2",
    "salt_pepper_noise_L3",
    "diagonal_motion_blur_L1",
    "diagonal_motion_blur_L2",
    "diagonal_motion_blur_L3",
    "gamma_darkening_L1",
    "gamma_darkening_L2",
    "gamma_darkening_L3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean, seen, and unseen grayscale degradation suites.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--training", default="")
    parser.add_argument("--seen-yaml-dir", required=True)
    parser.add_argument("--unseen-yaml-dir", required=True)
    parser.add_argument("--clean-yaml", default="./data/dart_dut/yamls/dut_gray_test_clean.yaml")
    parser.add_argument("--repo", default="ultralytics_repo")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name-prefix", default="seen_unseen")
    parser.add_argument("--seen-conditions", default=",".join(SEEN_CONDITIONS))
    parser.add_argument("--unseen-conditions", default=",".join(UNSEEN_CONDITIONS))
    return parser.parse_args()


def split_conditions(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def condition_type(condition: str) -> str:
    if condition == "clean":
        return "clean"
    prefixes = [
        "diagonal_motion_blur",
        "salt_pepper_noise",
        "jpeg_compression",
        "gaussian_blur",
        "defocus_blur",
        "gamma_darkening",
        "low_contrast",
        "motion_blur",
        "low_light",
        "noise",
        "mixed",
    ]
    for prefix in prefixes:
        if condition.startswith(prefix):
            return prefix
    return "unknown"


def condition_level(condition: str) -> str:
    return condition.rsplit("_", 1)[-1] if condition.endswith(("L1", "L2", "L3")) else ""


def metric_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def yaml_for_condition(condition: str, yaml_dir: Path) -> Path:
    path = yaml_dir / f"dut_gray_test_{condition}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing condition YAML: {path}")
    return path


def format_row(values: Dict[str, float | str]) -> Dict[str, str]:
    row = {key: str(value) for key, value in values.items()}
    for key in ("precision", "recall", "map50", "map5095"):
        value = values.get(key, "")
        row[key] = f"{float(value):.6f}" if value != "" else ""
    return row


def write_markdown(csv_path: Path, rows: List[Dict[str, str]]) -> Path:
    md_path = csv_path.with_suffix(".md")
    headers = ["suite", "condition", "type", "level", "mAP50", "mAP50-95", "precision", "recall", "AP-S"]
    keys = ["suite", "condition", "type", "level", "map50", "map5095", "precision", "recall", "ap_s"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[key] for key in keys) + " |\n")
    return md_path


def summary_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    clean = next(row for row in rows if row["condition"] == "clean")
    clean_map50 = float(clean["map50"])
    out: List[Dict[str, str]] = []
    for suite, label in (("seen", "SEEN_AVG"), ("unseen", "UNSEEN_AVG")):
        suite_rows = [row for row in rows if row["suite"] == suite]
        avg_map50 = mean(float(row["map50"]) for row in suite_rows)
        avg_map5095 = mean(float(row["map5095"]) for row in suite_rows)
        avg_recall = mean(float(row["recall"]) for row in suite_rows)
        out.append(
            {
                "method": clean["method"],
                "training": clean["training"],
                "suite": suite,
                "condition": label,
                "type": "summary",
                "level": "",
                "precision": "",
                "recall": f"{avg_recall:.6f}",
                "map50": f"{avg_map50:.6f}",
                "map5095": f"{avg_map5095:.6f}",
                "ap_s": "N/A",
            }
        )
        out.append(
            {
                "method": clean["method"],
                "training": clean["training"],
                "suite": suite,
                "condition": f"{suite.upper()}_ROBUSTNESS_RATIO",
                "type": "summary",
                "level": "",
                "precision": "",
                "recall": "",
                "map50": f"{avg_map50 / clean_map50:.6f}" if clean_map50 else "nan",
                "map5095": "",
                "ap_s": "N/A",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO  # noqa: WPS433

    seen_yaml_dir = Path(args.seen_yaml_dir).resolve()
    unseen_yaml_dir = Path(args.unseen_yaml_dir).resolve()
    clean_yaml = Path(args.clean_yaml).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_plan = [("clean", "clean", clean_yaml)]
    eval_plan.extend(("seen", condition, yaml_for_condition(condition, seen_yaml_dir)) for condition in split_conditions(args.seen_conditions))
    eval_plan.extend(
        ("unseen", condition, yaml_for_condition(condition, unseen_yaml_dir)) for condition in split_conditions(args.unseen_conditions)
    )

    model = YOLO(str(Path(args.weights).resolve()))
    rows: List[Dict[str, str]] = []
    for suite, condition, data_yaml in eval_plan:
        print(f"\nEvaluating {suite}/{condition}: {data_yaml}")
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
            name=f"{args.name_prefix}_{suite}_{condition}",
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
            "ap_s": "N/A",
        }
        rows.append(format_row(values))

    rows.extend(summary_rows(rows))
    fieldnames = ["method", "training", "suite", "condition", "type", "level", "precision", "recall", "map50", "map5095", "ap_s"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = write_markdown(output_csv, rows)
    print(f"\nCSV: {output_csv}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
