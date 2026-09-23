#!/usr/bin/env python3
"""Evaluate one detector on clean and fixed degradation test YAMLs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List


ORDER = [
    "clean",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the DART degradation suite.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test-yaml-dir", required=True, help="Directory containing dut_gray_test_*.yaml.")
    parser.add_argument("--clean-yaml", required=True, help="Clean test yaml, usually dut_gray_test_clean.yaml.")
    parser.add_argument("--repo", default="ultralytics_repo", help="Modified Ultralytics repo path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--project", default=None, help="Optional Ultralytics val output project.")
    parser.add_argument("--name-prefix", default="eval")
    return parser.parse_args()


def condition_type(condition: str) -> str:
    if condition == "clean":
        return "clean"
    if condition.startswith("low_light"):
        return "low_light"
    if condition.startswith("low_contrast"):
        return "low_contrast"
    if condition.startswith("noise"):
        return "noise"
    if condition.startswith("motion_blur"):
        return "motion_blur"
    if condition.startswith("mixed"):
        return "mixed"
    return "unknown"


def condition_level(condition: str) -> str:
    return condition.rsplit("_", 1)[-1] if condition.endswith(("L1", "L2", "L3")) else ""


def metric_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def write_markdown(csv_path: Path, rows: List[Dict[str, str]]) -> Path:
    md_path = csv_path.with_suffix(".md")
    headers = ["condition", "mAP50", "mAP50-95", "precision", "recall", "AP-S"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        row["condition"],
                        row["map50"],
                        row["map5095"],
                        row["precision"],
                        row["recall"],
                        row["ap_s"],
                    ]
                )
                + " |\n"
            )
    return md_path


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO  # noqa: WPS433

    weights = Path(args.weights).resolve()
    test_yaml_dir = Path(args.test_yaml_dir).resolve()
    clean_yaml = Path(args.clean_yaml).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    yaml_by_condition = {"clean": clean_yaml}
    for condition in ORDER:
        if condition == "clean":
            continue
        yaml_path = test_yaml_dir / f"dut_gray_test_{condition}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing degraded test yaml: {yaml_path}")
        yaml_by_condition[condition] = yaml_path

    model = YOLO(str(weights))
    rows: List[Dict[str, str]] = []
    numeric_rows: List[Dict[str, float | str]] = []
    for condition in ORDER:
        data_yaml = yaml_by_condition[condition]
        print(f"\nEvaluating {condition}: {data_yaml}")
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
            "condition": condition,
            "type": condition_type(condition),
            "level": condition_level(condition),
            "precision": metric_float(getattr(box, "mp", float("nan"))),
            "recall": metric_float(getattr(box, "mr", float("nan"))),
            "map50": metric_float(getattr(box, "map50", float("nan"))),
            "map5095": metric_float(getattr(box, "map", float("nan"))),
            "ap_s": "N/A",
        }
        numeric_rows.append(values)
        rows.append(
            {
                **{k: str(v) for k, v in values.items()},
                "precision": f"{values['precision']:.6f}",
                "recall": f"{values['recall']:.6f}",
                "map50": f"{values['map50']:.6f}",
                "map5095": f"{values['map5095']:.6f}",
            }
        )

    clean = next(r for r in numeric_rows if r["condition"] == "clean")
    degraded = [r for r in numeric_rows if r["condition"] != "clean"]
    avg_map50 = mean(float(r["map50"]) for r in degraded)
    avg_map5095 = mean(float(r["map5095"]) for r in degraded)
    avg_recall = mean(float(r["recall"]) for r in degraded)
    ratio = avg_map50 / float(clean["map50"]) if float(clean["map50"]) else float("nan")

    summary_rows = [
        {
            "condition": "AVG_DEGRADED",
            "type": "summary",
            "level": "",
            "precision": "",
            "recall": f"{avg_recall:.6f}",
            "map50": f"{avg_map50:.6f}",
            "map5095": f"{avg_map5095:.6f}",
            "ap_s": "N/A",
        },
        {
            "condition": "ROBUSTNESS_RATIO",
            "type": "summary",
            "level": "",
            "precision": "",
            "recall": "",
            "map50": f"{ratio:.6f}",
            "map5095": "",
            "ap_s": "N/A",
        },
    ]
    rows.extend(summary_rows)

    fieldnames = ["condition", "type", "level", "precision", "recall", "map50", "map5095", "ap_s"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = write_markdown(output_csv, rows)
    print("\nResults:")
    for row in rows:
        print(f"{row['condition']:<22} mAP50={row['map50']:<10} mAP50-95={row['map5095']:<10} R={row['recall']}")
    print(f"\nCSV: {output_csv}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
