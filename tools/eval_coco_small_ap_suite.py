#!/usr/bin/env python3
"""Run COCO AP-S helper over the full DART clean/degraded test suite."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List

from eval_coco_small_ap import evaluate_coco, load_image_ids, read_images, write_predictions


CONDITIONS = [
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
    parser = argparse.ArgumentParser(description="Compute COCO AP metrics across the DART test suite.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name-prefix", required=True, help="Output filename prefix, e.g. sphantomnet_gray_dart.")
    parser.add_argument("--yaml-dir", required=True, help="Directory containing test_<condition>.txt files.")
    parser.add_argument("--annotation-dir", required=True, help="Directory containing instances_test_<condition>.json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", default="ultralytics_repo")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="Comma-separated condition list.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def condition_type(condition: str) -> str:
    if condition == "clean":
        return "clean"
    for prefix in ("low_light", "low_contrast", "noise", "motion_blur", "mixed"):
        if condition.startswith(prefix):
            return prefix
    return "unknown"


def condition_level(condition: str) -> str:
    return condition.rsplit("_", 1)[-1] if condition.endswith(("L1", "L2", "L3")) else ""


def read_metric_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one metric row in {path}, got {len(rows)}")
    return rows[0]


def write_metric_csv(path: Path, metrics: Dict[str, float], pred_count: int, image_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["images", "detections", *metrics.keys()])
        writer.writeheader()
        writer.writerow(
            {
                "images": image_count,
                "detections": pred_count,
                **{key: f"{value:.6f}" for key, value in metrics.items()},
            }
        )


def write_summary(output_dir: Path, name_prefix: str, rows: List[Dict[str, str]]) -> None:
    degraded = [row for row in rows if row["condition"] != "clean"]
    summary_rows = list(rows)
    summary_rows.append(
        {
            "condition": "AVG_DEGRADED",
            "type": "summary",
            "level": "",
            "images": "",
            "detections": "",
            "AP": f"{mean(float(row['AP']) for row in degraded):.6f}",
            "AP50": f"{mean(float(row['AP50']) for row in degraded):.6f}",
            "AP75": f"{mean(float(row['AP75']) for row in degraded):.6f}",
            "AP-S": f"{mean(float(row['AP-S']) for row in degraded):.6f}",
            "AP-M": f"{mean(float(row['AP-M']) for row in degraded):.6f}",
            "AP-L": f"{mean(float(row['AP-L']) for row in degraded):.6f}",
        }
    )

    headers = ["condition", "type", "level", "images", "detections", "AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L"]
    csv_path = output_dir / f"{name_prefix}_suite_summary.csv"
    md_path = output_dir / f"{name_prefix}_suite_summary.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in summary_rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary MD: {md_path}")


def main() -> None:
    args = parse_args()
    yaml_dir = Path(args.yaml_dir).resolve()
    annotation_dir = Path(args.annotation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    rows: List[Dict[str, str]] = []
    for condition in conditions:
        images_txt = yaml_dir / f"test_{condition}.txt"
        gt_json = annotation_dir / f"instances_test_{condition}.json"
        metric_csv = output_dir / f"{args.name_prefix}_{condition}_ap.csv"
        pred_json = output_dir / f"{args.name_prefix}_{condition}_predictions.json"

        if args.skip_existing and metric_csv.exists() and pred_json.exists():
            print(f"Skipping existing {condition}: {metric_csv}")
            metric_row = read_metric_csv(metric_csv)
        else:
            print(f"\nEvaluating {condition}")
            images = read_images(images_txt)
            image_ids = load_image_ids(gt_json)
            pred_count = write_predictions(
                weights=Path(args.weights).resolve(),
                image_paths=images,
                image_ids=image_ids,
                repo=Path(args.repo).resolve(),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                output_json=pred_json,
            )
            selected_img_ids = [image_ids[Path(path).name] for path in images]
            metrics = evaluate_coco(gt_json, pred_json, args.max_det, selected_img_ids)
            write_metric_csv(metric_csv, metrics, pred_count, len(images))
            metric_row = read_metric_csv(metric_csv)

        rows.append(
            {
                "condition": condition,
                "type": condition_type(condition),
                "level": condition_level(condition),
                "images": metric_row["images"],
                "detections": metric_row["detections"],
                "AP": metric_row["AP"],
                "AP50": metric_row["AP50"],
                "AP75": metric_row["AP75"],
                "AP-S": metric_row["AP-S"],
                "AP-M": metric_row["AP-M"],
                "AP-L": metric_row["AP-L"],
            }
        )

    write_summary(output_dir, args.name_prefix, rows)


if __name__ == "__main__":
    main()
