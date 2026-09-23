#!/usr/bin/env python3
"""Summarize clean/seen/unseen evaluation CSVs into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


UNSEEN_TYPES = [
    ("jpeg_compression", "JPEG"),
    ("gaussian_blur", "Gaussian Blur"),
    ("defocus_blur", "Defocus Blur"),
    ("salt_pepper_noise", "Salt-Pepper"),
    ("diagonal_motion_blur", "Diagonal Blur"),
    ("gamma_darkening", "Gamma"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize seen/unseen degradation evaluation results.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD|TRAINING|CSV",
        help="Run spec, e.g. 'SPhantomNet-Gray DART|DART|outputs/x.csv'.",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        metavar="MODEL|MODULE|TRAINING|PARAMS|GFLOPS|AP-S|LATENCY|CSV",
        help="Optional module table spec. Unknown values can be N/A.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_run(spec: str) -> Tuple[str, str, Path]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid --run spec: {spec}")
    return parts[0], parts[1], Path(parts[2]).resolve()


def parse_module(spec: str) -> Tuple[str, str, str, str, str, str, str, Path]:
    parts = spec.split("|")
    if len(parts) != 8:
        raise ValueError(f"Invalid --module spec: {spec}")
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], Path(parts[7]).resolve()


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key]) if row.get(key) not in ("", "N/A", None) else float("nan")


def fmt(value: float) -> str:
    return f"{value:.6f}"


def normal_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if row["type"] != "summary"]


def find_clean(rows: List[Dict[str, str]]) -> Dict[str, str]:
    for row in normal_rows(rows):
        if row["condition"] == "clean":
            return row
    raise KeyError("clean")


def avg(rows: Iterable[Dict[str, str]], key: str) -> float:
    values = [f(row, key) for row in rows]
    return mean(values) if values else float("nan")


def suite_rows(rows: List[Dict[str, str]], suite: str) -> List[Dict[str, str]]:
    return [row for row in normal_rows(rows) if row["suite"] == suite]


def write_csv(path: Path, rows: List[Dict[str, str]], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, str]], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def write_table(output_dir: Path, name: str, rows: List[Dict[str, str]], headers: List[str]) -> None:
    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"
    write_csv(csv_path, rows, headers)
    write_md(md_path, rows, headers)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    if not args.run:
        raise ValueError("Provide at least one --run 'METHOD|TRAINING|CSV'.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: List[Tuple[str, str, Path, List[Dict[str, str]]]] = []
    for spec in args.run:
        method, training, path = parse_run(spec)
        loaded.append((method, training, path, read_csv(path)))

    seen_rows: List[Dict[str, str]] = []
    unseen_rows: List[Dict[str, str]] = []
    combined_rows: List[Dict[str, str]] = []
    unseen_type_rows: List[Dict[str, str]] = []
    module_rows: List[Dict[str, str]] = []

    for method, training, _path, rows in loaded:
        clean = find_clean(rows)
        clean_map50 = f(clean, "map50")
        seen = suite_rows(rows, "seen")
        unseen = suite_rows(rows, "unseen")
        seen_avg = avg(seen, "map50")
        unseen_avg = avg(unseen, "map50")

        seen_rows.append(
            {
                "Method": method,
                "Clean mAP50": fmt(clean_map50),
                "Seen Avg mAP50": fmt(seen_avg),
                "Seen Robustness Ratio": fmt(seen_avg / clean_map50),
            }
        )
        unseen_rows.append(
            {
                "Method": method,
                "Clean mAP50": fmt(clean_map50),
                "Unseen Avg mAP50": fmt(unseen_avg),
                "Unseen Robustness Ratio": fmt(unseen_avg / clean_map50),
            }
        )
        combined_rows.append(
            {
                "Method": method,
                "Training": training,
                "Clean mAP50": fmt(clean_map50),
                "Seen Avg mAP50": fmt(seen_avg),
                "Unseen Avg mAP50": fmt(unseen_avg),
                "Seen Ratio": fmt(seen_avg / clean_map50),
                "Unseen Ratio": fmt(unseen_avg / clean_map50),
            }
        )

        by_type = {type_name: [row for row in unseen if row["type"] == type_name] for type_name, _label in UNSEEN_TYPES}
        unseen_type_rows.append(
            {
                "Method": method,
                **{label: fmt(avg(by_type[type_name], "map50")) for type_name, label in UNSEEN_TYPES},
                "Avg-Unseen": fmt(unseen_avg),
            }
        )

        module_rows.append(
            {
                "Model": method,
                "Module": "N/A",
                "Training": training,
                "Params": "N/A",
                "GFLOPs": "N/A",
                "Clean": fmt(clean_map50),
                "Seen Avg": fmt(seen_avg),
                "Unseen Avg": fmt(unseen_avg),
                "AP-S": "N/A",
                "Latency": "N/A",
            }
        )

    for spec in args.module:
        model, module, training, params, gflops, ap_s, latency, path = parse_module(spec)
        rows = read_csv(path)
        clean = find_clean(rows)
        seen_avg = avg(suite_rows(rows, "seen"), "map50")
        unseen_avg = avg(suite_rows(rows, "unseen"), "map50")
        module_rows.append(
            {
                "Model": model,
                "Module": module,
                "Training": training,
                "Params": params,
                "GFLOPs": gflops,
                "Clean": fmt(f(clean, "map50")),
                "Seen Avg": fmt(seen_avg),
                "Unseen Avg": fmt(unseen_avg),
                "AP-S": ap_s,
                "Latency": latency,
            }
        )

    write_table(output_dir, "table_seen_fairness", seen_rows, list(seen_rows[0].keys()))
    write_table(output_dir, "table_unseen_degradation", unseen_rows, list(unseen_rows[0].keys()))
    write_table(output_dir, "table_combined_seen_unseen", combined_rows, list(combined_rows[0].keys()))
    write_table(output_dir, "table_unseen_degradation_type", unseen_type_rows, list(unseen_type_rows[0].keys()))
    write_table(output_dir, "table_module_comparison", module_rows, list(module_rows[0].keys()))


if __name__ == "__main__":
    main()
