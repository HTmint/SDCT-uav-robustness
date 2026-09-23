#!/usr/bin/env python3
"""Summarize multiple DART eval CSV files into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


DEG_TYPES = ["low_light", "low_contrast", "noise", "motion_blur", "mixed"]
LEVELS = ["L1", "L2", "L3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DART evaluation CSV files.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD|TRAIN_SET|CSV",
        help="Run spec. May be repeated, e.g. 'SPhantomNet-Gray DART|clean+degraded|eval.csv'.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="METHOD|LL|LC|NOISE|BLUR|MIXED|CSV",
        help="Optional ablation spec for table 4. Values should be yes/no or check marks.",
    )
    return parser.parse_args()


def read_eval_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: Dict[str, str], key: str) -> float:
    return float(row[key]) if row.get(key) not in ("", "N/A", None) else float("nan")


def normal_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r["type"] != "summary"]


def find_condition(rows: List[Dict[str, str]], condition: str) -> Dict[str, str]:
    for row in rows:
        if row["condition"] == condition:
            return row
    raise KeyError(condition)


def avg(rows: List[Dict[str, str]], key: str) -> float:
    vals = [f(r, key) for r in rows]
    return mean(vals) if vals else float("nan")


def parse_run(spec: str) -> Tuple[str, str, Path]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid --run spec: {spec}")
    return parts[0], parts[1], Path(parts[2]).resolve()


def parse_ablation(spec: str) -> Tuple[str, str, str, str, str, str, Path]:
    parts = spec.split("|")
    if len(parts) != 7:
        raise ValueError(f"Invalid --ablation spec: {spec}")
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], Path(parts[6]).resolve()


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


def fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.run:
        raise ValueError("Provide at least one --run 'METHOD|TRAIN_SET|CSV'.")

    main_rows: List[Dict[str, str]] = []
    type_rows: List[Dict[str, str]] = []
    level_rows: List[Dict[str, str]] = []

    for spec in args.run:
        method, train_set, csv_path = parse_run(spec)
        rows = normal_rows(read_eval_csv(csv_path))
        clean = find_condition(rows, "clean")
        degraded = [r for r in rows if r["condition"] != "clean"]
        main_rows.append(
            {
                "Method": method,
                "Train Set": train_set,
                "Clean mAP50": fmt(f(clean, "map50")),
                "Clean mAP50-95": fmt(f(clean, "map5095")),
                "Avg-Degraded mAP50": fmt(avg(degraded, "map50")),
                "Avg-Degraded mAP50-95": fmt(avg(degraded, "map5095")),
                "Recall": fmt(avg(degraded, "recall")),
                "Robustness Ratio": fmt(avg(degraded, "map50") / f(clean, "map50")),
            }
        )

        by_type = {t: [r for r in degraded if r["type"] == t] for t in DEG_TYPES}
        type_rows.append(
            {
                "Method": method,
                "Low-light": fmt(avg(by_type["low_light"], "map50")),
                "Low-contrast": fmt(avg(by_type["low_contrast"], "map50")),
                "Noise": fmt(avg(by_type["noise"], "map50")),
                "Motion-blur": fmt(avg(by_type["motion_blur"], "map50")),
                "Mixed": fmt(avg(by_type["mixed"], "map50")),
                "Avg-Degraded": fmt(avg(degraded, "map50")),
            }
        )

        by_level = {level: [r for r in degraded if r["level"] == level] for level in LEVELS}
        level_rows.append(
            {
                "Method": method,
                "Clean": fmt(f(clean, "map50")),
                "L1 Avg": fmt(avg(by_level["L1"], "map50")),
                "L2 Avg": fmt(avg(by_level["L2"], "map50")),
                "L3 Avg": fmt(avg(by_level["L3"], "map50")),
            }
        )

    tables = [
        ("table1_main_results", main_rows, list(main_rows[0].keys())),
        ("table2_degradation_type", type_rows, list(type_rows[0].keys())),
        ("table3_degradation_level", level_rows, list(level_rows[0].keys())),
    ]

    if args.ablation:
        ab_rows: List[Dict[str, str]] = []
        for spec in args.ablation:
            method, ll, lc, noise, blur, mixed, csv_path = parse_ablation(spec)
            rows = [r for r in normal_rows(read_eval_csv(csv_path)) if r["condition"] != "clean"]
            ab_rows.append(
                {
                    "Method": method,
                    "Low-light train": ll,
                    "Low-contrast train": lc,
                    "Noise train": noise,
                    "Blur train": blur,
                    "Mixed train": mixed,
                    "Avg-Degraded mAP50": fmt(avg(rows, "map50")),
                }
            )
        tables.append(("table4_ablation", ab_rows, list(ab_rows[0].keys())))

    for name, rows, headers in tables:
        csv_path = out / f"{name}.csv"
        md_path = out / f"{name}.md"
        write_csv(csv_path, rows, headers)
        write_md(md_path, rows, headers)
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
