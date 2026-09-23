#!/usr/bin/env python3
"""Summarize repeated-seed strong-unseen stress-test results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    training: str
    prefix: str


METHODS = [
    MethodSpec("SPhantomNet-Gray Clean", "Clean", "sphantomnet_gray_clean"),
    MethodSpec("SPhantomNet-Gray Clean Repeat", "Clean Repeat", "sphantomnet_gray_clean_repeat"),
    MethodSpec("SPhantomNet-Gray DART", "DART", "sphantomnet_gray_dart"),
    MethodSpec("SPhantomNet-Gray SDCT", "SDCT", "sphantomnet_gray_random_aug"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strong-unseen mean/std tables.")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "outputs/sdct_strong_unseen_eval")
    parser.add_argument("--output-table-dir", type=Path, default=ROOT / "outputs/paper_tables")
    parser.add_argument("--output-analysis-dir", type=Path, default=ROOT / "outputs/paper_analysis")
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def split_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row[h] for h in headers) + " |\n")


def find(rows: list[dict[str, str]], condition: str) -> dict[str, str]:
    for row in rows:
        if row["condition"] == condition:
            return row
    raise KeyError(condition)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_mean_std(values: list[float]) -> str:
    mean, std = mean_std(values)
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.6f} +/- {std:.6f}"


def fmt(value: float) -> str:
    return f"{value:.6f}"


def normal_strong(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["suite"] == "strong_unseen" and row["type"] != "summary"]


def main() -> None:
    args = parse_args()
    seeds = split_ints(args.seeds)
    args.output_table_dir.mkdir(parents=True, exist_ok=True)
    args.output_analysis_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[int, list[dict[str, str]]]] = {}
    for spec in METHODS:
        loaded[spec.prefix] = {}
        for seed in seeds:
            path = args.eval_dir / f"{spec.prefix}_seed{seed}.csv"
            if path.exists():
                loaded[spec.prefix][seed] = read_csv(path)

    headers = [
        "Method",
        "Training",
        "Seeds",
        "Clean mAP50 mean +/- std",
        "Strong-Unseen Avg mAP50 mean +/- std",
        "Strong-Unseen Avg mAP50-95 mean +/- std",
        "Strong-Unseen Robustness Ratio mean +/- std",
    ]
    summary_rows: list[dict[str, str]] = []
    numeric_summary: list[dict[str, float | str]] = []
    for spec in METHODS:
        seed_rows = loaded[spec.prefix]
        used_seeds = sorted(seed_rows)
        clean_vals: list[float] = []
        avg50_vals: list[float] = []
        avg5095_vals: list[float] = []
        ratio_vals: list[float] = []
        for seed in used_seeds:
            rows = seed_rows[seed]
            clean_vals.append(float(find(rows, "clean")["map50"]))
            avg50_vals.append(float(find(rows, "STRONG_UNSEEN_AVG")["map50"]))
            avg5095_vals.append(float(find(rows, "STRONG_UNSEEN_AVG")["map5095"]))
            ratio_vals.append(float(find(rows, "STRONG_UNSEEN_ROBUSTNESS_RATIO")["map50"]))
        summary_rows.append(
            {
                "Method": spec.method,
                "Training": spec.training,
                "Seeds": ",".join(str(seed) for seed in used_seeds) if used_seeds else "N/A",
                "Clean mAP50 mean +/- std": fmt_mean_std(clean_vals),
                "Strong-Unseen Avg mAP50 mean +/- std": fmt_mean_std(avg50_vals),
                "Strong-Unseen Avg mAP50-95 mean +/- std": fmt_mean_std(avg5095_vals),
                "Strong-Unseen Robustness Ratio mean +/- std": fmt_mean_std(ratio_vals),
            }
        )
        avg50_mean, _ = mean_std(avg50_vals)
        ratio_mean, _ = mean_std(ratio_vals)
        numeric_summary.append({"method": spec.method, "avg50": avg50_mean, "ratio": ratio_mean, "seeds": ",".join(map(str, used_seeds))})

    write_csv(args.output_table_dir / "table_strong_unseen_sdct_mean_std.csv", summary_rows, headers)
    write_md(args.output_table_dir / "table_strong_unseen_sdct_mean_std.md", summary_rows, headers)

    first_available = next(rows for method_rows in loaded.values() for rows in method_rows.values())
    conditions = [row["condition"] for row in normal_strong(first_available)]
    per_headers = ["Condition", "Type", *[spec.method for spec in METHODS]]
    per_rows: list[dict[str, str]] = []
    for condition in conditions:
        first = find(first_available, condition)
        out = {"Condition": condition, "Type": first["type"]}
        for spec in METHODS:
            values = [float(find(rows, condition)["map50"]) for rows in loaded[spec.prefix].values()]
            out[spec.method] = fmt_mean_std(values)
        per_rows.append(out)

    write_csv(args.output_table_dir / "table_strong_unseen_per_condition_mean_std.csv", per_rows, per_headers)
    write_md(args.output_table_dir / "table_strong_unseen_per_condition_mean_std.md", per_rows, per_headers)

    best_avg = max(numeric_summary, key=lambda row: float(row["avg50"]))
    best_ratio = max(numeric_summary, key=lambda row: float(row["ratio"]))
    dart = next(row for row in numeric_summary if row["method"] == "SPhantomNet-Gray DART")
    sdct = next(row for row in numeric_summary if row["method"] == "SPhantomNet-Gray SDCT")
    analysis = args.output_analysis_dir / "strong_unseen_mean_std_interpretation.md"
    analysis.write_text(
        "\n".join(
            [
                "# Strong-Unseen Mean/Std Interpretation",
                "",
                "This is a synthetic strong-unseen stress test generated from DUT-Gray/test. It is useful for stress testing SDCT coverage, but it is not real external open-set evidence.",
                "",
                f"- Best mean Strong-Unseen Avg mAP50: {best_avg['method']} ({fmt(float(best_avg['avg50']))}).",
                f"- Best mean Strong-Unseen robustness ratio: {best_ratio['method']} ({fmt(float(best_ratio['ratio']))}).",
                f"- SDCT vs DART Avg mAP50 gap: {fmt(float(sdct['avg50']) - float(dart['avg50']))}.",
                f"- SDCT vs DART robustness-ratio gap: {fmt(float(sdct['ratio']) - float(dart['ratio']))}.",
                "- Report this table as an additional stress-test result, preferably after the main three-seed formal seen/in-family table.",
                "- Keep the wording conservative: SDCT is the strongest strategy in this synthetic stress test; DART remains the deterministic protocol/baseline.",
                "",
                "Generated files:",
                "- `outputs/paper_tables/table_strong_unseen_sdct_mean_std.md`",
                "- `outputs/paper_tables/table_strong_unseen_per_condition_mean_std.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(args.output_table_dir / "table_strong_unseen_sdct_mean_std.md")
    print(args.output_table_dir / "table_strong_unseen_per_condition_mean_std.md")
    print(analysis)


if __name__ == "__main__":
    main()
