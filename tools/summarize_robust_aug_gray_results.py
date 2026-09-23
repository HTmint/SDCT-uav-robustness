#!/usr/bin/env python3
"""Summarize robust augmentation baseline results with existing SDCT baselines."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUST_OUT = ROOT / "outputs/robust_aug_baselines"
FORMAL_EXISTING = ROOT / "outputs/dart_formal_eval"
ORDINARY_EXISTING = ROOT / "outputs/dart_seen_unseen_eval"
STRONG_EXISTING = ROOT / "outputs/sdct_strong_unseen_eval"
FORMAL_NEW = ROBUST_OUT / "formal_seen"
ORDINARY_NEW = ROBUST_OUT / "ordinary_heldout"
STRONG_NEW = ROBUST_OUT / "strong_heldout"
SUMMARY_DIR = ROBUST_OUT / "summary"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    table_name: str
    prefix: str
    formal_dir: Path
    ordinary_dir: Path
    strong_dir: Path


METHODS = [
    MethodSpec(
        method="SPhantomNet-Gray Clean Repeat",
        table_name="Clean Repeat",
        prefix="sphantomnet_gray_clean_repeat",
        formal_dir=FORMAL_EXISTING,
        ordinary_dir=ORDINARY_EXISTING,
        strong_dir=STRONG_EXISTING,
    ),
    MethodSpec(
        method="SPhantomNet-Gray DART",
        table_name="DART",
        prefix="sphantomnet_gray_dart",
        formal_dir=FORMAL_EXISTING,
        ordinary_dir=ORDINARY_EXISTING,
        strong_dir=STRONG_EXISTING,
    ),
    MethodSpec(
        method="SPhantomNet-Gray RandAugment-Gray",
        table_name="RandAugment-Gray",
        prefix="sphantomnet_gray_randaugment_gray",
        formal_dir=FORMAL_NEW,
        ordinary_dir=ORDINARY_NEW,
        strong_dir=STRONG_NEW,
    ),
    MethodSpec(
        method="SPhantomNet-Gray AugMix-Input-Gray",
        table_name="AugMix-Input-Gray",
        prefix="sphantomnet_gray_augmix_input_gray",
        formal_dir=FORMAL_NEW,
        ordinary_dir=ORDINARY_NEW,
        strong_dir=STRONG_NEW,
    ),
    MethodSpec(
        method="SPhantomNet-Gray SDCT",
        table_name="SDCT",
        prefix="sphantomnet_gray_random_aug",
        formal_dir=FORMAL_EXISTING,
        ordinary_dir=ORDINARY_EXISTING,
        strong_dir=STRONG_EXISTING,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-seed and mean/std tables for generic baselines.")
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def seeds_from_text(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_metric(rows: list[dict[str, str]], condition: str, suite: str | None = None) -> float:
    for row in rows:
        if row.get("condition") == condition and (suite is None or row.get("suite") == suite):
            return float(row["map50"])
    raise KeyError(condition)


def read_seed_metrics(spec: MethodSpec, seed: int) -> dict[str, float | str]:
    formal_path = spec.formal_dir / f"{spec.prefix}_seed{seed}.csv"
    ordinary_path = spec.ordinary_dir / f"{spec.prefix}_seed{seed}.csv"
    strong_path = spec.strong_dir / f"{spec.prefix}_seed{seed}.csv"
    row: dict[str, float | str] = {
        "Method": spec.table_name,
        "Full Method": spec.method,
        "Seed": seed,
        "formal_csv": str(formal_path) if formal_path.exists() else "",
        "ordinary_csv": str(ordinary_path) if ordinary_path.exists() else "",
        "strong_csv": str(strong_path) if strong_path.exists() else "",
    }
    if formal_path.exists():
        formal = read_csv(formal_path)
        row["Clean mAP50"] = find_metric(formal, "clean")
        row["Seen Avg mAP50"] = find_metric(formal, "AVG_DEGRADED")
        row["Seen Ratio"] = find_metric(formal, "ROBUSTNESS_RATIO")
    else:
        row["Clean mAP50"] = math.nan
        row["Seen Avg mAP50"] = math.nan
        row["Seen Ratio"] = math.nan
    if ordinary_path.exists():
        ordinary = read_csv(ordinary_path)
        row["Ordinary Held-Out Avg mAP50"] = find_metric(ordinary, "UNSEEN_AVG", "unseen")
        row["Ordinary Held-Out Ratio"] = find_metric(ordinary, "UNSEEN_ROBUSTNESS_RATIO", "unseen")
    else:
        row["Ordinary Held-Out Avg mAP50"] = math.nan
        row["Ordinary Held-Out Ratio"] = math.nan
    if strong_path.exists():
        strong = read_csv(strong_path)
        row["Strong Held-Out Avg mAP50"] = find_metric(strong, "STRONG_UNSEEN_AVG", "strong_unseen")
        row["Strong Held-Out Ratio"] = find_metric(strong, "STRONG_UNSEEN_ROBUSTNESS_RATIO", "strong_unseen")
    else:
        row["Strong Held-Out Avg mAP50"] = math.nan
        row["Strong Held-Out Ratio"] = math.nan
    return row


def finite(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def mean_std(values: list[float]) -> tuple[float, float]:
    xs = finite(values)
    if not xs:
        return math.nan, math.nan
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt_value(value: float) -> str:
    return "TODO_NOT_FOUND" if math.isnan(value) else f"{value:.6f}"


def fmt_mean_std(values: list[float]) -> str:
    m, s = mean_std(values)
    if math.isnan(m):
        return "TODO_NOT_FOUND"
    return f"{m:.6f} +/- {s:.6f}"


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_latex(path: Path, rows: list[dict[str, str]]) -> None:
    def latex_metric(value: str) -> str:
        return value.replace(" +/- ", " $\\pm$ ")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison with generic robustness augmentation baselines. Metrics are mAP50 mean $\\pm$ std over available seeds; ratios are computed per seed before aggregation.}",
        "\\label{tab:generic_robust_aug}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Method & Clean & Seen Avg & Held-Out Avg & Strong Avg & Strong Ratio \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['Method']} & {latex_metric(row['Clean mAP50'])} & {latex_metric(row['Seen Avg'])} & "
            f"{latex_metric(row['Held-Out Avg'])} & {latex_metric(row['Strong Avg'])} & "
            f"{latex_metric(row['Strong Ratio'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds_from_text(args.seeds)
    per_seed_raw: list[dict[str, float | str]] = []
    for spec in METHODS:
        for seed in seeds:
            per_seed_raw.append(read_seed_metrics(spec, seed))

    per_seed_headers = [
        "Method",
        "Full Method",
        "Seed",
        "Clean mAP50",
        "Seen Avg mAP50",
        "Ordinary Held-Out Avg mAP50",
        "Strong Held-Out Avg mAP50",
        "Seen Ratio",
        "Ordinary Held-Out Ratio",
        "Strong Held-Out Ratio",
        "formal_csv",
        "ordinary_csv",
        "strong_csv",
    ]
    per_seed_rows: list[dict[str, str]] = []
    metric_headers = per_seed_headers[3:10]
    for raw in per_seed_raw:
        row: dict[str, str] = {}
        for header in per_seed_headers:
            value = raw[header]
            row[header] = fmt_value(value) if isinstance(value, float) else str(value)
        per_seed_rows.append(row)
    write_csv(out_dir / "robust_aug_baselines_per_seed.csv", per_seed_rows, per_seed_headers)

    summary_headers = [
        "Method",
        "Seeds formal",
        "Seeds ordinary held-out",
        "Seeds strong held-out",
        "Clean mAP50",
        "Seen Avg",
        "Held-Out Avg",
        "Strong Avg",
        "Seen Ratio",
        "Held-Out Ratio",
        "Strong Ratio",
    ]
    summary_rows: list[dict[str, str]] = []
    summary_numeric: dict[str, dict[str, float]] = {}
    for spec in METHODS:
        rows = [r for r in per_seed_raw if r["Method"] == spec.table_name]
        formal_seeds = [str(r["Seed"]) for r in rows if not math.isnan(float(r["Seen Avg mAP50"]))]
        ordinary_seeds = [str(r["Seed"]) for r in rows if not math.isnan(float(r["Ordinary Held-Out Avg mAP50"]))]
        strong_seeds = [str(r["Seed"]) for r in rows if not math.isnan(float(r["Strong Held-Out Avg mAP50"]))]
        metric_values = {header: [float(r[header]) for r in rows] for header in metric_headers}
        summary_numeric[spec.table_name] = {
            "Clean mAP50": mean_std(metric_values["Clean mAP50"])[0],
            "Seen Avg": mean_std(metric_values["Seen Avg mAP50"])[0],
            "Held-Out Avg": mean_std(metric_values["Ordinary Held-Out Avg mAP50"])[0],
            "Strong Avg": mean_std(metric_values["Strong Held-Out Avg mAP50"])[0],
            "Strong Ratio": mean_std(metric_values["Strong Held-Out Ratio"])[0],
        }
        summary_rows.append(
            {
                "Method": spec.table_name,
                "Seeds formal": ",".join(formal_seeds) if formal_seeds else "TODO_NOT_FOUND",
                "Seeds ordinary held-out": ",".join(ordinary_seeds) if ordinary_seeds else "TODO_NOT_FOUND",
                "Seeds strong held-out": ",".join(strong_seeds) if strong_seeds else "TODO_NOT_FOUND",
                "Clean mAP50": fmt_mean_std(metric_values["Clean mAP50"]),
                "Seen Avg": fmt_mean_std(metric_values["Seen Avg mAP50"]),
                "Held-Out Avg": fmt_mean_std(metric_values["Ordinary Held-Out Avg mAP50"]),
                "Strong Avg": fmt_mean_std(metric_values["Strong Held-Out Avg mAP50"]),
                "Seen Ratio": fmt_mean_std(metric_values["Seen Ratio"]),
                "Held-Out Ratio": fmt_mean_std(metric_values["Ordinary Held-Out Ratio"]),
                "Strong Ratio": fmt_mean_std(metric_values["Strong Held-Out Ratio"]),
            }
        )
    write_csv(out_dir / "robust_aug_baselines_mean_std.csv", summary_rows, summary_headers)
    write_md(out_dir / "robust_aug_baselines_mean_std.md", summary_rows, summary_headers)
    write_latex(out_dir / "robust_aug_baselines_main_table.tex", summary_rows)

    delta_headers = ["Comparison", "Clean mAP50 delta", "Seen Avg delta", "Held-Out Avg delta", "Strong Avg delta", "Strong Ratio delta"]
    delta_rows: list[dict[str, str]] = []
    sdct = summary_numeric.get("SDCT", {})
    for baseline in ("RandAugment-Gray", "AugMix-Input-Gray"):
        base = summary_numeric.get(baseline, {})
        delta_rows.append(
            {
                "Comparison": f"SDCT - {baseline}",
                "Clean mAP50 delta": fmt_value(sdct.get("Clean mAP50", math.nan) - base.get("Clean mAP50", math.nan)),
                "Seen Avg delta": fmt_value(sdct.get("Seen Avg", math.nan) - base.get("Seen Avg", math.nan)),
                "Held-Out Avg delta": fmt_value(sdct.get("Held-Out Avg", math.nan) - base.get("Held-Out Avg", math.nan)),
                "Strong Avg delta": fmt_value(sdct.get("Strong Avg", math.nan) - base.get("Strong Avg", math.nan)),
                "Strong Ratio delta": fmt_value(sdct.get("Strong Ratio", math.nan) - base.get("Strong Ratio", math.nan)),
            }
        )
    write_csv(out_dir / "sdct_vs_generic_aug_deltas.csv", delta_rows, delta_headers)
    write_md(out_dir / "sdct_vs_generic_aug_deltas.md", delta_rows, delta_headers)

    print(f"Per-seed CSV: {out_dir / 'robust_aug_baselines_per_seed.csv'}")
    print(f"Mean/std CSV: {out_dir / 'robust_aug_baselines_mean_std.csv'}")
    print(f"LaTeX table:  {out_dir / 'robust_aug_baselines_main_table.tex'}")
    print(f"Deltas:       {out_dir / 'sdct_vs_generic_aug_deltas.csv'}")


if __name__ == "__main__":
    main()
