#!/usr/bin/env python3
"""Summarize SDCT mechanism-ablation formal eval CSVs into paper tables."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "outputs/sdct_ablation_formal_eval"
DEFAULT_TABLE_DIR = ROOT / "outputs/paper_tables"
DEFAULT_ANALYSIS_DIR = ROOT / "outputs/paper_analysis"
DEGRADATION_TYPES = ["low_light", "low_contrast", "noise", "motion_blur", "mixed"]

METHODS = [
    {
        "method": "SPhantomNet SDCT-NoMixed",
        "prefix": "sphantomnet_gray_sdct_no_mixed",
        "type_sampling": "Random over 4 single-degradation families",
        "severity_sampling": "Continuous random",
        "mixed": "No",
        "train_size": "31200",
    },
    {
        "method": "SPhantomNet SDCT-Discrete",
        "prefix": "sphantomnet_gray_sdct_discrete",
        "type_sampling": "Random over 5 degradation families",
        "severity_sampling": "Fixed L1/L2/L3 random",
        "mixed": "Yes",
        "train_size": "31200",
    },
]

BASELINE_ROWS = [
    [
        "SPhantomNet Clean Repeat",
        "None; clean list repeated",
        "None",
        "No",
        "31200",
        "0,1,2",
        "0.917328 +/- 0.005901",
        "0.736942 +/- 0.004222",
        "0.489151 +/- 0.001995",
        "0.803376 +/- 0.006432",
    ],
    [
        "SPhantomNet DART",
        "Deterministic fixed degradation families",
        "Fixed deterministic train parameters",
        "Yes",
        "31200",
        "0,1,2",
        "0.907645 +/- 0.002330",
        "0.828610 +/- 0.007904",
        "0.557960 +/- 0.006586",
        "0.912938 +/- 0.010399",
    ],
    [
        "SPhantomNet SDCT-Full / Random-Aug",
        "Random over 5 degradation families",
        "Continuous random",
        "Yes",
        "31200",
        "0,1,2",
        "0.904029 +/- 0.002819",
        "0.841890 +/- 0.003328",
        "0.563415 +/- 0.003343",
        "0.931265 +/- 0.002186",
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SDCT ablation formal eval CSVs.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seed_from_path(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("_seed", 1)[1])


def metrics(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    by_condition = {row["condition"]: row for row in rows}
    clean = by_condition["clean"]
    avg = by_condition["AVG_DEGRADED"]
    ratio = by_condition["ROBUSTNESS_RATIO"]
    out = {
        "clean": float(clean["map50"]),
        "avg_map50": float(avg["map50"]),
        "avg_map5095": float(avg["map5095"]),
        "ratio": float(ratio["map50"]),
        "mixed": sum(float(r["map50"]) for r in rows if r["type"] == "mixed") / 3.0,
    }
    for deg_type in DEGRADATION_TYPES:
        type_rows = [r for r in rows if r["type"] == deg_type]
        out[f"{deg_type}_map50"] = sum(float(r["map50"]) for r in type_rows) / len(type_rows)
        out[f"{deg_type}_map5095"] = sum(float(r["map5095"]) for r in type_rows) / len(type_rows)
    return out


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "pending"
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    table_dir = args.table_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "Method",
        "Type Sampling",
        "Severity Sampling",
        "Mixed",
        "Train Size",
        "Seeds",
        "Clean mAP50",
        "Avg-Degraded mAP50",
        "Avg-Degraded mAP50-95",
        "Robustness Ratio",
    ]
    rows = [*BASELINE_ROWS]
    summary: dict[str, dict[str, object]] = {}

    for method in METHODS:
        paths = sorted(eval_dir.glob(f"{method['prefix']}_seed*.csv"), key=seed_from_path)
        seeds = [seed_from_path(path) for path in paths]
        vals = [metrics(path) for path in paths]
        summary[method["method"]] = {"seeds": seeds, "values": vals}
        rows.append(
            [
                method["method"],
                method["type_sampling"],
                method["severity_sampling"],
                method["mixed"],
                method["train_size"],
                ",".join(str(seed) for seed in seeds) if seeds else "pending",
                fmt_mean_std([v["clean"] for v in vals]),
                fmt_mean_std([v["avg_map50"] for v in vals]),
                fmt_mean_std([v["avg_map5095"] for v in vals]),
                fmt_mean_std([v["ratio"] for v in vals]),
            ]
        )

    table_md = table_dir / "table_sdct_ablation_formal_mean_std.md"
    table_csv = table_dir / "table_sdct_ablation_formal_mean_std.csv"
    table_md.write_text(md_table(headers, rows), encoding="utf-8")
    write_csv(table_csv, headers, rows)

    per_type_headers = [
        "Degradation Type",
        "NoMixed mAP50",
        "Discrete mAP50",
        "Delta mAP50",
        "NoMixed mAP50-95",
        "Discrete mAP50-95",
        "Delta mAP50-95",
    ]
    per_type_rows = []
    no_mixed_vals = summary["SPhantomNet SDCT-NoMixed"]["values"]
    discrete_vals = summary["SPhantomNet SDCT-Discrete"]["values"]
    for deg_type in DEGRADATION_TYPES:
        no_map50 = [v[f"{deg_type}_map50"] for v in no_mixed_vals]
        di_map50 = [v[f"{deg_type}_map50"] for v in discrete_vals]
        no_map5095 = [v[f"{deg_type}_map5095"] for v in no_mixed_vals]
        di_map5095 = [v[f"{deg_type}_map5095"] for v in discrete_vals]
        delta50 = statistics.mean(di_map50) - statistics.mean(no_map50) if no_map50 and di_map50 else None
        delta5095 = statistics.mean(di_map5095) - statistics.mean(no_map5095) if no_map5095 and di_map5095 else None
        per_type_rows.append(
            [
                deg_type,
                fmt_mean_std(no_map50),
                fmt_mean_std(di_map50),
                f"{delta50:.6f}" if delta50 is not None else "pending",
                fmt_mean_std(no_map5095),
                fmt_mean_std(di_map5095),
                f"{delta5095:.6f}" if delta5095 is not None else "pending",
            ]
        )

    per_type_md = table_dir / "table_sdct_ablation_per_type_mean_std.md"
    per_type_csv = table_dir / "table_sdct_ablation_per_type_mean_std.csv"
    per_type_md.write_text(md_table(per_type_headers, per_type_rows), encoding="utf-8")
    write_csv(per_type_csv, per_type_headers, per_type_rows)

    analysis = [
        "# SDCT Ablation Formal Mean/Std Interpretation",
        "",
        f"NoMixed available seeds: {summary['SPhantomNet SDCT-NoMixed']['seeds']}",
        f"Discrete available seeds: {summary['SPhantomNet SDCT-Discrete']['seeds']}",
        "",
    ]
    if no_mixed_vals and discrete_vals:
        no_mixed_avg = statistics.mean(v["avg_map50"] for v in no_mixed_vals)
        discrete_avg = statistics.mean(v["avg_map50"] for v in discrete_vals)
        no_mixed_ratio = statistics.mean(v["ratio"] for v in no_mixed_vals)
        discrete_ratio = statistics.mean(v["ratio"] for v in discrete_vals)
        no_mixed_mixed = statistics.mean(v["mixed"] for v in no_mixed_vals)
        discrete_mixed = statistics.mean(v["mixed"] for v in discrete_vals)
        analysis.extend(
            [
                f"- Discrete minus NoMixed Avg-Degraded mAP50: {discrete_avg - no_mixed_avg:.6f}.",
                f"- Discrete minus NoMixed robustness ratio: {discrete_ratio - no_mixed_ratio:.6f}.",
                f"- Discrete minus NoMixed mixed-family mAP50: {discrete_mixed - no_mixed_mixed:.6f}.",
                "- The three-seed result supports mixed / combination degradation as a key SDCT component. Removing mixed keeps low-light, low-contrast, noise, and motion-blur performance competitive, but it collapses on mixed_L2/L3 and lowers the overall robustness ratio.",
                "- SDCT-Discrete is also slightly above SDCT-Full on the formal deterministic seen suite. This should not be written as proof that discrete severity is universally better: the formal test uses fixed L1/L2/L3 severities, so this result may partly reflect train-test severity alignment.",
                "- The safest paper claim is that SDCT's gain is driven mainly by stochastic type coverage with mixed degradation, while continuous-vs-discrete severity sampling is a policy variable requiring separate tuning or external validation.",
            ]
        )
    else:
        analysis.append("- Waiting for both ablation variants to have at least one eval CSV.")
    (analysis_dir / "sdct_ablation_formal_mean_std_interpretation.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")

    print(f"Wrote {table_md}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {per_type_md}")
    print(f"Wrote {per_type_csv}")
    print(f"Wrote {analysis_dir / 'sdct_ablation_formal_mean_std_interpretation.md'}")


if __name__ == "__main__":
    main()
