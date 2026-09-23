#!/usr/bin/env python3
"""Merge latency benchmark rows with formal accuracy metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize deployment benchmark outputs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: Method|Training|benchmark_csv|formal_eval_csv",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected one benchmark row in {path}, got {len(rows)}")
    return rows[0]


def read_formal_metrics(path: Path) -> dict[str, str]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["condition"]] = row
    clean = rows["clean"]
    avg = rows["AVG_DEGRADED"]
    ratio = rows["ROBUSTNESS_RATIO"]
    return {
        "Clean mAP50": clean["map50"],
        "Avg-Degraded mAP50": avg["map50"],
        "Avg-Degraded mAP50-95": avg["map5095"],
        "Robustness Ratio": ratio["map50"],
    }


def fmt_params(value: str) -> str:
    try:
        return f"{int(value) / 1_000_000:.3f}"
    except Exception:
        return value


def write_md(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_rows: list[dict[str, str]] = []
    for spec in args.run:
        parts = spec.split("|")
        if len(parts) != 4:
            raise ValueError(f"Bad --run spec: {spec}")
        method, training, bench_csv, formal_csv = parts
        bench = read_one_row(Path(bench_csv))
        metrics = read_formal_metrics(Path(formal_csv))
        out_rows.append(
            {
                "Method": method,
                "Training": training,
                "Params (M)": fmt_params(bench["params"]),
                "Model size (MB)": bench["model_size_mb"],
                "Device": bench["gpu_info"].split(" | ")[0],
                "Precision": bench["precision"],
                "Input": f"{bench['imgsz']}x{bench['imgsz']}",
                "Batch": bench["batch"],
                "Latency mean (ms)": bench["latency_mean_ms"],
                "Latency median (ms)": bench["latency_median_ms"],
                "Latency P90 (ms)": bench["latency_p90_ms"],
                "Latency P95 (ms)": bench["latency_p95_ms"],
                "FPS": bench["fps"],
                **metrics,
            }
        )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    write_md(out_rows, Path(args.output_md))


if __name__ == "__main__":
    main()
