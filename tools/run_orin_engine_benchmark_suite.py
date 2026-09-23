#!/usr/bin/env python3
"""Run AGX Orin TensorRT engine benchmarks from an existing engine directory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = "./outputs/agx_orin_onnx_export_20260804/export_manifest.csv"
DEFAULT_ENGINE_DIR = "./outputs/agx_orin_onnx_export_20260804/engine"
DEFAULT_OUTPUT_DIR = "./outputs/agx_orin_engine_benchmark_20260804"
DEFAULT_BENCH_SCRIPT = "./tools/benchmark_tensorrt_engine.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a suite of existing TensorRT .engine files on AGX Orin."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--engine-suffix", default=".engine")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-script", default=DEFAULT_BENCH_SCRIPT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--input-shape", default="1,1,640,640")
    parser.add_argument("--precision", default="FP16")
    parser.add_argument("--measure", choices=("end_to_end", "execute_only"), default="end_to_end")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--tegrastats-log", default=None)
    parser.add_argument("--metric-table", default="./outputs/paper_tables/table_main_three_seed.md")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--allow-non-orin",
        action="store_true",
        help="Allow real benchmark execution on non-AGX-Orin hardware for debugging only.",
    )
    parser.add_argument(
        "--allow-missing-power",
        action="store_true",
        help="Allow real benchmark outputs without tegrastats-derived power for debugging only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def command_output(cmd: list[str], timeout: int = 5) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception:
        return "N/A"


def device_tree_model() -> str:
    return command_output(["bash", "-lc", "tr -d '\\0' </proc/device-tree/model 2>/dev/null"])


def is_agx_orin_model(model: str) -> bool:
    text = model.lower()
    return "agx" in text and "orin" in text


def normalize_method(value: str) -> str:
    text = value.lower().replace(" / random-aug", "").replace("random-aug", "sdct")
    return "".join(ch for ch in text if ch.isalnum())


def parse_metric_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        return {}
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    out: dict[str, dict[str, str]] = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        method = row.get("Method", "")
        if method:
            out[normalize_method(method)] = row
    return out


def metric_for(method: str, manifest_row: dict[str, str], metrics: dict[str, dict[str, str]]) -> dict[str, str]:
    key = normalize_method(method)
    row = metrics.get(key)
    if row is None:
        for candidate_key, candidate in metrics.items():
            if key and (key in candidate_key or candidate_key in key):
                row = candidate
                break
    return {
        "Clean mAP50": row.get("Clean mAP50", "TODO_NOT_FOUND") if row else "TODO_NOT_FOUND",
        "Avg-Degraded mAP50": row.get("Avg-Degraded mAP50", manifest_row.get("avg_degraded_map50", "TODO_NOT_FOUND"))
        if row
        else manifest_row.get("avg_degraded_map50", "TODO_NOT_FOUND"),
        "Avg-Degraded mAP50-95": row.get("Avg-Degraded mAP50-95", "TODO_NOT_FOUND")
        if row
        else "TODO_NOT_FOUND",
        "Robustness Ratio": row.get("Robustness Ratio", "TODO_NOT_FOUND") if row else "TODO_NOT_FOUND",
    }


def method_stem(row: dict[str, str]) -> str:
    onnx_path = row.get("onnx_path", "")
    if onnx_path:
        return Path(onnx_path).stem
    copied_pt = row.get("copied_pt", "")
    if copied_pt:
        return Path(copied_pt).stem.replace("_best", "_best")
    return row["method"].lower().replace(" ", "_")


def safe_name(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(rows: list[dict[str, str]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0])
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    engine_dir = Path(args.engine_dir)
    output_dir = Path(args.output_dir)
    per_run_dir = output_dir / "per_model"
    latency_dir = output_dir / "latencies"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    latency_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    metrics = parse_metric_table(Path(args.metric_table))
    summary_rows: list[dict[str, str]] = []
    existing_engines = [
        engine_dir / f"{method_stem(row)}{args.engine_suffix}" for row in rows
    ]
    will_run_real_benchmark = any(path.exists() for path in existing_engines) and not args.dry_run
    if will_run_real_benchmark and not args.allow_non_orin:
        model = device_tree_model()
        if not is_agx_orin_model(model):
            raise RuntimeError(
                "Refusing to run real deployment benchmark because this is not detected as AGX Orin. "
                f"Detected device model: {model!r}. Use --allow-non-orin only for debugging; "
                "do not report non-Orin results as AGX Orin."
            )
    if will_run_real_benchmark and not args.allow_missing_power and not args.tegrastats_log:
        raise RuntimeError(
            "Power measurement is required for paper-facing AGX Orin deployment results. "
            "Start tegrastats logging and pass --tegrastats-log, or use "
            "--allow-missing-power only for debugging."
        )

    for row in rows:
        stem = method_stem(row)
        engine_path = engine_dir / f"{stem}{args.engine_suffix}"
        run_name = safe_name(stem)
        json_path = per_run_dir / f"{run_name}.json"
        csv_path = per_run_dir / f"{run_name}.csv"
        md_path = per_run_dir / f"{run_name}.md"
        lat_path = latency_dir / f"{run_name}_latencies.csv"

        if not engine_path.exists():
            if not args.allow_missing:
                raise FileNotFoundError(f"Missing engine: {engine_path}")
            metric = metric_for(row["method"], row, metrics)
            summary_rows.append(
                {
                    "Method": row["method"],
                    "Role": row.get("role", ""),
                    "Status": "MISSING_ENGINE",
                    "Engine": str(engine_path),
                    "Precision": args.precision,
                    "Device": "TODO_NOT_FOUND",
                    "Measurement": args.measure,
                    "Input shape": args.input_shape,
                    "Latency mean (ms)": "TODO_NOT_FOUND",
                    "Latency median (ms)": "TODO_NOT_FOUND",
                    "Latency P90 (ms)": "TODO_NOT_FOUND",
                    "Latency P95 (ms)": "TODO_NOT_FOUND",
                    "FPS": "TODO_NOT_FOUND",
                    "Power avg (W)": "TODO_NOT_FOUND",
                    **metric,
                }
            )
            continue

        cmd = [
            args.python,
            str(Path(args.benchmark_script)),
            "--engine",
            str(engine_path),
            "--name",
            row["method"],
            "--input-shape",
            args.input_shape,
            "--precision",
            args.precision,
            "--measure",
            args.measure,
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--output-md",
            str(md_path),
            "--latencies-csv",
            str(lat_path),
        ]
        if args.source:
            cmd.extend(["--source", args.source])
        if args.recursive:
            cmd.append("--recursive")
        if args.tegrastats_log:
            cmd.extend(["--tegrastats-log", args.tegrastats_log])
        if args.allow_non_orin:
            cmd.append("--allow-non-orin")
        if args.allow_missing_power:
            cmd.append("--allow-missing-power")
        run_command(cmd, args.dry_run)

        if args.dry_run:
            continue

        bench = read_json(json_path)
        metric = metric_for(row["method"], row, metrics)
        summary_rows.append(
            {
                "Method": row["method"],
                "Role": row.get("role", ""),
                "Status": "OK",
                "Engine": str(engine_path),
                "Precision": str(bench.get("precision", args.precision)),
                "Device": str(bench.get("device", "N/A")),
                "Measurement": str(bench.get("measurement_mode", args.measure)),
                "Input shape": json.dumps(bench.get("input_shape", args.input_shape)),
                "Latency mean (ms)": str(bench.get("latency_mean_ms", "N/A")),
                "Latency median (ms)": str(bench.get("latency_median_ms", "N/A")),
                "Latency P90 (ms)": str(bench.get("latency_p90_ms", "N/A")),
                "Latency P95 (ms)": str(bench.get("latency_p95_ms", "N/A")),
                "FPS": str(bench.get("fps", "N/A")),
                "Power avg (W)": str(bench.get("power_avg_w", "N/A")),
                **metric,
            }
        )

    if not args.dry_run:
        write_table(
            summary_rows,
            output_dir / "orin_engine_benchmark_summary.csv",
            output_dir / "orin_engine_benchmark_summary.md",
        )


if __name__ == "__main__":
    main()
