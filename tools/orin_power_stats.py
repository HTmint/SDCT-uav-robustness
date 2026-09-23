#!/usr/bin/env python3
"""Minimal AGX Orin power logging and tegrastats parsing.

Modes:
1. Parse an existing tegrastats log with --parse-only.
2. Log power for --duration seconds.
3. Log power while running a command after "--".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


POWER_RE = re.compile(r"\b([A-Z][A-Z0-9_]*?)\s+([0-9.]+)mW(?:/[0-9.]+mW)?")
PRIMARY_RAILS = ("VDD_IN", "POM_5V_IN", "VDD_SYS_IN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tegrastats power logger/parser.")
    parser.add_argument("--log", required=True, help="tegrastats logfile path.")
    parser.add_argument("--interval", type=int, default=1000, help="tegrastats interval in ms.")
    parser.add_argument("--duration", type=float, default=None, help="Log for N seconds.")
    parser.add_argument("--parse-only", action="store_true", help="Only parse an existing log.")
    parser.add_argument("--out-json", default="orin_power_stats.json")
    parser.add_argument("--out-csv", default="orin_power_stats.csv")
    parser.add_argument(
        "--allow-non-orin",
        action="store_true",
        help="Debug only. Do not report non-Orin power as AGX Orin evidence.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional command after '--'.")
    return parser.parse_args()


def shell(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=5).strip()
    except Exception:
        return "N/A"


def device_model() -> str:
    return shell("tr -d '\\0' </proc/device-tree/model 2>/dev/null")


def require_orin(allow_non_orin: bool) -> str:
    model = device_model()
    if allow_non_orin:
        return model
    text = model.lower()
    if "agx" not in text or "orin" not in text:
        raise RuntimeError(
            f"Not AGX Orin: {model!r}. Use --allow-non-orin only for debug; "
            "do not report non-Orin results as AGX Orin."
        )
    return model


def clean_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def parse_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    rail_samples: dict[str, list[float]] = {}
    fallback_totals: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        rails = {name: float(mw) for name, mw in POWER_RE.findall(line)}
        rails = {name: mw for name, mw in rails.items() if name.startswith(("VDD", "POM", "VIN"))}
        if not rails:
            continue
        fallback_totals.append(sum(rails.values()))
        for name, mw in rails.items():
            rail_samples.setdefault(name, []).append(mw)

    if not rail_samples:
        raise RuntimeError(f"No power rails found in {path}")

    primary = None
    primary_values = None
    for rail in PRIMARY_RAILS:
        if rail in rail_samples:
            primary = rail
            primary_values = rail_samples[rail]
            break
    if primary_values is None:
        primary = "SUM_ALL_RAILS"
        primary_values = fallback_totals

    rail_avg_w = {
        name: round(statistics.mean(values) / 1000.0, 6)
        for name, values in sorted(rail_samples.items())
    }
    values_w = [v / 1000.0 for v in primary_values]
    return {
        "samples": len(values_w),
        "primary_rail": primary,
        "power_avg_w": round(statistics.mean(values_w), 6),
        "power_median_w": round(statistics.median(values_w), 6),
        "power_min_w": round(min(values_w), 6),
        "power_max_w": round(max(values_w), 6),
        "power_p90_w": round(percentile(values_w, 0.90), 6),
        "power_p95_w": round(percentile(values_w, 0.95), 6),
        "rail_avg_w": rail_avg_w,
    }


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def start_tegrastats(log_path: Path, interval_ms: int) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    cmd = ["tegrastats", "--interval", str(interval_ms), "--logfile", str(log_path)]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_with_power(args: argparse.Namespace) -> dict[str, Any]:
    log_path = Path(args.log)
    cmd = clean_command(args.command)
    proc = start_tegrastats(log_path, args.interval)
    command_rc = None
    command_text = "N/A"
    started = time.time()
    try:
        time.sleep(max(1.0, args.interval / 1000.0))
        if cmd:
            command_text = " ".join(cmd)
            command_rc = subprocess.run(cmd).returncode
        elif args.duration is not None:
            time.sleep(args.duration)
        else:
            raise ValueError("Provide --parse-only, --duration, or a command after '--'.")
    finally:
        stop_process(proc)
    elapsed = time.time() - started
    stats = parse_log(log_path)
    stats.update(
        {
            "log": str(log_path),
            "interval_ms": args.interval,
            "elapsed_s": round(elapsed, 3),
            "command": command_text,
            "command_returncode": command_rc,
        }
    )
    return stats


def write_outputs(stats: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in stats.items()}
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def main() -> None:
    args = parse_args()
    model = require_orin(args.allow_non_orin)
    if args.parse_only:
        stats = parse_log(Path(args.log))
        stats.update({"log": str(Path(args.log)), "command": "PARSE_ONLY", "command_returncode": None})
    else:
        stats = run_with_power(args)
    stats["device"] = model
    write_outputs(stats, Path(args.out_json), Path(args.out_csv))
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
