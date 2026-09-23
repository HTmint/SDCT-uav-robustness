#!/usr/bin/env python3
"""Minimal AGX Orin TensorRT engine latency statistics.

Input: one or more existing .engine files.
Output: raw JSON + CSV statistics only. No paper table generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

np: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TensorRT engine benchmark on AGX Orin.")
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="Engine spec: name=/path/model.engine or /path/model.engine. Repeat for multiple engines.",
    )
    parser.add_argument("--shape", default="1,1,640,640", help="NCHW input shape.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--precision", default="FP16", help="Recorded label only.")
    parser.add_argument("--power-log", default=None, help="Optional tegrastats logfile to parse.")
    parser.add_argument("--out-json", default="orin_engine_stats.json")
    parser.add_argument("--out-csv", default="orin_engine_stats.csv")
    parser.add_argument(
        "--allow-non-orin",
        action="store_true",
        help="Debug only. Do not use non-Orin results in the paper.",
    )
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


def parse_shape(text: str) -> tuple[int, int, int, int]:
    dims = tuple(int(x) for x in text.split(","))
    if len(dims) != 4:
        raise ValueError(f"Expected NCHW shape, got {text}")
    return dims


def parse_engine_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return path.stem, path


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


def parse_power_w(path: str | None) -> str:
    if not path:
        return "N/A"
    log = Path(path)
    if not log.exists():
        return "N/A"

    import re

    totals = []
    pattern = re.compile(r"\b(VDD\w*)\s+(\d+)mW(?:/\d+mW)?")
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        rails = [int(mw) for _, mw in pattern.findall(line)]
        if rails:
            totals.append(sum(rails))
    if not totals:
        return "N/A"
    return f"{statistics.mean(totals) / 1000.0:.3f}"


class EngineRunner:
    def __init__(self, engine_path: Path, shape: tuple[int, int, int, int]):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.cuda = cuda
        self.trt = trt
        self.shape = shape
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.tensor_api = hasattr(engine, "num_io_tensors")
        self.host_in = None
        self.dev_in = None
        self.outputs = []
        self.bindings = None
        self.input_name = ""
        self.input_dtype = np.float32
        self.output_shapes = []
        self._allocate()

    def _allocate(self) -> None:
        if self.tensor_api:
            self._allocate_tensor_api()
        else:
            self._allocate_binding_api()

    def _allocate_tensor_api(self) -> None:
        cuda, trt = self.cuda, self.trt
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for name in names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.context.set_input_shape(name, self.shape)
                break
        if not self.input_name:
            raise RuntimeError("No input tensor found")

        for name in names:
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            if mode == trt.TensorIOMode.INPUT:
                self.host_in, self.dev_in, self.input_dtype = host, dev, dtype
            else:
                self.outputs.append((host, dev))
                self.output_shapes.append(list(shape))

    def _allocate_binding_api(self) -> None:
        cuda, trt = self.cuda, self.trt
        self.bindings = [0] * int(self.engine.num_bindings)
        for i in range(int(self.engine.num_bindings)):
            if self.engine.binding_is_input(i):
                self.context.set_binding_shape(i, self.shape)

        for i in range(int(self.engine.num_bindings)):
            shape = tuple(int(x) for x in self.context.get_binding_shape(i))
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings[i] = int(dev)
            if self.engine.binding_is_input(i):
                self.host_in, self.dev_in, self.input_dtype = host, dev, dtype
                self.input_name = self.engine.get_binding_name(i)
            else:
                self.outputs.append((host, dev))
                self.output_shapes.append(list(shape))

    def run_once(self, x: np.ndarray) -> float:
        cuda = self.cuda
        np.copyto(self.host_in, x.ravel())
        start = time.perf_counter()
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        if self.tensor_api:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        for host, dev in self.outputs:
            cuda.memcpy_dtoh_async(host, dev, self.stream)
        self.stream.synchronize()
        return (time.perf_counter() - start) * 1000.0


def benchmark_engine(name: str, path: Path, shape: tuple[int, int, int, int], args: argparse.Namespace) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    runner = EngineRunner(path, shape)
    x = np.zeros(shape, dtype=runner.input_dtype)
    for _ in range(args.warmup):
        runner.run_once(x)

    latencies = [runner.run_once(x) for _ in range(args.repeat)]
    mean_ms = statistics.mean(latencies)
    batch = shape[0]
    return {
        "name": name,
        "engine": str(path),
        "precision": args.precision,
        "shape": list(shape),
        "input_name": runner.input_name,
        "output_shapes": runner.output_shapes,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "latency_mean_ms": round(mean_ms, 4),
        "latency_median_ms": round(statistics.median(latencies), 4),
        "latency_p90_ms": round(percentile(latencies, 0.90), 4),
        "latency_p95_ms": round(percentile(latencies, 0.95), 4),
        "latency_min_ms": round(min(latencies), 4),
        "latency_max_ms": round(max(latencies), 4),
        "fps": round(1000.0 * batch / mean_ms, 4),
        "engine_size_mb": round(path.stat().st_size / (1024 * 1024), 4),
    }


def write_outputs(rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat_rows.append({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    global np
    args = parse_args()
    model = require_orin(args.allow_non_orin)
    import numpy as _np

    np = _np
    shape = parse_shape(args.shape)

    rows = []
    for spec in args.engine:
        name, path = parse_engine_spec(spec)
        row = benchmark_engine(name, path, shape, args)
        row["device"] = model
        rows.append(row)

    power_w = parse_power_w(args.power_log)
    for row in rows:
        row["power_avg_w"] = power_w

    write_outputs(rows, Path(args.out_json), Path(args.out_csv))
    print(json.dumps(rows, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
