#!/usr/bin/env python3
"""Benchmark an existing TensorRT engine on Jetson/AGX Orin.

This script assumes the `.engine` file already exists. It does not build
TensorRT engines from ONNX.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

np: Any = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark latency/FPS for an existing TensorRT engine."
    )
    parser.add_argument("--engine", required=True, help="Path to a TensorRT .engine file.")
    parser.add_argument("--name", default=None, help="Method/model name recorded in outputs.")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional image file or directory. If omitted, use synthetic tensor input.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan source.")
    parser.add_argument(
        "--input-shape",
        default="1,1,640,640",
        help="Input shape used for dynamic engines, e.g. 1,1,640,640. Use 'auto' for static engines.",
    )
    parser.add_argument(
        "--precision",
        default="engine",
        help="Precision label recorded in output, e.g. FP16, FP32, INT8, or engine.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument(
        "--measure",
        choices=("end_to_end", "execute_only"),
        default="end_to_end",
        help="end_to_end includes H2D, execute, D2H; execute_only times TensorRT enqueue only.",
    )
    parser.add_argument(
        "--random-mode",
        choices=("zeros", "uniform"),
        default="zeros",
        help="Synthetic input content when --source is omitted.",
    )
    parser.add_argument(
        "--tegrastats-log",
        default=None,
        help="Optional tegrastats log path to parse average power rails.",
    )
    parser.add_argument(
        "--allow-non-orin",
        action="store_true",
        help="Allow execution on non-AGX-Orin hardware for debugging only.",
    )
    parser.add_argument(
        "--allow-missing-power",
        action="store_true",
        help="Allow benchmark outputs without tegrastats power values for debugging only.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--latencies-csv", default=None)
    return parser.parse_args()


def parse_shape(value: str) -> tuple[int, ...] | None:
    if value.lower() == "auto":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("input shape cannot be empty")
    return tuple(int(p) for p in parts)


def has_dynamic_dim(shape: tuple[int, ...]) -> bool:
    return any(int(dim) < 0 for dim in shape)


def volume(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        if int(dim) < 0:
            raise ValueError(f"Cannot allocate dynamic shape: {shape}")
        n *= int(dim)
    return n


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return xs[idx]


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


def require_agx_orin(allow_non_orin: bool) -> str:
    model = device_tree_model()
    if allow_non_orin:
        return model
    if not is_agx_orin_model(model):
        raise RuntimeError(
            "Refusing to run deployment benchmark because this is not detected as AGX Orin. "
            f"Detected device model: {model!r}. Use --allow-non-orin only for debugging; "
            "do not report non-Orin results as AGX Orin."
        )
    return model


def jetson_release() -> str:
    path = Path("/etc/nv_tegra_release")
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return "N/A"
    return "N/A"


def collect_images(source: str | None, recursive: bool) -> list[Path] | None:
    if source is None:
        return None
    path = Path(source)
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    images = sorted(p for p in path.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images


def read_image(path: Path, channels: int, height: int, width: int) -> np.ndarray:
    try:
        import cv2

        if channels == 1:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cv2 failed to read {path}")
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            image = image[None, :, :]
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cv2 failed to read {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            image = image.transpose(2, 0, 1)
    except Exception:
        from PIL import Image

        mode = "L" if channels == 1 else "RGB"
        pil = Image.open(path).convert(mode).resize((width, height))
        image = np.asarray(pil)
        if channels == 1:
            image = image[None, :, :]
        else:
            image = image.transpose(2, 0, 1)
    return image.astype(np.float32) / 255.0


def make_input(
    shape: tuple[int, ...],
    dtype: np.dtype,
    images: list[Path] | None,
    image_offset: int,
    random_mode: str,
) -> np.ndarray:
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW input shape, got {shape}")
    batch, channels, height, width = [int(v) for v in shape]
    if images is None:
        if random_mode == "uniform":
            arr = np.random.random_sample(shape).astype(np.float32)
        else:
            arr = np.zeros(shape, dtype=np.float32)
    else:
        samples = []
        for i in range(batch):
            samples.append(read_image(images[(image_offset + i) % len(images)], channels, height, width))
        arr = np.stack(samples, axis=0)

    if np.issubdtype(dtype, np.floating):
        return np.ascontiguousarray(arr.astype(dtype))
    return np.ascontiguousarray(np.clip(arr * 255.0, 0, 255).astype(dtype))


def parse_tegrastats_power(path: str | None) -> dict[str, Any]:
    if not path:
        return {"power_avg_w": "N/A", "power_rails_avg_mw": {}}
    log_path = Path(path)
    if not log_path.exists():
        return {"power_avg_w": "N/A", "power_rails_avg_mw": {}, "power_note": "log_not_found"}

    rail_samples: dict[str, list[int]] = {}
    totals: list[int] = []
    pattern = re.compile(r"\b([A-Z0-9_]+)\s+(\d+)mW(?:/\d+mW)?")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        rails = {name: int(mw) for name, mw in pattern.findall(line) if name.startswith("VDD")}
        if not rails:
            continue
        totals.append(sum(rails.values()))
        for name, mw in rails.items():
            rail_samples.setdefault(name, []).append(mw)

    if not totals:
        return {"power_avg_w": "N/A", "power_rails_avg_mw": {}, "power_note": "no_power_rails"}
    rails_avg = {name: round(statistics.mean(vals), 3) for name, vals in sorted(rail_samples.items())}
    return {
        "power_avg_w": f"{statistics.mean(totals) / 1000.0:.3f}",
        "power_rails_avg_mw": rails_avg,
        "power_samples": len(totals),
    }


@dataclass
class TensorBuffer:
    name: str
    is_input: bool
    shape: tuple[int, ...]
    dtype: np.dtype
    host: Any
    device: Any
    index: int | None = None


class TRTSession:
    def __init__(self, engine_path: Path, input_shape: tuple[int, ...] | None):
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT engine benchmark requires Python packages 'tensorrt' and 'pycuda' "
                "on the AGX Orin environment."
            ) from exc

        self.cuda = cuda
        self.trt = trt
        self.engine_path = engine_path
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.uses_tensor_api = hasattr(engine, "num_io_tensors")
        self.buffers: list[TensorBuffer] = []
        self.inputs: list[TensorBuffer] = []
        self.outputs: list[TensorBuffer] = []
        self.bindings: list[int] | None = None
        self._allocate(input_shape)

    def _dtype(self, trt_dtype: Any) -> np.dtype:
        return np.dtype(self.trt.nptype(trt_dtype))

    def _shape_tuple(self, shape: Any) -> tuple[int, ...]:
        return tuple(int(dim) for dim in shape)

    def _allocate(self, requested_input_shape: tuple[int, ...] | None) -> None:
        if self.uses_tensor_api:
            self._allocate_tensor_api(requested_input_shape)
        else:
            self._allocate_binding_api(requested_input_shape)
        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected exactly one input tensor, got {len(self.inputs)}")

    def _allocate_tensor_api(self, requested_input_shape: tuple[int, ...] | None) -> None:
        trt = self.trt
        cuda = self.cuda
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_names = [
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        for name in input_names:
            shape = self._shape_tuple(self.engine.get_tensor_shape(name))
            if has_dynamic_dim(shape):
                if requested_input_shape is None:
                    raise ValueError(f"Dynamic input {name} needs --input-shape.")
                shape = requested_input_shape
            elif requested_input_shape is not None and tuple(shape) != tuple(requested_input_shape):
                raise ValueError(
                    f"Static engine input shape {shape} does not match --input-shape "
                    f"{requested_input_shape}. Use --input-shape auto or provide the static shape."
                )
            self.context.set_input_shape(name, shape)

        for idx, name in enumerate(names):
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self._shape_tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            buf = TensorBuffer(name, is_input, shape, dtype, host, device, idx)
            self.buffers.append(buf)
            (self.inputs if is_input else self.outputs).append(buf)

    def _allocate_binding_api(self, requested_input_shape: tuple[int, ...] | None) -> None:
        cuda = self.cuda
        bindings: list[int] = [0] * int(self.engine.num_bindings)
        for idx in range(int(self.engine.num_bindings)):
            if self.engine.binding_is_input(idx):
                shape = self._shape_tuple(self.engine.get_binding_shape(idx))
                if has_dynamic_dim(shape):
                    if requested_input_shape is None:
                        raise ValueError(f"Dynamic input binding {idx} needs --input-shape.")
                    shape = requested_input_shape
                elif requested_input_shape is not None and tuple(shape) != tuple(requested_input_shape):
                    raise ValueError(
                        f"Static engine input shape {shape} does not match --input-shape "
                        f"{requested_input_shape}. Use --input-shape auto or provide the static shape."
                    )
                self.context.set_binding_shape(idx, shape)

        for idx in range(int(self.engine.num_bindings)):
            name = self.engine.get_binding_name(idx)
            is_input = self.engine.binding_is_input(idx)
            shape = self._shape_tuple(self.context.get_binding_shape(idx))
            dtype = self._dtype(self.engine.get_binding_dtype(idx))
            host = cuda.pagelocked_empty(volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            bindings[idx] = int(device)
            buf = TensorBuffer(name, is_input, shape, dtype, host, device, idx)
            self.buffers.append(buf)
            (self.inputs if is_input else self.outputs).append(buf)
        self.bindings = bindings

    def _execute_async(self) -> None:
        if self.uses_tensor_api:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.inputs[0].shape

    @property
    def input_dtype(self) -> np.dtype:
        return self.inputs[0].dtype

    def infer_once(self, input_array: np.ndarray, measure: str) -> float:
        cuda = self.cuda
        inp = self.inputs[0]
        np.copyto(inp.host, input_array.ravel())

        if measure == "execute_only":
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            self.stream.synchronize()
            start_event = cuda.Event()
            end_event = cuda.Event()
            start_event.record(self.stream)
            self._execute_async()
            end_event.record(self.stream)
            end_event.synchronize()
            return float(start_event.time_till(end_event))

        start = time.perf_counter()
        cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        self._execute_async()
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def info(self) -> dict[str, Any]:
        return {
            "tensorrt_version": getattr(self.trt, "__version__", "N/A"),
            "input_name": self.inputs[0].name,
            "input_shape": list(self.inputs[0].shape),
            "input_dtype": str(self.inputs[0].dtype),
            "outputs": [
                {"name": out.name, "shape": list(out.shape), "dtype": str(out.dtype)}
                for out in self.outputs
            ],
        }


def write_csv(row: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def write_md(row: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in row.items()}
    headers = list(flat)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(flat[h] for h in headers) + " |",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(result: dict[str, Any], path: str | None) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=True)
    if path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def write_latencies(values: list[float], path: str | None) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "latency_ms"])
        writer.writeheader()
        for idx, value in enumerate(values):
            writer.writerow({"iter": idx, "latency_ms": f"{value:.6f}"})


def main() -> None:
    global np
    args = parse_args()
    device_model = require_agx_orin(args.allow_non_orin)
    if not args.allow_missing_power and not args.tegrastats_log:
        raise RuntimeError(
            "Power measurement is required for paper-facing AGX Orin deployment results. "
            "Start tegrastats logging and pass --tegrastats-log, or use "
            "--allow-missing-power only for debugging."
        )
    import numpy as _np

    np = _np
    engine_path = Path(args.engine).resolve()
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    images = collect_images(args.source, args.recursive)
    requested_shape = parse_shape(args.input_shape)
    session = TRTSession(engine_path, requested_shape)
    info = session.info()
    input_shape = tuple(info["input_shape"])
    batch = int(input_shape[0])

    for i in range(args.warmup):
        arr = make_input(input_shape, session.input_dtype, images, i * batch, args.random_mode)
        session.infer_once(arr, args.measure)

    latencies: list[float] = []
    for i in range(args.repeat):
        arr = make_input(input_shape, session.input_dtype, images, i * batch, args.random_mode)
        latencies.append(session.infer_once(arr, args.measure))

    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p90_ms = percentile(latencies, 0.90)
    p95_ms = percentile(latencies, 0.95)
    fps = 1000.0 * batch / mean_ms if mean_ms > 0 else float("nan")
    power = parse_tegrastats_power(args.tegrastats_log)
    if not args.allow_missing_power and power.get("power_avg_w") == "N/A":
        raise RuntimeError(
            "Power measurement was required but no valid tegrastats VDD power samples were parsed. "
            f"Log path: {args.tegrastats_log!r}. Use --allow-missing-power only for debugging."
        )

    result: dict[str, Any] = {
        "name": args.name or engine_path.stem,
        "engine": str(engine_path),
        "engine_size_mb": f"{engine_path.stat().st_size / (1024 * 1024):.3f}",
        "source": args.source or "synthetic_tensor",
        "precision": args.precision,
        "measurement_mode": args.measure,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "batch": batch,
        "latency_mean_ms": f"{mean_ms:.3f}",
        "latency_median_ms": f"{median_ms:.3f}",
        "latency_p90_ms": f"{p90_ms:.3f}",
        "latency_p95_ms": f"{p95_ms:.3f}",
        "latency_min_ms": f"{min(latencies):.3f}",
        "latency_max_ms": f"{max(latencies):.3f}",
        "fps": f"{fps:.3f}",
        "power_avg_w": power.get("power_avg_w", "N/A"),
        "power_rails_avg_mw": power.get("power_rails_avg_mw", {}),
        "device": device_model,
        "agx_orin_required": not args.allow_non_orin,
        "power_required": not args.allow_missing_power,
        "uname": platform.platform(),
        "machine": platform.machine(),
        "jetson_release": jetson_release(),
        "cuda_driver_version": command_output(["bash", "-lc", "cat /proc/driver/nvidia/version 2>/dev/null"]),
        **info,
    }

    write_latencies(latencies, args.latencies_csv)
    write_csv(result, args.output_csv)
    write_md(result, args.output_md)
    write_json(result, args.output_json)


if __name__ == "__main__":
    main()
