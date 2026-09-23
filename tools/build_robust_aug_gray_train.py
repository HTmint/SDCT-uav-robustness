#!/usr/bin/env python3
"""Build fixed offline RandAugment-Gray and AugMix-Input-Gray train sets.

The generated data follows the SDCT/Random-Aug scale convention used in this
workspace: each clean train image is listed once, and each image receives a
fixed number of offline photometric augmented copies. Labels are inherited
unchanged because no operation changes image geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps

from build_dut_gray_degradation import read_gray, stable_rng


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/dart_dut"
DEFAULT_YAML_DIR = DEFAULT_DATA_ROOT / "yamls"
DEFAULT_DRY_RUN_DIR = ROOT / "outputs/robust_aug_baselines/dry_run"

OPS = (
    "Identity",
    "AutoContrast",
    "Equalize",
    "Posterize",
    "Solarize",
    "Brightness",
    "Contrast",
    "Sharpness",
)
SIGNED_OPS = {"Brightness", "Contrast", "Sharpness"}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display: str
    seed: int
    dir_name: str
    list_name: str
    yaml_name: str
    manifest_name: str
    file_tag: str


METHODS = {
    "randaugment_gray": MethodSpec(
        key="randaugment_gray",
        display="RandAugment-Gray",
        seed=2027,
        dir_name="DUT-Gray-RandAugmentTrain",
        list_name="train_randaugment_gray.txt",
        yaml_name="dut_gray_randaugment_gray.yaml",
        manifest_name="train_randaugment_gray_manifest.csv",
        file_tag="randaugmentgray",
    ),
    "augmix_input_gray": MethodSpec(
        key="augmix_input_gray",
        display="AugMix-Input-Gray",
        seed=2028,
        dir_name="DUT-Gray-AugMixInputTrain",
        list_name="train_augmix_input_gray.txt",
        yaml_name="dut_gray_augmix_input_gray.yaml",
        manifest_name="train_augmix_input_gray_manifest.csv",
        file_tag="augmixinputgray",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed offline generic gray robustness baselines.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument(
        "--method",
        choices=["randaugment_gray", "augmix_input_gray", "all"],
        default="all",
    )
    parser.add_argument("--classes", nargs="+", default=["uav"])
    parser.add_argument("--copies-per-image", type=int, default=5)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--dry-run", action="store_true", help="Generate a small preview under outputs/ only.")
    parser.add_argument("--dry-run-count", type=int, default=8)
    parser.add_argument("--dry-run-dir", type=Path, default=DEFAULT_DRY_RUN_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this script's outputs for the selected method.")
    parser.add_argument("--skip-verify-decode", action="store_true", help="Skip full generated-image decode verification.")
    return parser.parse_args()


def read_list(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(item) for item in items) + "\n", encoding="utf-8")


def write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def read_label_text(image_path: Path) -> str:
    path = label_path(image_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_label_like(src_image: Path, dst_image: Path) -> None:
    dst_label = label_path(dst_image)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    src_label = label_path(src_image)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def pil_from_gray(gray: np.ndarray) -> Image.Image:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"Expected single-channel uint8 array, got shape={gray.shape} dtype={gray.dtype}")
    return Image.fromarray(gray)


def gray_from_pil(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"))
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def randaugment_magnitude(op: str, magnitude: int, num_bins: int, rng: np.random.Generator) -> tuple[float | int, int]:
    if op in {"Identity", "AutoContrast", "Equalize"}:
        return 0.0, 1
    if not (0 <= magnitude < num_bins):
        raise ValueError(f"RandAugment magnitude must be in [0, {num_bins - 1}], got {magnitude}.")
    if op in {"Brightness", "Contrast", "Sharpness"}:
        mag = float(np.linspace(0.0, 0.9, num_bins)[magnitude])
        sign = -1 if int(rng.integers(0, 2)) else 1
        return sign * mag, sign
    if op == "Posterize":
        bits = 8 - int(np.rint(np.arange(num_bins)[magnitude] / ((num_bins - 1) / 4)))
        return int(np.clip(bits, 1, 8)), 1
    if op == "Solarize":
        return float(np.linspace(255.0, 0.0, num_bins)[magnitude]), 1
    raise ValueError(f"Unsupported op: {op}")


def augmix_magnitude(op: str, severity: int, rng: np.random.Generator) -> tuple[float | int, int, int]:
    if op in {"Identity", "AutoContrast", "Equalize"}:
        return 0.0, 1, -1
    if not (1 <= severity <= 10):
        raise ValueError(f"AugMix severity must be in [1, 10], got {severity}.")
    index = int(rng.integers(0, severity))
    if op in {"Brightness", "Contrast", "Sharpness"}:
        mag = float(np.linspace(0.0, 0.9, 10)[index])
        sign = -1 if int(rng.integers(0, 2)) else 1
        return sign * mag, sign, index
    if op == "Posterize":
        bits = 4 - int(np.rint(np.arange(10)[index] / ((10 - 1) / 4)))
        return int(np.clip(bits, 1, 8)), 1, index
    if op == "Solarize":
        return float(np.linspace(255.0, 0.0, 10)[index]), 1, index
    raise ValueError(f"Unsupported op: {op}")


def apply_op(image: Image.Image, op: str, magnitude: float | int) -> Image.Image:
    if op == "Identity":
        return image
    if op == "AutoContrast":
        return ImageOps.autocontrast(image)
    if op == "Equalize":
        return ImageOps.equalize(image)
    if op == "Posterize":
        return ImageOps.posterize(image, int(magnitude))
    if op == "Solarize":
        return ImageOps.solarize(image, int(round(float(magnitude))))
    if op == "Brightness":
        return ImageEnhance.Brightness(image).enhance(max(0.0, 1.0 + float(magnitude)))
    if op == "Contrast":
        return ImageEnhance.Contrast(image).enhance(max(0.0, 1.0 + float(magnitude)))
    if op == "Sharpness":
        return ImageEnhance.Sharpness(image).enhance(max(0.0, 1.0 + float(magnitude)))
    raise ValueError(f"Unsupported op: {op}")


def randaugment_gray(gray: np.ndarray, seed: int, key: str) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    rng = stable_rng(seed, key)
    image = pil_from_gray(gray)
    trace: list[dict[str, object]] = []
    for step in range(2):
        op = OPS[int(rng.integers(0, len(OPS)))]
        magnitude, sign = randaugment_magnitude(op, magnitude=9, num_bins=31, rng=rng)
        image = apply_op(image, op, magnitude)
        trace.append(
            {
                "step": step + 1,
                "op": op,
                "magnitude": magnitude,
                "sign": sign if op in SIGNED_OPS else "",
                "magnitude_bin": 9 if op not in {"Identity", "AutoContrast", "Equalize"} else "",
                "num_magnitude_bins": 31,
            }
        )
    return gray_from_pil(image), trace, {"N": 2, "M": 9, "num_magnitude_bins": 31}


def augmix_input_gray(gray: np.ndarray, seed: int, key: str) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    rng = stable_rng(seed, key)
    base = pil_from_gray(gray)
    base_float = gray.astype(np.float32)
    beta = rng.dirichlet([1.0, 1.0])
    chain_weights = rng.dirichlet([1.0, 1.0, 1.0])
    mix = float(beta[0]) * base_float
    trace: list[dict[str, object]] = []
    for chain_idx in range(3):
        image = base
        depth = int(rng.integers(1, 4))
        chain_trace: list[dict[str, object]] = []
        for step in range(depth):
            op = OPS[int(rng.integers(0, len(OPS)))]
            magnitude, sign, magnitude_index = augmix_magnitude(op, severity=3, rng=rng)
            image = apply_op(image, op, magnitude)
            chain_trace.append(
                {
                    "step": step + 1,
                    "op": op,
                    "magnitude": magnitude,
                    "sign": sign if op in SIGNED_OPS else "",
                    "magnitude_index": magnitude_index if magnitude_index >= 0 else "",
                }
            )
        chain_arr = gray_from_pil(image).astype(np.float32)
        weight = float(beta[1] * chain_weights[chain_idx])
        mix += weight * chain_arr
        trace.append({"chain": chain_idx + 1, "depth": depth, "weight": weight, "ops": chain_trace})
    out = np.clip(np.rint(mix), 0, 255).astype(np.uint8)
    params = {
        "width": 3,
        "depth": "uniform{1,2,3}",
        "severity": 3,
        "dirichlet_alpha": 1.0,
        "beta_alpha": 1.0,
        "clean_weight": float(beta[0]),
        "aug_weight": float(beta[1]),
        "chain_weights": [float(x) for x in chain_weights],
        "jsd_loss": False,
    }
    return out, trace, params


def output_image_path(out_dir: Path, src_image: Path, copy_index: int, spec: MethodSpec, ext: str) -> Path:
    suffix = ext if ext.startswith(".") else f".{ext}"
    return out_dir / f"{src_image.stem}_{spec.file_tag}{copy_index:02d}{suffix.lower()}"


def render_preview(method_dir: Path, rows: list[dict[str, str]], max_sources: int = 8) -> Path | None:
    clean_rows = [r for r in rows if r["entry_kind"] == "clean"][:max_sources]
    if not clean_rows:
        return None
    tiles: list[np.ndarray] = []
    for clean_row in clean_rows:
        src = Path(clean_row["source"])
        src_img = read_gray(src)
        variants = [src_img]
        aug_rows = [r for r in rows if r["source"] == str(src) and r["entry_kind"] == "augmented"][:5]
        for row in aug_rows:
            variants.append(read_gray(Path(row["output"])))
        resized = [cv2.resize(v, (192, 108), interpolation=cv2.INTER_AREA) for v in variants]
        tiles.append(np.hstack(resized))
    grid = np.vstack(tiles)
    out = method_dir / "preview_grid.jpg"
    cv2.imwrite(str(out), grid)
    return out


def validate_manifest(
    train_list: Path,
    manifest: Path,
    clean_images: list[Path],
    val_clean: Path,
    test_clean: Path,
    copies_per_image: int,
    verify_decode: bool,
) -> dict[str, object]:
    train_entries = read_list(train_list)
    val_images = set(read_list(val_clean))
    test_images = set(read_list(test_clean))
    rows: list[dict[str, str]]
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    clean_rows = [r for r in rows if r["entry_kind"] == "clean"]
    aug_rows = [r for r in rows if r["entry_kind"] == "augmented"]
    if len(clean_rows) != len(clean_images):
        raise RuntimeError(f"Clean manifest rows mismatch: {len(clean_rows)} vs {len(clean_images)}")
    expected_aug = len(clean_images) * copies_per_image
    if len(aug_rows) != expected_aug:
        raise RuntimeError(f"Augmented manifest rows mismatch: {len(aug_rows)} vs {expected_aug}")
    if len(train_entries) != len(clean_images) + expected_aug:
        raise RuntimeError(f"Train-list entries mismatch: {len(train_entries)} vs {len(clean_images) + expected_aug}")
    if set(clean_images) & val_images or set(clean_images) & test_images or val_images & test_images:
        raise RuntimeError("Detected path overlap among clean train/val/test lists.")

    per_source = Counter(Path(r["source"]) for r in aug_rows)
    bad_counts = [str(src) for src, count in per_source.items() if count != copies_per_image]
    if bad_counts:
        raise RuntimeError(f"Some sources do not have exactly {copies_per_image} augmented copies: {bad_counts[:5]}")
    if set(per_source) != set(clean_images):
        raise RuntimeError("Augmented manifest sources do not match train_clean.txt.")

    output_paths = {Path(r["output"]) for r in aug_rows}
    if output_paths & val_images or output_paths & test_images:
        raise RuntimeError("Generated output path overlaps val/test lists.")
    source_paths = {Path(r["source"]) for r in aug_rows}
    if source_paths & val_images or source_paths & test_images:
        raise RuntimeError("Generated rows use val/test sources.")

    label_mismatches = 0
    shape_mismatches = 0
    unreadable = 0
    dtype_or_channels_bad = 0
    decoded = 0
    for row in aug_rows:
        source = Path(row["source"])
        output = Path(row["output"])
        if read_label_text(source) != read_label_text(output):
            label_mismatches += 1
        if verify_decode:
            src_img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            out_img = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
            if src_img is None or out_img is None:
                unreadable += 1
                continue
            decoded += 1
            src_shape = src_img.shape[:2]
            out_shape = out_img.shape[:2]
            if src_shape != out_shape:
                shape_mismatches += 1
            if out_img.dtype != np.uint8 or out_img.ndim != 2:
                dtype_or_channels_bad += 1
    if label_mismatches:
        raise RuntimeError(f"Label content mismatches: {label_mismatches}")
    if shape_mismatches:
        raise RuntimeError(f"Image shape mismatches: {shape_mismatches}")
    if unreadable:
        raise RuntimeError(f"Unreadable generated/source images: {unreadable}")
    if dtype_or_channels_bad:
        raise RuntimeError(f"Generated images not single-channel uint8: {dtype_or_channels_bad}")

    return {
        "train_entries": len(train_entries),
        "manifest_rows": len(rows),
        "clean_entries": len(clean_rows),
        "augmented_entries": len(aug_rows),
        "sources_with_five_augmented": len(per_source),
        "decoded_augmented_images": decoded,
        "label_mismatches": label_mismatches,
        "shape_mismatches": shape_mismatches,
        "unreadable_images": unreadable,
        "dtype_or_channels_bad": dtype_or_channels_bad,
        "train_val_test_path_overlap": False,
    }


def build_method(args: argparse.Namespace, spec: MethodSpec) -> dict[str, object]:
    data_root = args.data_root.resolve()
    yaml_dir = args.yaml_dir.resolve()
    train_clean = yaml_dir / "train_clean.txt"
    val_clean = yaml_dir / "val_clean.txt"
    test_clean = yaml_dir / "test_clean.txt"
    for required in (train_clean, val_clean, test_clean):
        if not required.exists():
            raise FileNotFoundError(required)

    clean_images = read_list(train_clean)
    if args.dry_run:
        clean_images = clean_images[: args.dry_run_count]
        out_root = (args.dry_run_dir / spec.key / "train").resolve()
        train_list = (args.dry_run_dir / spec.key / spec.list_name).resolve()
        data_yaml = (args.dry_run_dir / spec.key / spec.yaml_name).resolve()
        manifest = (args.dry_run_dir / spec.key / spec.manifest_name).resolve()
    else:
        out_root = (data_root / spec.dir_name / "train").resolve()
        train_list = (yaml_dir / spec.list_name).resolve()
        data_yaml = (yaml_dir / spec.yaml_name).resolve()
        manifest = (yaml_dir / spec.manifest_name).resolve()

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_root}. Use --overwrite to regenerate.")
        shutil.rmtree(out_root)
    for path in (train_list, data_yaml, manifest):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to regenerate.")

    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    rows: list[dict[str, str]] = []
    op_counts: Counter[str] = Counter()
    for image_path in clean_images:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing clean train image: {image_path}")
        gray = read_gray(image_path)
        source_label = label_path(image_path)
        source_label_sha = sha256_file(source_label) if source_label.exists() else hashlib.sha256(b"").hexdigest()
        rows.append(
            {
                "method": spec.display,
                "entry_kind": "clean",
                "source": str(image_path),
                "output": str(image_path),
                "output_label": str(source_label),
                "copy_index": "0",
                "seed": str(spec.seed),
                "source_width": str(gray.shape[1]),
                "source_height": str(gray.shape[0]),
                "output_width": str(gray.shape[1]),
                "output_height": str(gray.shape[0]),
                "source_label_sha256": source_label_sha,
                "output_label_sha256": source_label_sha,
                "operation_trace_json": json.dumps([{"op": "Identity"}], separators=(",", ":")),
                "params_json": json.dumps({"clean_entry": True}, separators=(",", ":")),
            }
        )
        for copy_index in range(1, args.copies_per_image + 1):
            key = f"{spec.key}:{image_path.name}:{copy_index}"
            if spec.key == "randaugment_gray":
                aug, trace, params = randaugment_gray(gray, spec.seed, key)
            elif spec.key == "augmix_input_gray":
                aug, trace, params = augmix_input_gray(gray, spec.seed, key)
            else:
                raise ValueError(f"Unknown method: {spec.key}")
            out_img = output_image_path(out_root, image_path, copy_index, spec, args.image_ext)
            if not cv2.imwrite(str(out_img), aug):
                raise RuntimeError(f"Failed to write image: {out_img}")
            write_label_like(image_path, out_img)
            generated.append(out_img)
            for event in trace:
                if "op" in event:
                    op_counts[str(event["op"])] += 1
                else:
                    for op_event in event.get("ops", []):
                        op_counts[str(op_event["op"])] += 1
            output_label = label_path(out_img)
            rows.append(
                {
                    "method": spec.display,
                    "entry_kind": "augmented",
                    "source": str(image_path),
                    "output": str(out_img),
                    "output_label": str(output_label),
                    "copy_index": str(copy_index),
                    "seed": str(spec.seed),
                    "source_width": str(gray.shape[1]),
                    "source_height": str(gray.shape[0]),
                    "output_width": str(aug.shape[1]),
                    "output_height": str(aug.shape[0]),
                    "source_label_sha256": source_label_sha,
                    "output_label_sha256": sha256_file(output_label),
                    "operation_trace_json": json.dumps(trace, separators=(",", ":")),
                    "params_json": json.dumps(params, separators=(",", ":")),
                }
            )

    write_lines(train_list, [*clean_images, *generated])
    write_yaml(
        data_yaml,
        {
            "nc": len(args.classes),
            "names": args.classes,
            "ch": 1,
            "channels": 1,
            "train": str(train_list),
            "val": str(val_clean),
            "test": str(test_clean),
        },
    )

    fieldnames = [
        "method",
        "entry_kind",
        "source",
        "output",
        "output_label",
        "copy_index",
        "seed",
        "source_width",
        "source_height",
        "output_width",
        "output_height",
        "source_label_sha256",
        "output_label_sha256",
        "operation_trace_json",
        "params_json",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    preview = render_preview(manifest.parent, rows) if args.dry_run else None
    verification = validate_manifest(
        train_list=train_list,
        manifest=manifest,
        clean_images=clean_images,
        val_clean=val_clean,
        test_clean=test_clean,
        copies_per_image=args.copies_per_image,
        verify_decode=not args.skip_verify_decode,
    )
    report_path = manifest.with_suffix(".verification.json")
    report = {
        "method": spec.display,
        "seed": spec.seed,
        "dry_run": bool(args.dry_run),
        "output_root": str(out_root),
        "train_list": str(train_list),
        "data_yaml": str(data_yaml),
        "manifest": str(manifest),
        "preview_grid": str(preview) if preview else "",
        "operation_counts": dict(sorted(op_counts.items())),
        "verification": verification,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.copies_per_image != 5:
        print(f"Warning: copies-per-image is {args.copies_per_image}; formal protocol expects 5.", flush=True)
    selected = list(METHODS) if args.method == "all" else [args.method]
    reports = [build_method(args, METHODS[key]) for key in selected]
    print("Robust gray augmentation train build complete.")
    for report in reports:
        v = report["verification"]
        print(f"\nmethod:             {report['method']}")
        print(f"seed:               {report['seed']}")
        print(f"dry_run:            {report['dry_run']}")
        print(f"output root:        {report['output_root']}")
        print(f"train list:         {report['train_list']}")
        print(f"data yaml:          {report['data_yaml']}")
        print(f"manifest:           {report['manifest']}")
        print(f"verification json:  {Path(report['manifest']).with_suffix('.verification.json')}")
        if report["preview_grid"]:
            print(f"preview grid:       {report['preview_grid']}")
        print(f"clean entries:      {v['clean_entries']}")
        print(f"augmented entries:  {v['augmented_entries']}")
        print(f"train entries:      {v['train_entries']}")
        print(f"decoded augmented:  {v['decoded_augmented_images']}")
        print("operation counts:")
        for op, count in report["operation_counts"].items():
            print(f"  {op}: {count}")


if __name__ == "__main__":
    main()
