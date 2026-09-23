#!/usr/bin/env python3
"""Build offline SDCT mechanism-ablation training sets for grayscale DUT data.

The ablation sets keep the same list scale as SDCT-Full / DART / Clean Repeat:
clean train images are included once, and every clean image receives a fixed
number of degraded copies. Only the degradation sampling policy changes.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import yaml

from build_dut_gray_degradation import (
    TEST_VARIANTS,
    apply_brightness,
    apply_contrast,
    apply_motion_blur,
    apply_noise,
    read_gray,
    stable_rng,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/dart_dut"
DEFAULT_YAML_DIR = DEFAULT_DATA_ROOT / "yamls"
CONTINUOUS_TYPES = ("low_light", "low_contrast", "noise", "motion_blur", "mixed")
NO_MIXED_TYPES = ("low_light", "low_contrast", "noise", "motion_blur")
DISCRETE_TYPES = ("low_light", "low_contrast", "noise", "motion_blur", "mixed")
DISCRETE_LEVELS = ("L1", "L2", "L3")

VARIANT_INFO = {
    "no_mixed": {
        "dir_name": "DUT-Gray-SDCTAblation-NoMixedTrain",
        "list_name": "train_sdct_no_mixed.txt",
        "yaml_name": "dut_gray_sdct_no_mixed.yaml",
        "manifest_name": "train_sdct_no_mixed_manifest.csv",
        "file_tag": "sdctnomixed",
    },
    "discrete": {
        "dir_name": "DUT-Gray-SDCTAblation-DiscreteTrain",
        "list_name": "train_sdct_discrete.txt",
        "yaml_name": "dut_gray_sdct_discrete.yaml",
        "manifest_name": "train_sdct_discrete_manifest.csv",
        "file_tag": "sdctdiscrete",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SDCT mechanism-ablation train data and YAMLs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--dataset-name", default="DUT")
    parser.add_argument("--classes", nargs="+", default=["uav"])
    parser.add_argument("--variant", choices=["no_mixed", "discrete", "all"], default="all")
    parser.add_argument("--copies-per-image", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--motion-direction", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ablation outputs.")
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


def copy_label(src_image: Path, dst_image: Path) -> None:
    src_label = src_image.with_suffix(".txt")
    dst_label = dst_image.with_suffix(".txt")
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def continuous_params(kind: str, rng) -> tuple[str, dict]:
    if kind == "low_light":
        return "", {"brightness": float(rng.uniform(0.35, 0.80))}
    if kind == "low_contrast":
        return "", {"contrast": float(rng.uniform(0.30, 0.85))}
    if kind == "noise":
        return "", {"noise": float(rng.uniform(5.0, 20.0))}
    if kind == "motion_blur":
        return "", {"blur": int(rng.choice([5, 7, 9, 11, 13]))}
    if kind == "mixed":
        return "", {
            "brightness": float(rng.uniform(0.35, 0.80)),
            "contrast": float(rng.uniform(0.30, 0.85)),
            "noise": float(rng.uniform(5.0, 20.0)),
            "blur": int(rng.choice([5, 7, 9, 11, 13])),
        }
    raise ValueError(f"Unknown continuous type: {kind}")


def discrete_params(kind: str, rng) -> tuple[str, dict]:
    level = str(rng.choice(DISCRETE_LEVELS))
    condition = f"{kind}_{level}"
    params = {k: v for k, v in TEST_VARIANTS[condition].items() if k in {"brightness", "contrast", "noise", "blur"}}
    return level, params


def choose_policy(variant: str, rng) -> tuple[str, str, dict]:
    if variant == "no_mixed":
        kind = str(rng.choice(NO_MIXED_TYPES))
        level, params = continuous_params(kind, rng)
        return kind, level, params
    if variant == "discrete":
        kind = str(rng.choice(DISCRETE_TYPES))
        level, params = discrete_params(kind, rng)
        return kind, level, params
    raise ValueError(f"Unsupported variant: {variant}")


def apply_degradation(gray, params: dict, rng, motion_direction: str):
    out = gray.copy()
    if "brightness" in params:
        out = apply_brightness(out, params["brightness"])
    if "contrast" in params:
        out = apply_contrast(out, params["contrast"])
    if "noise" in params:
        out = apply_noise(out, params["noise"], rng)
    if "blur" in params:
        out = apply_motion_blur(out, params["blur"], motion_direction)
    return out


def output_image_path(out_dir: Path, src_image: Path, copy_index: int, variant: str, kind: str, level: str, image_ext: str) -> Path:
    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    tag = VARIANT_INFO[variant]["file_tag"]
    suffix = f"{kind}_{level}" if level else kind
    return out_dir / f"{src_image.stem}_{tag}{copy_index:02d}_{suffix}{ext.lower()}"


def build_variant(args: argparse.Namespace, variant: str) -> dict[str, object]:
    data_root = args.data_root.resolve()
    yaml_dir = args.yaml_dir.resolve()
    info = VARIANT_INFO[variant]

    out_root = data_root / str(info["dir_name"]) / "train"
    train_clean = yaml_dir / "train_clean.txt"
    val_clean = yaml_dir / "val_clean.txt"
    test_clean = yaml_dir / "test_clean.txt"
    train_list = yaml_dir / str(info["list_name"])
    data_yaml = yaml_dir / str(info["yaml_name"])
    manifest = yaml_dir / str(info["manifest_name"])

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_root}. Use --overwrite to regenerate.")
        shutil.rmtree(out_root)
    for path in (train_list, data_yaml, manifest):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to regenerate.")

    clean_images = read_list(train_clean)
    for p in clean_images:
        if not p.exists():
            raise FileNotFoundError(f"Missing clean train image: {p}")
    for p in (val_clean, test_clean):
        if not p.exists():
            raise FileNotFoundError(f"Missing clean split list: {p}")

    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest_rows: list[dict[str, str]] = []
    type_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}

    for image_path in clean_images:
        gray = read_gray(image_path)
        for copy_index in range(1, args.copies_per_image + 1):
            key = f"{variant}:{image_path.name}:{copy_index}"
            rng = stable_rng(args.seed, key)
            kind, level, params = choose_policy(variant, rng)
            pixel_rng = stable_rng(args.seed, f"{key}:pixels")
            degraded = apply_degradation(gray, params, pixel_rng, args.motion_direction)
            out_img = output_image_path(out_root, image_path, copy_index, variant, kind, level, args.image_ext)
            ok = cv2.imwrite(str(out_img), degraded)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_img}")
            copy_label(image_path, out_img)
            generated.append(out_img.resolve())
            type_counts[kind] = type_counts.get(kind, 0) + 1
            if level:
                level_counts[level] = level_counts.get(level, 0) + 1
            manifest_rows.append(
                {
                    "variant": variant,
                    "source": str(image_path),
                    "output": str(out_img.resolve()),
                    "copy_index": str(copy_index),
                    "type": kind,
                    "severity": level,
                    **{k: str(v) for k, v in params.items()},
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

    fieldnames = ["variant", "source", "output", "copy_index", "type", "severity", "brightness", "contrast", "noise", "blur"]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "variant": variant,
        "out_root": out_root,
        "train_list": train_list,
        "data_yaml": data_yaml,
        "manifest": manifest,
        "clean_count": len(clean_images),
        "generated_count": len(generated),
        "train_count": len(clean_images) + len(generated),
        "type_counts": type_counts,
        "level_counts": level_counts,
    }


def main() -> None:
    args = parse_args()
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be >= 1.")

    variants = ["no_mixed", "discrete"] if args.variant == "all" else [args.variant]
    summaries = [build_variant(args, variant) for variant in variants]

    print("SDCT ablation train build complete.")
    for summary in summaries:
        print(f"\nvariant:            {summary['variant']}")
        print(f"output root:        {summary['out_root']}")
        print(f"clean images:       {summary['clean_count']}")
        print(f"degraded copies:    {summary['generated_count']}")
        print(f"train list images:  {summary['train_count']}")
        print(f"train list:         {summary['train_list']}")
        print(f"data yaml:          {summary['data_yaml']}")
        print(f"manifest:           {summary['manifest']}")
        print("type counts:")
        for kind, count in sorted(summary["type_counts"].items()):
            print(f"  {kind}: {count}")
        if summary["level_counts"]:
            print("severity counts:")
            for level, count in sorted(summary["level_counts"].items()):
                print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
