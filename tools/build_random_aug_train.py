#!/usr/bin/env python3
"""Build a fixed offline Random-Aug training set for grayscale DUT data.

The output is a scale-matched baseline for DART: clean train images are kept
once, and each clean image gets a fixed number of randomly parameterized
photometric degraded copies. Validation/test lists stay clean.
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
RANDOM_TYPES = ("low_light", "low_contrast", "noise", "motion_blur", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed offline Random-Aug train data and YAML.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--dataset-name", default="DUT")
    parser.add_argument("--classes", nargs="+", default=["uav"])
    parser.add_argument("--copies-per-image", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--motion-direction", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Random-Aug outputs.")
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


def random_params(kind: str, rng) -> dict:
    if kind == "low_light":
        return {"brightness": float(rng.uniform(0.35, 0.80))}
    if kind == "low_contrast":
        return {"contrast": float(rng.uniform(0.30, 0.85))}
    if kind == "noise":
        return {"noise": float(rng.uniform(5.0, 20.0))}
    if kind == "motion_blur":
        return {"blur": int(rng.choice([5, 7, 9, 11, 13]))}
    if kind == "mixed":
        return {
            "brightness": float(rng.uniform(0.35, 0.80)),
            "contrast": float(rng.uniform(0.30, 0.85)),
            "noise": float(rng.uniform(5.0, 20.0)),
            "blur": int(rng.choice([5, 7, 9, 11, 13])),
        }
    raise ValueError(f"Unknown random augmentation type: {kind}")


def apply_random_aug(gray, params: dict, rng, motion_direction: str):
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


def output_image_path(out_dir: Path, src_image: Path, copy_index: int, kind: str, image_ext: str) -> Path:
    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    return out_dir / f"{src_image.stem}_randaug{copy_index:02d}_{kind}{ext.lower()}"


def main() -> None:
    args = parse_args()
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be >= 1.")

    data_root = args.data_root.resolve()
    yaml_dir = args.yaml_dir.resolve()
    random_root = data_root / f"{args.dataset_name}-Gray-RandomAugTrain" / "train"
    train_clean = yaml_dir / "train_clean.txt"
    val_clean = yaml_dir / "val_clean.txt"
    test_clean = yaml_dir / "test_clean.txt"
    train_random_aug = yaml_dir / "train_random_aug.txt"
    random_yaml = yaml_dir / f"{args.dataset_name.lower().replace('-', '_')}_gray_random_aug.yaml"
    manifest = yaml_dir / "train_random_aug_manifest.csv"

    if random_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {random_root}. Use --overwrite to regenerate.")
        shutil.rmtree(random_root)
    for path in (train_random_aug, random_yaml, manifest):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to regenerate.")

    clean_images = read_list(train_clean)
    for p in clean_images:
        if not p.exists():
            raise FileNotFoundError(f"Missing clean train image: {p}")
    if not val_clean.exists():
        raise FileNotFoundError(f"Missing val list: {val_clean}")
    if not test_clean.exists():
        raise FileNotFoundError(f"Missing test list: {test_clean}")

    random_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest_rows: list[dict[str, str]] = []
    type_counts = {name: 0 for name in RANDOM_TYPES}

    for image_path in clean_images:
        gray = read_gray(image_path)
        for copy_index in range(1, args.copies_per_image + 1):
            key = f"{image_path.name}:{copy_index}"
            rng = stable_rng(args.seed, key)
            kind = str(rng.choice(RANDOM_TYPES))
            params = random_params(kind, rng)
            aug_rng = stable_rng(args.seed, f"{key}:pixels")
            aug = apply_random_aug(gray, params, aug_rng, args.motion_direction)
            out_img = output_image_path(random_root, image_path, copy_index, kind, args.image_ext)
            ok = cv2.imwrite(str(out_img), aug)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_img}")
            copy_label(image_path, out_img)
            generated.append(out_img.resolve())
            type_counts[kind] += 1
            manifest_rows.append(
                {
                    "source": str(image_path),
                    "output": str(out_img.resolve()),
                    "copy_index": str(copy_index),
                    "type": kind,
                    **{k: str(v) for k, v in params.items()},
                }
            )

    write_lines(train_random_aug, [*clean_images, *generated])
    write_yaml(
        random_yaml,
        {
            "nc": len(args.classes),
            "names": args.classes,
            "ch": 1,
            "channels": 1,
            "train": str(train_random_aug),
            "val": str(val_clean),
            "test": str(test_clean),
        },
    )

    fieldnames = ["source", "output", "copy_index", "type", "brightness", "contrast", "noise", "blur"]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Random-Aug train build complete.")
    print(f"data_root:          {data_root}")
    print(f"random_root:        {random_root}")
    print(f"clean images:       {len(clean_images)}")
    print(f"random copies:      {len(generated)}")
    print(f"train list images:  {len(clean_images) + len(generated)}")
    print(f"train list:         {train_random_aug}")
    print(f"data yaml:          {random_yaml}")
    print(f"manifest:           {manifest}")
    print("random type counts:")
    for kind in RANDOM_TYPES:
        print(f"  {kind}: {type_counts[kind]}")


if __name__ == "__main__":
    main()
