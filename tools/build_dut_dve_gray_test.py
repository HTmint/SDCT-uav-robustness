#!/usr/bin/env python3
"""Build an isolated grayscale test-only copy of the DUT-Dve dataset.

The source dataset is never modified. Only images/test and labels/test are
read; labels are copied unchanged because grayscale conversion is photometric.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml


DEFAULT_SOURCE = Path("data/DUT-Dve")
DEFAULT_OUTPUT = Path("outputs/dut_dve_zero_shot/gray_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    return sorted(path.glob("*.jpg"))


def validate_label(label_path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    object_count = 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, errors
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric YOLO fields")
            continue
        if class_id != 0:
            errors.append(f"{label_path}:{line_number}: class id {class_id}, expected 0")
        if any(value < 0.0 or value > 1.0 for value in coords):
            errors.append(f"{label_path}:{line_number}: normalized coordinate outside [0, 1]")
        object_count += 1
    return object_count, errors


def write_yaml(output_root: Path) -> Path:
    test_list = output_root / "test.txt"
    yaml_path = output_root / "dut_dve_gray_test.yaml"
    content = {
        "nc": 1,
        "names": ["uav"],
        "ch": 1,
        "channels": 1,
        "train": str(test_list),
        "val": str(test_list),
        "test": str(test_list),
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)
    return yaml_path


def main() -> None:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_images = source_root / "images" / "test"
    source_labels = source_root / "labels" / "test"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("source root must contain images/test and labels/test")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty; pass --overwrite to regenerate: {output_root}")

    output_images = output_root / "images" / "test"
    output_labels = output_root / "labels" / "test"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    sources = image_paths(source_images)
    if not sources:
        raise RuntimeError(f"no JPG test images found in {source_images}")

    manifest_rows: list[dict[str, str]] = []
    label_errors: list[str] = []
    total_objects = 0
    for source_image in sources:
        label_path = source_labels / f"{source_image.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for {source_image.name}: {label_path}")

        gray = cv2.imread(str(source_image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"failed to decode image: {source_image}")
        if gray.dtype.name != "uint8" or len(gray.shape) != 2:
            raise RuntimeError(f"unexpected grayscale representation for {source_image}: {gray.shape} {gray.dtype}")

        output_image = output_images / source_image.name
        if not cv2.imwrite(str(output_image), gray, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
            raise RuntimeError(f"failed to write image: {output_image}")
        output_label = output_labels / label_path.name
        shutil.copy2(label_path, output_label)

        object_count, errors = validate_label(label_path)
        total_objects += object_count
        label_errors.extend(errors)
        manifest_rows.append(
            {
                "source_image": str(source_image),
                "gray_image": str(output_image),
                "source_label": str(label_path),
                "gray_label": str(output_label),
                "width": str(gray.shape[1]),
                "height": str(gray.shape[0]),
                "objects": str(object_count),
            }
        )

    if label_errors:
        raise ValueError("label validation failed:\n" + "\n".join(label_errors[:20]))

    list_path = output_root / "test.txt"
    list_path.write_text("".join(f"{row['gray_image']}\n" for row in manifest_rows), encoding="utf-8")
    yaml_path = write_yaml(output_root)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split": "test-only",
        "image_count": len(manifest_rows),
        "label_count": len(list(output_labels.glob("*.txt"))),
        "object_count": total_objects,
        "channels": 1,
        "dtype": "uint8",
        "class_mapping": {"0": "uav"},
        "jpeg_quality": args.jpeg_quality,
        "yaml": str(yaml_path),
        "manifest": str(manifest_path),
    }
    with (output_root / "build_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"source: {source_root}")
    print(f"output: {output_root}")
    print(f"images: {len(manifest_rows)}")
    print(f"labels: {len(list(output_labels.glob('*.txt')))}")
    print(f"objects: {total_objects}")
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
