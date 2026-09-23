#!/usr/bin/env python3
"""Build a stronger synthetic unseen degradation stress-test suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    params: dict[str, float | int]


CONDITIONS = [
    Condition("jpeg_compression_L4", "jpeg_compression", {"quality": 10}),
    Condition("gaussian_blur_L4", "gaussian_blur", {"kernel": 11}),
    Condition("defocus_blur_L4", "defocus_blur", {"radius": 9}),
    Condition("salt_pepper_noise_L4", "salt_pepper_noise", {"prob": 0.05}),
    Condition("diagonal_motion_blur_L4", "diagonal_motion_blur", {"kernel": 17, "angle": 45}),
    Condition("gamma_darkening_L4", "gamma_darkening", {"gamma": 3.0}),
    Condition(
        "mixed_unseen_L4",
        "mixed_unseen",
        {"quality": 15, "gamma": 2.6, "gaussian_kernel": 7, "salt_pepper_prob": 0.02},
    ),
    Condition("compound_blur_noise_L4", "compound_blur_noise", {"gaussian_kernel": 9, "salt_pepper_prob": 0.03}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stronger synthetic unseen degradation test YAMLs.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/dart_dut")
    parser.add_argument("--yaml-dir", type=Path, default=ROOT / "data/dart_dut/yamls_strong_unseen")
    parser.add_argument("--dataset-name", default="DUT")
    parser.add_argument("--classes", default="uav")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)
    return img


def scan_images(path: Path) -> list[Path]:
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {path}")
    return images


def write_list(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def output_name(src: Path, condition: str, ext: str) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{src.stem}_{condition}{ext.lower()}"


def copy_label(src_image: Path, dst_image: Path) -> None:
    src_label = src_image.with_suffix(".txt")
    dst_label = dst_image.with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def jpeg(gray: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    out = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if out is None:
        raise RuntimeError("JPEG decode failed")
    return out


def gaussian(gray: np.ndarray, kernel: int) -> np.ndarray:
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def disk_kernel(radius: int) -> np.ndarray:
    r = int(radius)
    size = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= r * r
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= float(kernel.sum())
    return kernel


def defocus(gray: np.ndarray, radius: int) -> np.ndarray:
    return cv2.filter2D(gray, -1, disk_kernel(radius))


def salt_pepper(gray: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    out = gray.copy()
    mask = rng.random(gray.shape)
    half = float(prob) / 2.0
    out[mask < half] = 0
    out[(mask >= half) & (mask < float(prob))] = 255
    return out


def motion_kernel(kernel_size: int, angle: float) -> np.ndarray:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    matrix = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def motion(gray: np.ndarray, kernel: int, angle: float) -> np.ndarray:
    return cv2.filter2D(gray, -1, motion_kernel(kernel, angle))


def gamma_dark(gray: np.ndarray, gamma: float) -> np.ndarray:
    norm = gray.astype(np.float32) / 255.0
    return np.clip(np.power(norm, float(gamma)) * 255.0, 0, 255).astype(np.uint8)


def degrade(gray: np.ndarray, condition: Condition, rng: np.random.Generator) -> np.ndarray:
    p = condition.params
    if condition.family == "jpeg_compression":
        return jpeg(gray, int(p["quality"]))
    if condition.family == "gaussian_blur":
        return gaussian(gray, int(p["kernel"]))
    if condition.family == "defocus_blur":
        return defocus(gray, int(p["radius"]))
    if condition.family == "salt_pepper_noise":
        return salt_pepper(gray, float(p["prob"]), rng)
    if condition.family == "diagonal_motion_blur":
        return motion(gray, int(p["kernel"]), float(p["angle"]))
    if condition.family == "gamma_darkening":
        return gamma_dark(gray, float(p["gamma"]))
    if condition.family == "mixed_unseen":
        out = gamma_dark(gray, float(p["gamma"]))
        out = gaussian(out, int(p["gaussian_kernel"]))
        out = salt_pepper(out, float(p["salt_pepper_prob"]), rng)
        return jpeg(out, int(p["quality"]))
    if condition.family == "compound_blur_noise":
        out = gaussian(gray, int(p["gaussian_kernel"]))
        return salt_pepper(out, float(p["salt_pepper_prob"]), rng)
    raise KeyError(condition.family)


def selected_conditions(text: str) -> list[Condition]:
    if text == "all":
        return list(CONDITIONS)
    wanted = {x.strip() for x in text.split(",") if x.strip()}
    by_name = {c.name: c for c in CONDITIONS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"Unknown conditions: {', '.join(missing)}")
    return [c for c in CONDITIONS if c.name in wanted]


def update_coco_json(src_json: Path, dst_json: Path, condition: str, image_ext: str) -> None:
    if not src_json.exists():
        return
    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for image in data.get("images", []):
        image["file_name"] = output_name(Path(image.get("file_name", "")), condition, image_ext)
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to regenerate.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    yaml_dir = args.yaml_dir.resolve()
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    if not classes:
        raise ValueError("--classes is empty")

    clean_train = data_root / "yamls/train_clean.txt"
    clean_test = data_root / "yamls/test_clean.txt"
    if not clean_train.exists() or not clean_test.exists():
        raise FileNotFoundError("Missing clean train/test list files under data_root/yamls")

    test_dir = data_root / f"{args.dataset_name}-Gray/test"
    output_root = data_root / f"{args.dataset_name}-Gray-Unseen-Strong-Test"
    annotation_root = output_root / "annotations"
    clean_coco = data_root / f"{args.dataset_name}-Gray-Degraded-Test/annotations/instances_test_clean.json"
    images = scan_images(test_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    base_yaml = {"nc": len(classes), "names": classes, "ch": 1, "channels": 1}
    rows: list[dict[str, str]] = []

    for condition in selected_conditions(args.conditions):
        out_dir = output_root / condition.name
        prepare_dir(out_dir, args.overwrite)
        output_images: list[Path] = []
        for image in images:
            gray = read_gray(image)
            rng = stable_rng(args.seed, f"{condition.name}:{image.name}")
            out = degrade(gray, condition, rng)
            out_image = out_dir / output_name(image, condition.name, args.image_ext)
            if not cv2.imwrite(str(out_image), out):
                raise RuntimeError(f"Failed to write {out_image}")
            copy_label(image, out_image)
            output_images.append(out_image)

        list_path = yaml_dir / f"test_{condition.name}.txt"
        yaml_path = yaml_dir / f"dut_gray_test_{condition.name}.yaml"
        if (list_path.exists() or yaml_path.exists()) and not args.overwrite:
            raise FileExistsError(f"{condition.name} YAML/list exists; use --overwrite to regenerate.")
        write_list(list_path, output_images)
        write_yaml(yaml_path, {**base_yaml, "train": str(clean_train), "val": str(list_path), "test": str(list_path)})
        update_coco_json(clean_coco, annotation_root / f"instances_test_{condition.name}.json", condition.name, args.image_ext)
        rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "params": json.dumps(condition.params, sort_keys=True),
                "images": str(len(output_images)),
                "yaml": str(yaml_path),
            }
        )

    summary = yaml_dir / "strong_unseen_manifest.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "family", "params", "images", "yaml"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Strong unseen output root: {output_root}")
    print(f"Strong unseen YAML dir: {yaml_dir}")
    print(f"Manifest: {summary}")
    for row in rows:
        print(f"{row['condition']}: {row['images']} images")


if __name__ == "__main__":
    main()
