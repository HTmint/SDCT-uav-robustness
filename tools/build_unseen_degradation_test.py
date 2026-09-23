#!/usr/bin/env python3
"""Build deterministic unseen degradation test suites from DUT-Gray/test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageItem:
    image: Path
    label: Path


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    level: str
    params: Dict[str, float | int]


CONDITIONS = [
    Condition("jpeg_compression_L1", "jpeg_compression", "L1", {"quality": 70}),
    Condition("jpeg_compression_L2", "jpeg_compression", "L2", {"quality": 45}),
    Condition("jpeg_compression_L3", "jpeg_compression", "L3", {"quality": 25}),
    Condition("gaussian_blur_L1", "gaussian_blur", "L1", {"kernel": 3}),
    Condition("gaussian_blur_L2", "gaussian_blur", "L2", {"kernel": 5}),
    Condition("gaussian_blur_L3", "gaussian_blur", "L3", {"kernel": 7}),
    Condition("defocus_blur_L1", "defocus_blur", "L1", {"radius": 2}),
    Condition("defocus_blur_L2", "defocus_blur", "L2", {"radius": 4}),
    Condition("defocus_blur_L3", "defocus_blur", "L3", {"radius": 6}),
    Condition("salt_pepper_noise_L1", "salt_pepper_noise", "L1", {"prob": 0.005}),
    Condition("salt_pepper_noise_L2", "salt_pepper_noise", "L2", {"prob": 0.010}),
    Condition("salt_pepper_noise_L3", "salt_pepper_noise", "L3", {"prob": 0.020}),
    Condition("diagonal_motion_blur_L1", "diagonal_motion_blur", "L1", {"kernel": 5, "angle": 45}),
    Condition("diagonal_motion_blur_L2", "diagonal_motion_blur", "L2", {"kernel": 9, "angle": 45}),
    Condition("diagonal_motion_blur_L3", "diagonal_motion_blur", "L3", {"kernel": 13, "angle": 45}),
    Condition("gamma_darkening_L1", "gamma_darkening", "L1", {"gamma": 1.3}),
    Condition("gamma_darkening_L2", "gamma_darkening", "L2", {"gamma": 1.8}),
    Condition("gamma_darkening_L3", "gamma_darkening", "L3", {"gamma": 2.3}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic unseen degradation test YAMLs.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/dart_dut")
    parser.add_argument("--yaml-dir", type=Path, default=None, help="Defaults to DATA_ROOT/yamls.")
    parser.add_argument("--dataset-name", default="DUT")
    parser.add_argument("--classes", default="uav", help="Comma-separated class names.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated unseen conditions or 'all'. Defaults to all six families.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing unseen outputs.")
    return parser.parse_args()


def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(value)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        max_value = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else float(np.max(img) or 1.0)
        img = np.clip(img.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
    return img


def image_label_path(image: Path) -> Path:
    return image.with_suffix(".txt")


def scan_test_images(test_dir: Path) -> List[ImageItem]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing clean test directory: {test_dir}")
    images = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No test images found in {test_dir}")
    return [ImageItem(image=p, label=image_label_path(p)) for p in images]


def write_list(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def output_name(src: Path, suffix: str, ext: str) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{src.stem}_{suffix}{ext.lower()}"


def copy_label(src_label: Path, dst_label: Path) -> None:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def apply_jpeg(gray: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed.")
    return decoded


def apply_gaussian_blur(gray: np.ndarray, kernel: int) -> np.ndarray:
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


def apply_defocus_blur(gray: np.ndarray, radius: int) -> np.ndarray:
    return cv2.filter2D(gray, -1, disk_kernel(radius))


def apply_salt_pepper(gray: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
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


def apply_diagonal_motion(gray: np.ndarray, kernel: int, angle: float) -> np.ndarray:
    return cv2.filter2D(gray, -1, motion_kernel(kernel, angle))


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    norm = gray.astype(np.float32) / 255.0
    return np.clip(np.power(norm, float(gamma)) * 255.0, 0, 255).astype(np.uint8)


def degrade(gray: np.ndarray, condition: Condition, rng: np.random.Generator) -> np.ndarray:
    p = condition.params
    if condition.family == "jpeg_compression":
        return apply_jpeg(gray, int(p["quality"]))
    if condition.family == "gaussian_blur":
        return apply_gaussian_blur(gray, int(p["kernel"]))
    if condition.family == "defocus_blur":
        return apply_defocus_blur(gray, int(p["radius"]))
    if condition.family == "salt_pepper_noise":
        return apply_salt_pepper(gray, float(p["prob"]), rng)
    if condition.family == "diagonal_motion_blur":
        return apply_diagonal_motion(gray, int(p["kernel"]), float(p["angle"]))
    if condition.family == "gamma_darkening":
        return apply_gamma(gray, float(p["gamma"]))
    raise KeyError(condition.family)


def selected_conditions(text: str) -> List[Condition]:
    if text == "all":
        return list(CONDITIONS)
    wanted = {x.strip() for x in text.split(",") if x.strip()}
    by_name = {condition.name: condition for condition in CONDITIONS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"Unknown conditions: {', '.join(missing)}")
    return [condition for condition in CONDITIONS if condition.name in wanted]


def maybe_prepare_condition_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to regenerate.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def update_coco_json(src_json: Path, dst_json: Path, condition: str, image_ext: str) -> Optional[Path]:
    if not src_json.exists():
        return None
    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for image in data.get("images", []):
        image["file_name"] = output_name(Path(image.get("file_name", "")), condition, image_ext)
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return dst_json


def count_label_files(paths: Iterable[Path]) -> int:
    return sum(1 for path in paths if path.with_suffix(".txt").exists())


def write_summary(rows: List[Dict[str, str]]) -> None:
    headers = ["Condition", "Num images", "Label count", "Output yaml"]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row[h] for h in headers) + " |")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    yaml_dir = (args.yaml_dir or data_root / "yamls").resolve()
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class.")

    clean_train = yaml_dir / "train_clean.txt"
    clean_test = yaml_dir / "test_clean.txt"
    if not clean_train.exists() or not clean_test.exists():
        raise FileNotFoundError(f"Missing clean list files in {yaml_dir}")

    test_dir = data_root / f"{args.dataset_name}-Gray" / "test"
    output_root = data_root / f"{args.dataset_name}-Gray-Unseen-Degraded-Test"
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_root = output_root / "annotations"
    clean_coco = data_root / f"{args.dataset_name}-Gray-Degraded-Test" / "annotations" / "instances_test_clean.json"

    items = scan_test_images(test_dir)
    base_yaml = {"nc": len(classes), "names": classes, "ch": 1, "channels": 1}
    rows: List[Dict[str, str]] = []

    for condition in selected_conditions(args.conditions):
        out_dir = output_root / condition.name
        maybe_prepare_condition_dir(out_dir, args.overwrite)

        output_images: List[Path] = []
        for item in items:
            gray = read_gray(item.image)
            rng = stable_rng(args.seed, f"{condition.name}:{item.image.name}")
            degraded = degrade(gray, condition, rng)
            out_image = out_dir / output_name(item.image, condition.name, args.image_ext)
            ok = cv2.imwrite(str(out_image), degraded)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_image}")
            copy_label(item.label, out_image.with_suffix(".txt"))
            output_images.append(out_image)

        list_path = yaml_dir / f"test_{condition.name}.txt"
        yaml_path = yaml_dir / f"dut_gray_test_{condition.name}.yaml"
        if (list_path.exists() or yaml_path.exists()) and not args.overwrite:
            raise FileExistsError(f"YAML/list already exists for {condition.name}. Use --overwrite.")
        write_list(list_path, output_images)
        write_yaml(yaml_path, {**base_yaml, "train": str(clean_train), "val": str(list_path), "test": str(list_path)})
        update_coco_json(clean_coco, annotation_root / f"instances_test_{condition.name}.json", condition.name, args.image_ext)

        rows.append(
            {
                "Condition": condition.name,
                "Num images": str(len(output_images)),
                "Label count": str(count_label_files(output_images)),
                "Output yaml": str(yaml_path),
            }
        )

    write_summary(rows)
    print(f"\nUnseen output root: {output_root}")
    print(f"YAML dir: {yaml_dir}")


if __name__ == "__main__":
    main()
