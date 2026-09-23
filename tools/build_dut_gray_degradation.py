#!/usr/bin/env python3
"""Build fixed grayscale degradation datasets for DUT Anti-UAV style YOLO data.

The script is intentionally offline and deterministic: all degraded images are
materialized once, labels are inherited unchanged, and image lists/YAML files are
generated for Ultralytics training and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
TRAIN_VARIANTS = {
    "clean": {"suffix": "gray"},
    "low_light": {"suffix": "low_light", "brightness": 0.6},
    "low_contrast": {"suffix": "low_contrast", "contrast": 0.6},
    "noise": {"suffix": "noise", "noise": 10.0},
    "motion_blur": {"suffix": "motion_blur", "blur": 7},
    "mixed": {"suffix": "mixed", "brightness": 0.6, "contrast": 0.6, "noise": 5.0, "blur": 5},
}
TEST_VARIANTS = {
    "clean": {"suffix": "gray", "type": "clean", "level": ""},
    "low_light_L1": {"suffix": "low_light_L1", "type": "low_light", "level": "L1", "brightness": 0.7},
    "low_light_L2": {"suffix": "low_light_L2", "type": "low_light", "level": "L2", "brightness": 0.5},
    "low_light_L3": {"suffix": "low_light_L3", "type": "low_light", "level": "L3", "brightness": 0.35},
    "low_contrast_L1": {"suffix": "low_contrast_L1", "type": "low_contrast", "level": "L1", "contrast": 0.75},
    "low_contrast_L2": {"suffix": "low_contrast_L2", "type": "low_contrast", "level": "L2", "contrast": 0.5},
    "low_contrast_L3": {"suffix": "low_contrast_L3", "type": "low_contrast", "level": "L3", "contrast": 0.3},
    "noise_L1": {"suffix": "noise_L1", "type": "noise", "level": "L1", "noise": 5.0},
    "noise_L2": {"suffix": "noise_L2", "type": "noise", "level": "L2", "noise": 10.0},
    "noise_L3": {"suffix": "noise_L3", "type": "noise", "level": "L3", "noise": 20.0},
    "motion_blur_L1": {"suffix": "motion_blur_L1", "type": "motion_blur", "level": "L1", "blur": 5},
    "motion_blur_L2": {"suffix": "motion_blur_L2", "type": "motion_blur", "level": "L2", "blur": 9},
    "motion_blur_L3": {"suffix": "motion_blur_L3", "type": "motion_blur", "level": "L3", "blur": 13},
    "mixed_L1": {
        "suffix": "mixed_L1",
        "type": "mixed",
        "level": "L1",
        "brightness": 0.7,
        "contrast": 0.75,
        "noise": 5.0,
        "blur": 5,
    },
    "mixed_L2": {
        "suffix": "mixed_L2",
        "type": "mixed",
        "level": "L2",
        "brightness": 0.5,
        "contrast": 0.5,
        "noise": 10.0,
        "blur": 9,
    },
    "mixed_L3": {
        "suffix": "mixed_L3",
        "type": "mixed",
        "level": "L3",
        "brightness": 0.35,
        "contrast": 0.3,
        "noise": 20.0,
        "blur": 13,
    },
}


@dataclass(frozen=True)
class ImageItem:
    image: Path
    label: Optional[Path]
    rel_key: str


@dataclass
class DatasetInfo:
    root: Path
    splits: Dict[str, List[ImageItem]]
    format: str
    coco_jsons: Dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grayscale DART degradation datasets.")
    parser.add_argument("--src", required=True, help="Source dataset root or YOLO dataset yaml.")
    parser.add_argument("--output-root", required=True, help="Output directory, e.g. ./data/dart_dut.")
    parser.add_argument("--yaml-dir", default=None, help="Directory for generated yaml/txt files. Defaults to output root.")
    parser.add_argument("--dataset-name", default="DUT", help="Prefix used in output directory names.")
    parser.add_argument("--classes", default="uav", help="Comma-separated class names for generated YOLO yaml.")
    parser.add_argument("--image-ext", default=".jpg", help="Output image extension, default .jpg.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for reproducible noise.")
    parser.add_argument(
        "--motion-direction",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Motion blur direction. Default: horizontal.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated outputs.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_root_from_yaml(yaml_path: Path, data: dict) -> Path:
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_image_list(list_path: Path, root: Path) -> List[Path]:
    images: List[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            p = Path(text)
            if not p.is_absolute():
                p = root / p
            images.append(p)
    return images


def scan_images_dir(path: Path) -> List[Path]:
    if not path.exists():
        return []
    if path.is_file():
        raise ValueError(f"Expected directory but got file: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def split_from_yaml_entry(entry: str | Sequence[str], root: Path) -> List[Path]:
    entries = [entry] if isinstance(entry, str) else list(entry)
    images: List[Path] = []
    for item in entries:
        p = Path(str(item))
        if not p.is_absolute():
            p = root / p
        if p.suffix.lower() == ".txt":
            images.extend(load_image_list(p, root))
        elif p.is_dir():
            images.extend(scan_images_dir(p))
        else:
            raise FileNotFoundError(f"Cannot resolve split entry: {p}")
    return sorted(images)


def auto_detect_dataset(src: Path) -> DatasetInfo:
    coco_jsons: Dict[str, Path] = {}
    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        data = read_yaml(src)
        root = resolve_root_from_yaml(src, data)
        splits = {}
        for split in SPLITS:
            if split not in data or data[split] in (None, ""):
                splits[split] = []
                continue
            images = split_from_yaml_entry(data[split], root)
            splits[split] = [ImageItem(p, image_to_label_path(p), str(p.relative_to(root)) if p.is_relative_to(root) else p.name) for p in images]
        fmt = "yolo"
    else:
        root = src
        if not root.exists():
            raise FileNotFoundError(f"Source dataset does not exist: {root}")
        splits = {}
        for split in SPLITS:
            candidates = [root / "images" / split, root / split / "images", root / split]
            found = next((p for p in candidates if p.exists() and scan_images_dir(p)), None)
            if found is None:
                splits[split] = []
                continue
            images = scan_images_dir(found)
            splits[split] = [ImageItem(p, image_to_label_path(p), str(p.relative_to(root))) for p in images]
        fmt = "yolo"

    for split in SPLITS:
        for p in [
            root / "annotations" / f"instances_{split}.json",
            root / "cocoval" / f"{split}.json",
            root / "cocoval" / f"{split}_fix.json",
            root / split / "coco_annotations" / f"instances_{split}.json",
        ]:
            if p.exists():
                coco_jsons[split] = p
                break

    if not any(splits.values()):
        raise RuntimeError(
            "Could not auto-detect train/val/test images. Pass a YOLO yaml with train/val/test entries, "
            "or use a root containing images/train, images/val, images/test."
        )
    return DatasetInfo(root=root, splits=splits, format=fmt, coco_jsons=coco_jsons)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to regenerate.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(value)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        max_value = float(np.iinfo(gray.dtype).max) if np.issubdtype(gray.dtype, np.integer) else float(np.max(gray) or 1.0)
        gray = np.clip(gray.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
    return gray


def apply_brightness(gray: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(gray.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(gray: np.ndarray, factor: float) -> np.ndarray:
    # Fixed midpoint keeps the operation deterministic and independent of image content ordering.
    return np.clip((gray.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)


def apply_noise(gray: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, gray.shape).astype(np.float32)
    return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(gray: np.ndarray, kernel_size: int, direction: str) -> np.ndarray:
    k = int(kernel_size)
    if k <= 1:
        return gray
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    if direction == "vertical":
        kernel[:, k // 2] = 1.0 / k
    else:
        kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(gray, -1, kernel)


def degrade(gray: np.ndarray, params: dict, seed: int, key: str, direction: str) -> np.ndarray:
    out = gray.copy()
    if "brightness" in params:
        out = apply_brightness(out, float(params["brightness"]))
    if "contrast" in params:
        out = apply_contrast(out, float(params["contrast"]))
    if "noise" in params:
        out = apply_noise(out, float(params["noise"]), stable_rng(seed, key))
    if "blur" in params:
        out = apply_motion_blur(out, int(params["blur"]), direction)
    return out


def output_name(src: Path, suffix: str, ext: str) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{src.stem}_{suffix}{ext.lower()}"


def copy_label(label: Optional[Path], dst_label: Path) -> None:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if label and label.exists():
        shutil.copy2(label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def write_image_and_label(
    src_item: ImageItem,
    image: np.ndarray,
    out_dir: Path,
    suffix: str,
    image_ext: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / output_name(src_item.image, suffix, image_ext)
    label_path = image_path.with_suffix(".txt")
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    copy_label(src_item.label, label_path)
    return image_path


def write_list(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def update_coco_json(src_json: Path, dst_json: Path, variant_suffix: str, image_ext: str) -> None:
    with src_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for image in data.get("images", []):
        file_name = Path(image.get("file_name", ""))
        image["file_name"] = output_name(file_name, variant_suffix, image_ext)
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def build(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser()
    output_root = Path(args.output_root).expanduser().resolve()
    yaml_dir = Path(args.yaml_dir).expanduser().resolve() if args.yaml_dir else output_root
    class_names = [x.strip() for x in args.classes.split(",") if x.strip()]
    if not class_names:
        raise ValueError("--classes must contain at least one class name.")

    info = auto_detect_dataset(src)
    gray_root = output_root / f"{args.dataset_name}-Gray"
    deg_train_root = output_root / f"{args.dataset_name}-Gray-DegTrain"
    deg_test_root = output_root / f"{args.dataset_name}-Gray-Degraded-Test"

    for path in (gray_root, deg_train_root, deg_test_root):
        prepare_output_dir(path, args.overwrite)
    yaml_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "original": {split: len(info.splits.get(split, [])) for split in SPLITS},
        "gray": {split: 0 for split in SPLITS},
        "deg_train": {name: 0 for name in TRAIN_VARIANTS},
        "deg_test": {name: 0 for name in TEST_VARIANTS},
    }
    generated_yaml: List[Path] = []
    generated_json: List[Path] = []
    split_clean_images: Dict[str, List[Path]] = {split: [] for split in SPLITS}
    train_variant_images: Dict[str, List[Path]] = {name: [] for name in TRAIN_VARIANTS}
    test_variant_images: Dict[str, List[Path]] = {name: [] for name in TEST_VARIANTS}

    for split, items in info.splits.items():
        split_dir = gray_root / split
        for item in items:
            gray = read_gray(item.image)
            out_img = write_image_and_label(item, gray, split_dir, "gray", args.image_ext)
            split_clean_images[split].append(out_img)
            counts["gray"][split] += 1

        if split in info.coco_jsons:
            dst_json = gray_root / "annotations" / f"instances_{split}_gray.json"
            update_coco_json(info.coco_jsons[split], dst_json, "gray", args.image_ext)
            generated_json.append(dst_json)

    for item in info.splits.get("train", []):
        gray = read_gray(item.image)
        for name, params in TRAIN_VARIANTS.items():
            out_dir = deg_train_root / "train" / name
            image = degrade(gray, params, args.seed, f"train:{item.rel_key}:{name}", args.motion_direction)
            out_img = write_image_and_label(item, image, out_dir, str(params["suffix"]), args.image_ext)
            train_variant_images[name].append(out_img)
            counts["deg_train"][name] += 1

    for item in info.splits.get("test", []):
        gray = read_gray(item.image)
        for name, params in TEST_VARIANTS.items():
            out_dir = deg_test_root / name
            image = degrade(gray, params, args.seed, f"test:{item.rel_key}:{name}", args.motion_direction)
            out_img = write_image_and_label(item, image, out_dir, str(params["suffix"]), args.image_ext)
            test_variant_images[name].append(out_img)
            counts["deg_test"][name] += 1

    train_repeat = yaml_dir / "train_repeat.txt"
    write_list(train_repeat, split_clean_images["train"] * len(TRAIN_VARIANTS))
    dart_train = yaml_dir / "train_dart.txt"
    write_list(dart_train, [p for name in TRAIN_VARIANTS for p in train_variant_images[name]])
    clean_train = yaml_dir / "train_clean.txt"
    clean_val = yaml_dir / "val_clean.txt"
    clean_test = yaml_dir / "test_clean.txt"
    write_list(clean_train, split_clean_images["train"])
    write_list(clean_val, split_clean_images["val"])
    write_list(clean_test, split_clean_images["test"])

    base_yaml = {"nc": len(class_names), "names": class_names, "ch": 1, "channels": 1}
    yaml_specs = {
        "dut_gray_clean.yaml": {**base_yaml, "train": str(clean_train), "val": str(clean_val), "test": str(clean_test)},
        "dut_gray_clean_repeat.yaml": {**base_yaml, "train": str(train_repeat), "val": str(clean_val), "test": str(clean_test)},
        "dut_gray_dart.yaml": {**base_yaml, "train": str(dart_train), "val": str(clean_val), "test": str(clean_test)},
    }
    for name, spec in yaml_specs.items():
        path = yaml_dir / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"YAML exists: {path}. Use --overwrite.")
        write_yaml(path, spec)
        generated_yaml.append(path)

    for name, images in test_variant_images.items():
        list_path = yaml_dir / f"test_{name}.txt"
        write_list(list_path, images)
        yaml_name = f"dut_gray_test_{name}.yaml"
        yaml_path = yaml_dir / yaml_name
        write_yaml(yaml_path, {**base_yaml, "train": str(clean_train), "val": str(list_path), "test": str(list_path)})
        generated_yaml.append(yaml_path)

    if "train" in info.coco_jsons:
        for name, params in TRAIN_VARIANTS.items():
            dst = deg_train_root / "annotations" / f"instances_train_{name}.json"
            update_coco_json(info.coco_jsons["train"], dst, str(params["suffix"]), args.image_ext)
            generated_json.append(dst)
    if "test" in info.coco_jsons:
        for name, params in TEST_VARIANTS.items():
            dst = deg_test_root / "annotations" / f"instances_test_{name}.json"
            update_coco_json(info.coco_jsons["test"], dst, str(params["suffix"]), args.image_ext)
            generated_json.append(dst)

    print("\nBuild complete.")
    print(f"Source root: {info.root}")
    print(f"Output root: {output_root}")
    print(f"YAML dir:    {yaml_dir}")
    print("\nOriginal image counts:")
    for split, count in counts["original"].items():
        print(f"  {split}: {count}")
    print("\nGray clean image counts:")
    for split, count in counts["gray"].items():
        print(f"  {split}: {count}")
    print("\nDegraded train image counts:")
    for name, count in counts["deg_train"].items():
        print(f"  {name}: {count}")
    print("\nDegraded test image counts:")
    for name, count in counts["deg_test"].items():
        print(f"  {name}: {count}")
    print("\nGenerated YAML/list paths:")
    for path in generated_yaml + [train_repeat, dart_train, clean_train, clean_val, clean_test]:
        print(f"  {path}")
    if generated_json:
        print("\nGenerated COCO json paths:")
        for path in generated_json:
            print(f"  {path}")


if __name__ == "__main__":
    build(parse_args())
