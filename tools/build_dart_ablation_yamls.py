#!/usr/bin/env python3
"""Build training lists and data YAMLs for DART degradation ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml


CONDITIONS = ["low_light", "low_contrast", "noise", "motion_blur", "mixed"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean+single-degradation ablation YAMLs.")
    parser.add_argument("--data-root", type=Path, required=True, help="Generated DART root, e.g. data/dart_dut.")
    parser.add_argument("--yaml-dir", type=Path, required=True, help="Directory for generated train lists and YAMLs.")
    parser.add_argument("--dataset-name", default="DUT", help="Prefix used in generated YAML names.")
    parser.add_argument("--classes", nargs="+", default=["uav"], help="Class names.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    files = sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise RuntimeError(f"No images found in {directory}")
    return files


def write_lines(path: Path, lines: Iterable[Path], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to regenerate.")
    items = [str(p) for p in lines]
    path.write_text("\n".join(items) + "\n", encoding="utf-8")
    return len(items)


def write_yaml(path: Path, content: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to regenerate.")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)


def yaml_name(dataset_name: str, condition: str) -> str:
    prefix = dataset_name.lower().replace("-", "_")
    return f"{prefix}_gray_ablate_{condition}.yaml"


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    yaml_dir = args.yaml_dir.resolve()
    yaml_dir.mkdir(parents=True, exist_ok=True)

    train_root = data_root / "DUT-Gray-DegTrain" / "train"
    clean_files = image_files(train_root / "clean")
    val_list = yaml_dir / "val_clean.txt"
    test_list = yaml_dir / "test_clean.txt"
    if not val_list.exists():
        raise FileNotFoundError(f"Missing val list: {val_list}")
    if not test_list.exists():
        raise FileNotFoundError(f"Missing test list: {test_list}")

    print(f"data_root: {data_root}")
    print(f"yaml_dir:  {yaml_dir}")
    print(f"clean:     {len(clean_files)}")

    generated = []
    for condition in CONDITIONS:
        condition_files = image_files(train_root / condition)
        train_list = yaml_dir / f"train_ablate_clean_{condition}.txt"
        count = write_lines(train_list, [*clean_files, *condition_files], args.overwrite)
        data_yaml = yaml_dir / yaml_name(args.dataset_name, condition)
        write_yaml(
            data_yaml,
            {
                "nc": len(args.classes),
                "names": args.classes,
                "ch": 1,
                "channels": 1,
                "train": str(train_list),
                "val": str(val_list),
                "test": str(test_list),
            },
            args.overwrite,
        )
        generated.append((condition, count, train_list, data_yaml))

    print("\nGenerated ablation configs:")
    for condition, count, train_list, data_yaml in generated:
        print(f"- clean + {condition}: {count} images")
        print(f"  train list: {train_list}")
        print(f"  yaml:       {data_yaml}")


if __name__ == "__main__":
    main()
