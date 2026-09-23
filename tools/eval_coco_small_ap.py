#!/usr/bin/env python3
"""Compute COCO small-object AP for a YOLO model on one image list."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def chunks(items: List[Path], size: int) -> Iterable[List[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute COCO AP metrics, including AP-S, for one condition.")
    parser.add_argument("--weights", required=True, help="Model weights, e.g. runs/.../weights/best.pt.")
    parser.add_argument("--images", required=True, help="Text file containing absolute image paths, or an image directory.")
    parser.add_argument("--gt-json", required=True, help="COCO ground-truth JSON for the same condition.")
    parser.add_argument("--repo", default="ultralytics_repo", help="Modified Ultralytics repo path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold for AP evaluation.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold used during prediction.")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests.")
    parser.add_argument("--pred-json", required=True, help="Output COCO detection JSON.")
    parser.add_argument("--output-csv", default=None, help="Optional one-row metrics CSV.")
    return parser.parse_args()


def read_images(path: Path) -> List[Path]:
    if path.is_dir():
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        with path.open("r", encoding="utf-8") as f:
            images = [Path(line.strip()) for line in f if line.strip()]
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def load_image_ids(gt_json: Path) -> Dict[str, str | int]:
    with gt_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_ids: Dict[str, str | int] = {}
    for image in data.get("images", []):
        file_name = Path(image["file_name"]).name
        image_ids[file_name] = image["id"]
    if not image_ids:
        raise ValueError(f"No image entries found in {gt_json}")
    return image_ids


def write_predictions(
    *,
    weights: Path,
    image_paths: List[Path],
    image_ids: Dict[str, str | int],
    repo: Path,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    output_json: Path,
) -> int:
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO  # noqa: WPS433

    model = YOLO(str(weights))
    detections: List[dict] = []
    effective_batch = max(1, batch)
    for chunk in chunks(image_paths, effective_batch):
        stream = model.predict(
            source=[str(p) for p in chunk],
            stream=True,
            imgsz=imgsz,
            batch=len(chunk),
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        for result in stream:
            file_name = Path(result.path).name
            if file_name not in image_ids:
                raise KeyError(f"Prediction image not found in COCO gt images: {file_name}")
            image_id = image_ids[file_name]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = [float(v) for v in box]
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(detections, f)
    return len(detections)


def evaluate_coco(gt_json: Path, pred_json: Path, max_det: int, img_ids: List[str | int]) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with pred_json.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP-S": 0.0, "AP-M": 0.0, "AP-L": 0.0}

    coco_gt = COCO(str(gt_json))
    # Generated local COCO files only keep images/annotations/categories.
    # pycocotools.loadRes expects these optional metadata keys to exist.
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_dt = coco_gt.loadRes(str(pred_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = img_ids
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP-S": float(stats[3]),
        "AP-M": float(stats[4]),
        "AP-L": float(stats[5]),
    }


def write_csv(path: Path, metrics: Dict[str, float], pred_count: int, image_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["images", "detections", *metrics.keys()])
        writer.writeheader()
        writer.writerow(
            {
                "images": image_count,
                "detections": pred_count,
                **{k: f"{v:.6f}" for k, v in metrics.items()},
            }
        )


def main() -> None:
    args = parse_args()
    images = read_images(Path(args.images).resolve())
    if args.limit > 0:
        images = images[: args.limit]
    image_ids = load_image_ids(Path(args.gt_json).resolve())
    pred_count = write_predictions(
        weights=Path(args.weights).resolve(),
        image_paths=images,
        image_ids=image_ids,
        repo=Path(args.repo).resolve(),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        output_json=Path(args.pred_json).resolve(),
    )
    selected_img_ids = [image_ids[Path(path).name] for path in images]
    metrics = evaluate_coco(Path(args.gt_json).resolve(), Path(args.pred_json).resolve(), args.max_det, selected_img_ids)
    print(f"Images: {len(images)}")
    print(f"Detections: {pred_count}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    if args.output_csv:
        write_csv(Path(args.output_csv).resolve(), metrics, pred_count, len(images))
        print(f"CSV: {Path(args.output_csv).resolve()}")
    print(f"Pred JSON: {Path(args.pred_json).resolve()}")


if __name__ == "__main__":
    main()
