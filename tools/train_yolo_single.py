#!/usr/bin/env python3
"""Train one YOLO model with the local modified Ultralytics repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single YOLO model run.")
    parser.add_argument("--repo", default="ultralytics_repo", help="Modified YOLO/Ultralytics repo path.")
    parser.add_argument("--model", required=True, help="Model YAML or checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--project", required=True, help="Ultralytics project output directory.")
    parser.add_argument("--name", required=True, help="Ultralytics run name.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Default is disabled for reproducibility.")
    parser.add_argument("--nwdloss", action="store_true", help="Enable project-specific NWD loss.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume training from --model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repo not found: {repo}")
    sys.path.insert(0, str(repo))

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
        return

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        amp=args.amp,
        project=args.project,
        name=args.name,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        nwdloss=args.nwdloss,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
