#!/usr/bin/env python3
"""Export Ultralytics YOLO .pt weights to ONNX.

This script is meant to produce a Triton-friendly ONNX model artifact.

Example:
  python scripts/export_yolo_pt_to_onnx.py --weights models/yolo11x.pt --imgsz 640 --out artifacts/onnx/yolo11.onnx

Notes:
- Requires the "ml" extra (ultralytics + torch).
- By default we export WITHOUT integrated NMS (raw outputs). TritonYoloDetector has
  a raw-output parser that handles common Ultralytics export layouts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--imgsz", type=int, default=640, help="Square input size")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export dynamic shapes (default: false for Triton simplicity)",
    )
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try to simplify ONNX (requires onnxsim in environment)",
    )
    p.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export with NMS integrated (layout varies; default: false)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .onnx path (default: artifacts/onnx/<weights_stem>.onnx)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    out_path = Path(args.out) if args.out else Path("artifacts/onnx") / f"{weights_path.stem}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a writable work dir regardless of where the weights live (they may be on a read-only mount).
    project_dir = out_path.parent / "ultralytics_export"
    project_dir.mkdir(parents=True, exist_ok=True)

    work_weights = project_dir / weights_path.name
    if work_weights != weights_path:
        shutil.copy2(weights_path, work_weights)

    # Lazy import so this script is harmless in non-ML envs.
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(work_weights))

    # Ultralytics validates args, so keep overrides conservative.
    try:
        exported = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            nms=args.nms,
            half=False,
            int8=False,
            device="cpu",
            project=str(project_dir),
            name=weights_path.stem,
            exist_ok=True,
        )
    except SyntaxError:
        exported = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            nms=args.nms,
            half=False,
            int8=False,
            device="cpu",
        )

    exported_path = Path(str(exported))
    if exported_path.is_dir():
        candidates = list(exported_path.rglob("*.onnx"))
        if not candidates:
            raise SystemExit(f"Export did not produce an .onnx under: {exported_path}")
        exported_path = candidates[0]
    if not exported_path.exists():
        # Fallback: scan the project dir.
        candidates = list(project_dir.rglob("*.onnx"))
        if not candidates:
            raise SystemExit(f"Export did not produce an .onnx (looked in {project_dir})")
        exported_path = candidates[0]

    shutil.copy2(exported_path, out_path)
    print(f"Export complete: {out_path}")


if __name__ == "__main__":
    main()
