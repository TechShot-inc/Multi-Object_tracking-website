#!/usr/bin/env python3
"""Create a Triton model repository from an ONNX model.

Creates the directory layout Triton expects:
  <repo>/<model_name>/1/model.onnx
  <repo>/<model_name>/config.pbtxt

Example:
  python scripts/make_triton_model_repo.py \
    --onnx artifacts/onnx/yolo11x.onnx \
    --model-name yolo11 \
    --repo-out models_repo \
    --max-batch-size 1

Notes:
- Requires `onnx` in the environment (included in mot-web[ml]).
- We generate a conservative config.pbtxt based on the graph I/O names.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to .onnx model")
    p.add_argument("--model-name", required=True, help="Triton model name (folder name)")
    p.add_argument("--repo-out", default="models_repo", help="Output Triton model repo directory")
    p.add_argument("--version", default="1", help="Version subdir under model (default: 1)")
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Triton max_batch_size (use 1 for [N,C,H,W] inputs; 0 for no-batch)",
    )
    p.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing model directory if it exists",
    )
    return p.parse_args()


def _onnx_tensor_shape(onnx_value_info) -> list[int] | None:
    try:
        dims = onnx_value_info.type.tensor_type.shape.dim
    except Exception:
        return None

    out: list[int] = []
    for d in dims:
        if d.dim_value is not None and int(d.dim_value) != 0:
            out.append(int(d.dim_value))
        else:
            # unknown / symbolic
            out.append(-1)
    return out


def _render_config(
    *,
    model_name: str,
    platform: str,
    max_batch_size: int,
    input_name: str,
    input_dims: list[int] | None,
    output_name: str,
    output_dims: list[int] | None,
) -> str:
    # Triton expects dims without batch when max_batch_size > 0.
    def strip_batch(dims: list[int] | None) -> list[int] | None:
        if dims is None:
            return None
        if max_batch_size > 0 and len(dims) >= 1:
            return dims[1:]
        return dims

    in_dims = strip_batch(input_dims) or [3, 640, 640]
    out_dims = strip_batch(output_dims) or [-1, -1]

    in_dims_str = ", ".join(str(d) for d in in_dims)
    out_dims_str = ", ".join(str(d) for d in out_dims)

    input_format_line = ""
    if len(in_dims) == 3:
        input_format_line = "    format: FORMAT_NCHW\n"

    return (
        f"name: \"{model_name}\"\n"
        f"platform: \"{platform}\"\n"
        f"max_batch_size: {max_batch_size}\n"
        f"input [\n"
        f"  {{\n"
        f"    name: \"{input_name}\"\n"
        f"    data_type: TYPE_FP32\n"
        f"{input_format_line}"
        f"    dims: [ {in_dims_str} ]\n"
        f"  }}\n"
        f"]\n"
        f"output [\n"
        f"  {{\n"
        f"    name: \"{output_name}\"\n"
        f"    data_type: TYPE_FP32\n"
        f"    dims: [ {out_dims_str} ]\n"
        f"  }}\n"
        f"]\n"
    )


def main() -> None:
    args = _parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"onnx not found: {onnx_path}")

    repo_out = Path(args.repo_out)
    model_dir = repo_out / args.model_name
    version_dir = model_dir / str(args.version)

    if model_dir.exists():
        if not args.force:
            raise SystemExit(f"model directory exists (use --force): {model_dir}")
        shutil.rmtree(model_dir)

    version_dir.mkdir(parents=True, exist_ok=True)

    # Load ONNX to extract IO names and shapes.
    import onnx  # type: ignore

    m = onnx.load(str(onnx_path))
    if len(m.graph.input) < 1 or len(m.graph.output) < 1:
        raise SystemExit("ONNX graph must have at least 1 input and 1 output")

    # Prefer first non-initializer input.
    initializer_names = {i.name for i in m.graph.initializer}
    inputs = [vi for vi in m.graph.input if vi.name not in initializer_names]
    if not inputs:
        inputs = list(m.graph.input)

    input_vi = inputs[0]
    output_vi = m.graph.output[0]

    input_name = input_vi.name
    output_name = output_vi.name

    input_dims = _onnx_tensor_shape(input_vi)
    output_dims = _onnx_tensor_shape(output_vi)

    config = _render_config(
        model_name=args.model_name,
        platform="onnxruntime_onnx",
        max_batch_size=int(args.max_batch_size),
        input_name=input_name,
        input_dims=input_dims,
        output_name=output_name,
        output_dims=output_dims,
    )

    (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")
    shutil.copy2(onnx_path, version_dir / "model.onnx")

    print(f"Wrote: {model_dir}")
    print(f"- config: {model_dir / 'config.pbtxt'}")
    print(f"- model:  {version_dir / 'model.onnx'}")
    print("\nNext:")
    print(f"  bash scripts/publish_triton_model_repo_oci.sh {repo_out}")


if __name__ == "__main__":
    main()
