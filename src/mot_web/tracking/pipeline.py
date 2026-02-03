from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run_video(input_path: Path, output_dir: Path, params: dict[str, Any]) -> None:
    """
    Stub pipeline: creates an output artifact so the web flow is end-to-end.
    Replace internals later with CustomBoostTrack adapter call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate work (optional: remove if you don't want delay)
    time.sleep(0.1)

    # Minimal output artifact that your UI can later evolve to consume
    annotations = {
        "source_video": str(input_path),
        "created_at_unix": int(time.time()),
        "params": params,
        "tracks": [],  # later: list of {id, bboxes, timestamps, ...}
    }

    (output_dir / "annotations.json").write_text(
        json.dumps(annotations, indent=2),
        encoding="utf-8",
    )
