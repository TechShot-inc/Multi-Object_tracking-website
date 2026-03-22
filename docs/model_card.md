# Model card (tracking demo)

This repository uses an ensemble detector + tracker pipeline for multi-object tracking:

- Detector: YOLO11 + YOLO12 (two-model ensemble)
- Tracker: BoostTrack (via `CustomBoostTrack/`)
- Optional ReID: OSNet-AIN (used for embedding / association in some modes)

## Intended use

- Demonstrations and prototyping of multi-object tracking (MOT) on videos and realtime webcam streams.
- Measuring end-to-end performance (FPS/latency) under Dockerized, reproducible setups.

## Not intended for

- Safety-critical use (surveillance, law enforcement, high-stakes decisions).
- Determining identity or sensitive attributes.

## Data & privacy notes

- The realtime UI streams frames from the browser to the backend for inference and returns an annotated preview.
- Avoid running this on untrusted networks or with untrusted code on a self-hosted runner.
- If you handle real people footage, ensure you have appropriate consent and comply with local privacy laws.

## Limitations

- Detector accuracy depends heavily on lighting, motion blur, and camera angle.
- The ensemble and tracker parameters are tuned for speed and demo stability, not for benchmark leaderboard performance.

## Licensing

- Model weights have their own licenses and terms.
- This repo includes tooling to download weights and to package Triton model repositories; you are responsible for ensuring your usage complies with upstream licenses.

## Reproducibility

- Published Triton model bundles include:
  - `model_repo_manifest.json` (file list + sha256)
  - `model_repo.SHA256SUMS` (sha256sum-compatible)
  - `triton-models.tar.gz.sha256` (bundle checksum)

See the publishing workflow and scripts for details.
