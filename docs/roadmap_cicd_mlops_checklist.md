# CI/CD + DevOps + MLOps Roadmap (Docker-Only)

This checklist is designed for a workflow where **everything runs inside Docker containers**, including tests, lint, typechecks, benchmarks, and perf smoke tests.

Conventions
- Use `docker compose -f docker/compose.yml ...` for CPU-only workflows.
- Use `docker compose -f docker/compose.yml -f docker/compose.gpu.yml --profile triton ...` for Triton+GPU workflows.
- Prefer `docker compose run --rm <service> <cmd...>` for all tooling.

---

## Phase 1 — Baseline CI inside Docker (pytest + ruff + pyright)

Goal: Fast PR feedback, reproducible tooling, no "it works on my machine".

Checklist
- [x] Add `ruff` + `pyright` to dev dependencies in [pyproject.toml](../pyproject.toml)
- [x] Add ruff config in [pyproject.toml](../pyproject.toml) (minimal rule set to start)
- [x] Add pyright config in [pyrightconfig.json](../pyrightconfig.json)
- [x] Update lockfile (`uv.lock`) using Docker
- [x] Rebuild the `mot-tests` image
- [x] Run (inside Docker): format check, lint, typecheck, tests
- [x] Add GitHub Actions that run **the same Docker commands**

Docker commands (local verification)
- Update lock:
  - `docker run --rm -v "$PWD:/repo" -w /repo python:3.11-slim bash -lc "python -m pip install -U pip uv && uv lock"`
- Build tests image:
  - `docker compose -f docker/compose.yml build mot-tests`
- Run ruff:
  - `docker compose -f docker/compose.yml run --rm mot-tests ruff format --check src tests`
  - `docker compose -f docker/compose.yml run --rm mot-tests ruff check src tests`
- Run pyright:
  - `docker compose -f docker/compose.yml run --rm mot-tests pyright`
- Run pytest:
  - `docker compose -f docker/compose.yml run --rm mot-tests pytest -q tests`

GitHub Actions (design)
- Job `lint-typecheck-test`:
  - Build `mot-tests` image
  - Run the four commands above (ruff format, ruff check, pyright, pytest)
- Optional separate job `docker-build-smoke`:
  - `docker build` targets `web`, `worker`, `worker-cuda` (as needed)

---

## Phase 2 — Security scanning (secrets + dependency scanning)

Goal: prevent secrets from landing in git and keep dependencies patched.

Checklist
- [x] Add Dependabot config:
  - [x] Python deps (`pyproject.toml`)
  - [x] GitHub Actions updates
  - [x] Docker base image updates
- [x] Add secret scanning enforcement:
  - [ ] GitHub secret scanning + push protection (repo setting)
  - [x] CI secret scan job (gitleaks in Docker)
- [x] Add dependency vulnerability scan in CI:
  - [x] `pip-audit`
  - [ ] policy file for allowlist/false-positives
- [ ] (Optional) Container scan for built images (Trivy)

Docker-first commands
- Example (inside `mot-tests`):
  - `docker compose -f docker/compose.yml run --rm mot-tests pip-audit`

---

## Phase 3 — Reproducible benchmarking + automated perf smoke test

Goal: publish real numbers (FPS/latency/VRAM), catch perf regressions early.

Checklist
- [x] Define benchmark contract:
  - [x] input source (deterministic synthetic frames)
  - [x] metrics: FPS, p50/p95
  - [x] output: JSON
- [x] Add benchmark script:
  - [x] Talks to `/realtime/track`
  - [x] Records git SHA (via `GITHUB_SHA`)
  - [ ] Records model ref + Docker image digests
  - [ ] Captures VRAM peak via `nvidia-smi`
- [x] Add `benchmarks/README.md` with instructions
- [x] Add CI benchmark smoke (on-demand, CPU) that uploads JSON artifact
- [ ] Add CI perf smoke test on **self-hosted GPU runner**:
  - [ ] run benchmark
  - [ ] enforce thresholds (loose)
  - [ ] upload JSON artifact + PR comment summary

---

## Phase 4 — Model lifecycle polish (versioned artifacts + checksums + model card)

Goal: models are versioned, reproducible, and documented.

Checklist
- [ ] Add a model manifest (single source of truth): models, versions, SHA256, license
- [ ] Populate SHA256 checksums for downloaded weights (existing helpers in `scripts/download_models.py`)
- [ ] Version the model bundle separately from app version
- [ ] Automate release publishing:
  - [ ] PT → ONNX export
  - [ ] ONNX → Triton model repo
  - [ ] Publish OCI artifact (GHCR) + print digest ref
  - [ ] Attach checksums to GH Release
- [ ] Add a short model card:
  - [ ] data (high level), intended use, limits, privacy notes, license

---

## Phase 5 — Monitoring (Prometheus + Grafana)

Goal: visibility into FPS/latency/errors, queue depth, Triton health, GPU usage.

Checklist
- [ ] Add `/metrics` endpoints (Prometheus):
  - [ ] realtime backend: frame timings, fps, WS errors
  - [ ] worker: job durations, failures, queue depth
  - [ ] web: request durations/errors
- [ ] Add monitoring compose profile:
  - [ ] Prometheus service + config
  - [ ] Grafana service + provisioning
  - [ ] scrape Triton metrics (port 8002)
  - [ ] (Optional) redis exporter + DCGM GPU exporter
- [ ] Provide starter Grafana dashboards (checked into repo)

---

## Phase 6 — Improved UI/frontend experience

Goal: fewer "mystery failures", better realtime UX, better job UX.

Checklist
- [ ] Realtime stats strip:
  - [ ] FPS (client-side), server latency, object count
- [ ] Better error UX:
  - [ ] categorize tracker init vs WS timeout vs backend errors
  - [ ] persistent inline error panel
- [ ] Fix production cache-busting for JS/CSS
- [ ] Video upload page:
  - [ ] queue + job state UI (queued/running/done/failed)
  - [ ] download results + analytics summary

---

## Phase 7 — Repo hygiene (remove/relocate internal artifacts)

Goal: keep repo clean and user-facing.

Checklist
- [ ] Audit top-level artifacts (e.g. `digest.txt`, `repo_summary.txt`)
- [ ] Remove or move to `docs/internal/` if needed
- [ ] Add generated artifacts to `.gitignore`
- [ ] Ensure README/docs don’t duplicate long generated summaries

---

## Phase 8 — MOT analytics to improve the service

Goal: make results more valuable than "boxes + tracks".

Checklist (starter set)
- [ ] Occupancy over time (inside/outside, or zone-based)
- [ ] Dwell time per track + distribution
- [ ] Flow direction histogram + peak throughput
- [ ] Heatmaps (where people spend time)
- [ ] Track quality metrics:
  - [ ] avg track lifetime, fragmentation proxy, lost rate
- [ ] Export analytics:
  - [ ] JSON + CSV per run
  - [ ] summary panel in UI

Optional advanced analytics
- [ ] Trajectory clustering (common routes)
- [ ] Anomaly detection (loitering, wrong-way)
- [ ] Safety proximity events (distance threshold)
