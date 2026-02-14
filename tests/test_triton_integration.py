from __future__ import annotations

import os
import time

import pytest


@pytest.mark.skipif(os.getenv("RUN_TRITON_INTEGRATION") != "1", reason="Set RUN_TRITON_INTEGRATION=1 to enable")
def test_triton_server_and_models_ready():
    triton_url = os.getenv("TRITON_URL", "triton:8001")
    model_repo_ref = os.getenv("TRITON_MODEL_REPO_REF", "")
    if not model_repo_ref:
        pytest.skip("TRITON_MODEL_REPO_REF not set; no real model repo to validate")

    grpcclient = pytest.importorskip("tritonclient.grpc")

    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)

    deadline = time.time() + 90
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            if client.is_server_live() and client.is_server_ready():
                break
        except Exception as e:  # pragma: no cover
            last_err = e
        time.sleep(1.0)
    else:
        raise AssertionError(f"Triton not live/ready at {triton_url}. Last error: {last_err}")

    model1 = os.getenv("TRITON_YOLO11_MODEL", "yolo11")
    model2 = os.getenv("TRITON_YOLO12_MODEL", "yolo12")

    # Wait for models to show as ready (Triton may still be loading).
    deadline = time.time() + 90
    while time.time() < deadline:
        ready1 = client.is_model_ready(model1)
        ready2 = client.is_model_ready(model2)
        if ready1 and ready2:
            break
        time.sleep(1.0)

    assert client.is_model_ready(model1), f"Model not ready: {model1}"
    assert client.is_model_ready(model2), f"Model not ready: {model2}"

    # Basic metadata sanity (ensures the models are actually loaded).
    md1 = client.get_model_metadata(model1)
    md2 = client.get_model_metadata(model2)

    assert md1.name == model1
    assert md2.name == model2
    assert len(md1.inputs) >= 1
    assert len(md2.inputs) >= 1
