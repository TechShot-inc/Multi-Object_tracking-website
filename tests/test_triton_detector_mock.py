from __future__ import annotations

import types

import numpy as np
import pytest


def test_triton_yolo_detector_nms_mock(monkeypatch):
    # This test is meant for ML-enabled environments. The default `mot-tests`
    # image intentionally does not include CustomBoostTrack/torch.
    pytest.importorskip("torch")
    pytest.importorskip("CustomBoostTrack.detectors")

    # Import module lazily so we can inject a fake tritonclient.grpc
    import importlib

    # Build a fake tritonclient.grpc module
    class _InferInput:
        def __init__(self, name, shape, dtype):
            self.name, self.shape, self.dtype = name, shape, dtype
            self.data = None

        def set_data_from_numpy(self, arr):
            self.data = arr

    class _InferRequestedOutput:
        def __init__(self, name):
            self.name = name

    class _Resp:
        def __init__(self, arrays):
            self.arrays = arrays

        def as_numpy(self, name):
            return self.arrays.get(name)

    class _Client:
        def __init__(self, url, verbose=False):
            self.url = url

        def get_model_metadata(self, model_name, model_version=""):
            # mimic grpc metadata object with .inputs/.outputs having .name
            io = types.SimpleNamespace
            return io(
                inputs=[io(name="images")],
                outputs=[io(name="num_dets"), io(name="det_boxes"), io(name="det_scores"), io(name="det_classes")],
            )

        def get_model_config(self, model_name, model_version=""):
            return types.SimpleNamespace()

        def infer(self, model_name, inputs, outputs, model_version=""):
            # one person box in normalized xyxy in letterbox input space
            num = np.array([1], dtype=np.int32)
            boxes = np.array([[[0.1, 0.1, 0.5, 0.5]]], dtype=np.float32)
            scores = np.array([[0.9]], dtype=np.float32)
            classes = np.array([[0]], dtype=np.float32)
            return _Resp({"num_dets": num, "det_boxes": boxes, "det_scores": scores, "det_classes": classes})

    fake_grpc = types.SimpleNamespace(
        InferInput=_InferInput,
        InferRequestedOutput=_InferRequestedOutput,
        InferenceServerClient=_Client,
    )

    import sys

    # Inject fake module path: tritonclient.grpc
    fake_tritonclient = types.SimpleNamespace(grpc=fake_grpc)
    monkeypatch.setitem(sys.modules, "tritonclient", fake_tritonclient)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", fake_grpc)

    # Import/reload to ensure it picks our fake
    det_mod = importlib.import_module("CustomBoostTrack.detectors")
    det_mod = importlib.reload(det_mod)

    detector = det_mod.TritonYoloDetector(model_name="yolo11", conf=0.25, url="fake:8001", input_size=640)

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = detector(img)

    assert hasattr(out, "shape")
    assert out.shape[1] == 5
    assert out.shape[0] == 1


def test_triton_yolo_detector_raw_mock_d84_transposed(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("CustomBoostTrack.detectors")

    import importlib

    class _InferInput:
        def __init__(self, name, shape, dtype):
            self.name, self.shape, self.dtype = name, shape, dtype
            self.data = None

        def set_data_from_numpy(self, arr):
            self.data = arr

    class _InferRequestedOutput:
        def __init__(self, name):
            self.name = name

    class _Resp:
        def __init__(self, arrays):
            self.arrays = arrays

        def as_numpy(self, name):
            return self.arrays.get(name)

    class _Client:
        def __init__(self, url, verbose=False):
            self.url = url

        def get_model_metadata(self, model_name, model_version=""):
            io = types.SimpleNamespace
            return io(inputs=[io(name="images")], outputs=[io(name="output0")])

        def get_model_config(self, model_name, model_version=""):
            return types.SimpleNamespace()

        def infer(self, model_name, inputs, outputs, model_version=""):
            # Provide [1, D, N] to trigger transpose handling (D=84, N=128).
            d, n = 84, 128
            out = np.zeros((1, d, n), dtype=np.float32)

            # det 0: person class (cls0) high
            v0 = np.zeros((d,), dtype=np.float32)
            v0[0:4] = np.array([320, 320, 100, 200], dtype=np.float32)  # cx, cy, w, h
            v0[4] = 0.9  # cls0
            v0[5] = 0.1  # cls1
            out[0, :, 0] = v0

            # det 1: non-person class (cls1) high
            v1 = np.zeros((d,), dtype=np.float32)
            v1[0:4] = np.array([200, 200, 50, 50], dtype=np.float32)
            v1[4] = 0.05
            v1[5] = 0.95
            out[0, :, 1] = v1

            return _Resp({"output0": out})

    fake_grpc = types.SimpleNamespace(
        InferInput=_InferInput,
        InferRequestedOutput=_InferRequestedOutput,
        InferenceServerClient=_Client,
    )

    import sys

    fake_tritonclient = types.SimpleNamespace(grpc=fake_grpc)
    monkeypatch.setitem(sys.modules, "tritonclient", fake_tritonclient)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", fake_grpc)

    det_mod = importlib.import_module("CustomBoostTrack.detectors")
    det_mod = importlib.reload(det_mod)

    detector = det_mod.TritonYoloDetector(model_name="yolo11", conf=0.25, url="fake:8001", input_size=640)
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = detector(img)

    assert out.shape[1] == 5
    assert out.shape[0] == 1


def test_triton_yolo_detector_raw_mock_n6_xyxy(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("CustomBoostTrack.detectors")

    import importlib

    class _InferInput:
        def __init__(self, name, shape, dtype):
            self.name, self.shape, self.dtype = name, shape, dtype
            self.data = None

        def set_data_from_numpy(self, arr):
            self.data = arr

    class _InferRequestedOutput:
        def __init__(self, name):
            self.name = name

    class _Resp:
        def __init__(self, arrays):
            self.arrays = arrays

        def as_numpy(self, name):
            return self.arrays.get(name)

    class _Client:
        def __init__(self, url, verbose=False):
            self.url = url

        def get_model_metadata(self, model_name, model_version=""):
            io = types.SimpleNamespace
            return io(inputs=[io(name="images")], outputs=[io(name="output0")])

        def get_model_config(self, model_name, model_version=""):
            return types.SimpleNamespace()

        def infer(self, model_name, inputs, outputs, model_version=""):
            # [1, N, 6] => [x1,y1,x2,y2,score,cls]
            out = np.array(
                [
                    [
                        [10, 10, 50, 50, 0.9, 0],
                        [0, 0, 10, 10, 0.95, 2],
                        [5, 5, 10, 10, 0.1, 0],
                    ]
                ],
                dtype=np.float32,
            )
            return _Resp({"output0": out})

    fake_grpc = types.SimpleNamespace(
        InferInput=_InferInput,
        InferRequestedOutput=_InferRequestedOutput,
        InferenceServerClient=_Client,
    )

    import sys

    fake_tritonclient = types.SimpleNamespace(grpc=fake_grpc)
    monkeypatch.setitem(sys.modules, "tritonclient", fake_tritonclient)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", fake_grpc)

    det_mod = importlib.import_module("CustomBoostTrack.detectors")
    det_mod = importlib.reload(det_mod)

    detector = det_mod.TritonYoloDetector(model_name="yolo11", conf=0.25, url="fake:8001", input_size=640)
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = detector(img)

    assert out.shape[1] == 5
    assert out.shape[0] == 1
