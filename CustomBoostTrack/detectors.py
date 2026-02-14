import os
import concurrent.futures
from typing import Any

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2




from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



from abc import ABC, abstractmethod


_TRITON_ENSEMBLE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small, dependency-free Weighted Box Fusion implementation.

    Inputs are expected in normalized xyxy coordinates in [0, 1].

    This is sufficient for the project's ensemble use (two detectors, usually one class).
    """

    if weights is None:
        weights = [1.0] * len(boxes_list)
    if not (len(boxes_list) == len(scores_list) == len(labels_list) == len(weights)):
        raise ValueError("boxes_list/scores_list/labels_list/weights must have same length")

    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    all_weights: list[float] = []
    all_labels: list[int] = []

    for w, boxes, scores, labels in zip(weights, boxes_list, scores_list, labels_list):
        if boxes is None or scores is None or labels is None:
            continue
        if len(boxes) == 0:
            continue
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        for box, score, lab in zip(boxes, scores, labels):
            if float(score) < float(skip_box_thr):
                continue
            all_boxes.append(box.astype(np.float32))
            # Keep confidence in the original scale; use weights only for box averaging.
            all_scores.append(float(score))
            all_weights.append(float(w))
            all_labels.append(int(lab))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Cluster per label.
    out_boxes: list[np.ndarray] = []
    out_scores: list[float] = []
    out_labels: list[int] = []

    boxes_arr = np.stack(all_boxes, axis=0)
    scores_arr = np.asarray(all_scores, dtype=np.float32)
    weights_arr = np.asarray(all_weights, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)

    for lab in np.unique(labels_arr):
        idx = np.where(labels_arr == lab)[0]
        if idx.size == 0:
            continue

        # Sort by score descending.
        idx = idx[np.argsort(scores_arr[idx])[::-1]]
        used = np.zeros(idx.shape[0], dtype=bool)

        for i, gi in enumerate(idx):
            if used[i]:
                continue
            cluster = [gi]
            used[i] = True

            for j, gj in enumerate(idx[i + 1 :], start=i + 1):
                if used[j]:
                    continue
                if _iou_xyxy(boxes_arr[gi], boxes_arr[gj]) >= float(iou_thr):
                    used[j] = True
                    cluster.append(gj)

            cluster_boxes = boxes_arr[cluster]
            cluster_scores = scores_arr[cluster]
            cluster_weights = weights_arr[cluster]
            # Standard WBF uses weights to influence box averaging; confidence stays comparable
            # to detector confidences (so thresholds like 0.1 mean what users expect).
            weighted = cluster_scores * cluster_weights
            wsum = float(np.sum(weighted))
            if wsum <= 0:
                continue
            fused = np.sum(cluster_boxes * weighted[:, None], axis=0) / wsum
            fused_score = float(np.clip(np.max(cluster_scores), 0.0, 1.0))

            out_boxes.append(fused.astype(np.float32))
            out_scores.append(fused_score)
            out_labels.append(int(lab))

    if not out_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(out_boxes, axis=0).astype(np.float32),
        np.asarray(out_scores, dtype=np.float32),
        np.asarray(out_labels, dtype=np.float32),
    )

class Detector(ABC):
    @abstractmethod
    def __call__(self, img):
        pass


class YoloDetector(Detector):
    def __init__(self, yolo_path, conf = None):
        # Lazy import so this module can still be used in Triton mode
        # without requiring ultralytics at import time.
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(yolo_path)
        self.conf = conf

    def __call__(self, img):
        if self.conf is not None:
            results = self.model(img, conf=self.conf)[0]
        else:   
            results = self.model(img)[0]  # Let Ultralytics scale to input resolution
        annotations = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Only keep 'person' class (class ID 0)
                xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                conf = box.conf[0].item()
                annotations.append(xyxy + [conf])
        return torch.tensor(annotations, dtype=torch.float32) if annotations else torch.zeros((0, 5), dtype=torch.float32)


def _letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad to fit new_shape, returning (padded_img, scale, (pad_x, pad_y))."""
    shape = img.shape[:2]  # (h, w)
    new_h, new_w = new_shape
    r = min(new_w / shape[1], new_h / shape[0])
    resized = cv2.resize(img, (int(round(shape[1] * r)), int(round(shape[0] * r))), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - resized.shape[1]
    pad_h = new_h - resized.shape[0]
    pad_x = pad_w // 2
    pad_y = pad_h // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_h - pad_y,
        pad_x,
        pad_w - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, (pad_x, pad_y)


class TritonYoloDetector(Detector):
    """YOLO detector served by NVIDIA Triton (gRPC).

    This is intentionally flexible: it discovers input/output names from Triton metadata.
    It supports two common export styles:
      1) NMS-style outputs: num_dets, det_boxes, det_scores, det_classes
      2) Raw YOLOv8-style output tensor (e.g. output0: [1, N, 84])
    """

    def __init__(
        self,
        model_name: str,
        conf: float | None = None,
        url: str | None = None,
        input_size: int | None = None,
        model_version: str = "",
    ) -> None:
        try:
            import tritonclient.grpc as grpcclient  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "tritonclient is required for TritonYoloDetector. Install mot-web[ml] and set DETECTOR_BACKEND=triton."
            ) from e

        self.grpcclient = grpcclient
        def _env_num(name: str, default: str) -> str:
            v = os.getenv(name)
            if v is None:
                return default
            v = str(v).strip()
            return v if v != "" else default

        self.url = url or os.getenv("TRITON_URL", "localhost:8001")
        self.model_name = model_name
        self.model_version = model_version
        self.conf = float(conf) if conf is not None else float(_env_num("TRITON_CONF", "0.25"))
        self.input_size = int(input_size) if input_size is not None else int(_env_num("TRITON_INPUT_SIZE", "640"))

        # Postprocessing controls for raw outputs (i.e., not server-side NMS).
        # These help reduce false positives by suppressing overlapping/noisy boxes.
        self.nms_iou = float(_env_num("TRITON_NMS_IOU", "0.70"))
        self.max_det = int(_env_num("TRITON_MAX_DET", "300"))
        self.pre_topk = int(_env_num("TRITON_PRE_TOPK", "1000"))

        self._client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self._input_name: str | None = None
        self._output_names: list[str] | None = None
        self._output_mode: str | None = None  # "nms" | "raw"

    def _ensure_io(self) -> None:
        if self._input_name is not None and self._output_names is not None and self._output_mode is not None:
            return

        md = self._client.get_model_metadata(self.model_name, self.model_version)
        cfg = self._client.get_model_config(self.model_name, self.model_version)

        # pick first input
        self._input_name = md.inputs[0].name
        output_names = [o.name for o in md.outputs]

        # decide output mode
        nms_keys = {"num_dets", "det_boxes", "det_scores", "det_classes"}
        if nms_keys.issubset(set(output_names)):
            self._output_mode = "nms"
            self._output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        else:
            # Prefer a single raw output tensor
            self._output_mode = "raw"
            # heuristic: pick first output
            self._output_names = [output_names[0]]

    def preprocess(self, img: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Preprocess a BGR/RGB image into Triton NCHW float32 input.

        Returns (x, meta) where meta contains enough information to map boxes
        back to the original image coordinate space.
        """

        self._ensure_io()

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        # If caller passes BGR (common with cv2), convert to RGB unless explicitly disabled.
        if os.getenv("TRITON_EXPECTS_BGR", "0") != "1":
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        padded, r, (pad_x, pad_y) = _letterbox(img, (self.input_size, self.input_size))

        x = padded.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = np.expand_dims(x, 0)  # NCHW

        meta = {
            "h0": int(h0),
            "w0": int(w0),
            "r": float(r),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
        }
        return x, meta

    def infer_preprocessed(self, x: np.ndarray, meta: dict[str, Any]):
        """Run inference on already-preprocessed NCHW FP32 input."""

        self._ensure_io()

        inp = self.grpcclient.InferInput(self._input_name, x.shape, "FP32")
        inp.set_data_from_numpy(x)

        outputs = [self.grpcclient.InferRequestedOutput(n) for n in (self._output_names or [])]
        resp = self._client.infer(self.model_name, inputs=[inp], outputs=outputs, model_version=self.model_version)

        h0 = int(meta["h0"])
        w0 = int(meta["w0"])
        r = float(meta["r"])
        pad_x = int(meta["pad_x"])
        pad_y = int(meta["pad_y"])

        if self._output_mode == "nms":
            num = int(resp.as_numpy("num_dets")[0])
            boxes = resp.as_numpy("det_boxes")[0][:num]
            scores = resp.as_numpy("det_scores")[0][:num]
            classes = resp.as_numpy("det_classes")[0][:num]

            if boxes.size == 0:
                return torch.zeros((0, 5), dtype=torch.float32)

            keep = (classes.astype(np.int64) == 0) & (scores >= self.conf)
            boxes = boxes[keep]
            scores = scores[keep]
            if boxes.size == 0:
                return torch.zeros((0, 5), dtype=torch.float32)

            if float(np.max(boxes)) <= 1.5:
                boxes = boxes * np.array(
                    [self.input_size, self.input_size, self.input_size, self.input_size], dtype=np.float32
                )

            boxes[:, [0, 2]] -= pad_x
            boxes[:, [1, 3]] -= pad_y
            boxes /= max(r, 1e-9)

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

            ann = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1).astype(np.float32)
            return torch.from_numpy(ann)

        # raw YOLOv8-style output
        out0 = resp.as_numpy((self._output_names or [""])[0])
        if out0 is None:
            return torch.zeros((0, 5), dtype=torch.float32)

        preds = _triton_raw_to_preds(out0)
        if preds is None or preds.shape[1] < 5:
            return torch.zeros((0, 5), dtype=torch.float32)

        boxes, conf, cls = _decode_yolo_preds(preds)
        if boxes.size == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        keep = (cls == 0) & (conf >= self.conf)
        boxes = boxes[keep]
        conf = conf[keep]
        cls = cls[keep]
        if boxes.size == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        if float(np.max(boxes)) <= 1.5:
            boxes = boxes * np.array(
                [self.input_size, self.input_size, self.input_size, self.input_size], dtype=np.float32
            )

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= max(r, 1e-9)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        if boxes.shape[0] > 0:
            if int(self.pre_topk) > 0 and boxes.shape[0] > int(self.pre_topk):
                idx = np.argsort(conf)[::-1][: int(self.pre_topk)]
                boxes = boxes[idx]
                conf = conf[idx]
                cls = cls[idx]

            keep_all: list[int] = []
            try:
                from torchvision.ops import nms as tv_nms  # type: ignore

                for lab in np.unique(cls):
                    inds = np.where(cls == lab)[0]
                    if inds.size == 0:
                        continue
                    b = torch.from_numpy(boxes[inds]).to(torch.float32)
                    s = torch.from_numpy(conf[inds]).to(torch.float32)
                    kept = tv_nms(b, s, float(self.nms_iou)).cpu().numpy().astype(np.int64)
                    keep_all.extend(inds[kept].tolist())
            except Exception:
                order = np.argsort(conf)[::-1]
                while order.size > 0:
                    i = int(order[0])
                    keep_all.append(i)
                    if order.size == 1:
                        break
                    rest = order[1:]
                    ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
                    order = rest[ious < float(self.nms_iou)]

            if keep_all:
                keep_all = sorted(set(keep_all), key=lambda i: float(conf[i]), reverse=True)
                if int(self.max_det) > 0:
                    keep_all = keep_all[: int(self.max_det)]
                boxes = boxes[keep_all]
                conf = conf[keep_all]

        ann = np.concatenate([boxes, conf.reshape(-1, 1)], axis=1).astype(np.float32)
        return torch.from_numpy(ann)

    def __call__(self, img: Any):
        x, meta = self.preprocess(img)
        return self.infer_preprocessed(x, meta)


def _triton_raw_to_preds(out0: np.ndarray) -> np.ndarray | None:
    """Normalize Triton raw output into a [N, D] float32 matrix.

    Common Ultralytics ONNX layouts:
      - [1, N, D]
      - [1, D, N]
      - [N, D]
      - [D, N]
    """

    a = np.asarray(out0)
    if a.ndim == 3:
        a = a[0]
    elif a.ndim != 2:
        return None

    a = np.asarray(a, dtype=np.float32)

    # Heuristic transpose for [D, N] where N is usually much larger.
    if a.shape[0] <= 256 and a.shape[1] > a.shape[0] and a.shape[1] >= 64:
        return a.transpose(1, 0)

    return a


def _decode_yolo_preds(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a [N, D] prediction matrix into (boxes_xyxy, conf, cls).

    Supported formats:
      - D == 6: [x1,y1,x2,y2,score,class] (already NMS'd)
      - D >= 6: raw outputs with either:
          * [cx,cy,w,h, cls0..]  (typical Ultralytics v8/v11/v12 ONNX)
          * [cx,cy,w,h,obj, cls0..] (common in other exporters)
    """

    if preds.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    d = int(preds.shape[1])

    # NMS-like output: xyxy + score + class
    if d == 6:
        boxes = preds[:, :4].astype(np.float32)
        conf = preds[:, 4].astype(np.float32)
        cls = preds[:, 5].astype(np.int64)
        return boxes, conf, cls

    coords = preds[:, :4].astype(np.float32)

    # Decide whether column 4 is objectness or class0.
    # Ultralytics raw ONNX for COCO is usually D==84 (4 + 80 classes, no objectness).
    # Some exporters include objectness: D==85 (4 + 1 + 80).
    has_obj: bool
    if d == 85 or (7 <= d <= 10):
        has_obj = True
    else:
        has_obj = False

    if has_obj:
        if d <= 6:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        obj = preds[:, 4].astype(np.float32)
        cls_scores = preds[:, 5:].astype(np.float32)
        if cls_scores.shape[1] < 1:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        cls = np.argmax(cls_scores, axis=1).astype(np.int64)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls]
        conf = obj * cls_conf
    else:
        cls_scores = preds[:, 4:].astype(np.float32)
        if cls_scores.shape[1] < 1:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        cls = np.argmax(cls_scores, axis=1).astype(np.int64)
        conf = cls_scores[np.arange(cls_scores.shape[0]), cls]

    # Determine whether coords are xyxy or (cx,cy,w,h)
    # If the majority look like xyxy, keep as-is.
    if coords.shape[0] > 0:
        looks_xyxy = np.mean((coords[:, 2] >= coords[:, 0]) & (coords[:, 3] >= coords[:, 1]))
    else:
        looks_xyxy = 0.0

    if looks_xyxy >= 0.8:
        boxes = coords
    else:
        cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    return boxes, conf.astype(np.float32), cls


# EnsembleDetector remains unchanged as it assumes original resolution inputs
    

# Faster R-CNN Detector (removed conf_threshold)
class FasterRCNNDetector:
    def __init__(self, model_path):
        anchor_sizes = tuple((int(w),) for w, _ in [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet101',
            pretrained=False,
            trainable_layers=3
        )
        self.model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_test=4000,
            rpn_post_nms_top_n_test=400,
            rpn_nms_thresh=0.67,
            box_score_thresh=0.085,
            box_nms_thresh=0.5,
            box_detections_per_img=300
        )
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose([
            A.Resize(height=640, width=640, always_apply=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img_rgb)
        img_tensor = augmented['image'].to(self.device)
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu()
        scores = predictions['scores'].cpu()
        labels = predictions['labels'].cpu()
        mask = (labels == 1)  # Only filter by class (pedestrian), no confidence threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) > 0:
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            annotations = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
        else:
            annotations = torch.zeros((0, 5))
        
        return annotations



# In detectors.py

class EnsembleDetector(Detector):
    def __init__(self, model1: Detector, model2: Detector, model1_weight=0.7, model2_weight=0.3, iou_thresh=0.6, conf_thresh=0.3):
        self.model1 = model1
        self.model2 = model2
        self.model1_weight = model1_weight
        self.model2_weight = model2_weight
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh  # New parameter for confidence filtering
        self._parallel_disabled = False

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]

        # Get predictions (optionally share preprocess + overlap Triton calls)
        model1_preds = None
        model2_preds = None

        use_parallel = (os.getenv("REALTIME_TRITON_PARALLEL", "1").strip() != "0") and (not self._parallel_disabled)
        use_shared_pre = os.getenv("REALTIME_TRITON_SHARED_PREPROCESS", "1").strip() != "0"

        if (
            use_shared_pre
            and isinstance(self.model1, TritonYoloDetector)
            and isinstance(self.model2, TritonYoloDetector)
            and int(getattr(self.model1, "input_size", 0)) == int(getattr(self.model2, "input_size", -1))
        ):
            x, meta = self.model1.preprocess(img)
            if use_parallel:
                f1 = _TRITON_ENSEMBLE_EXECUTOR.submit(self.model1.infer_preprocessed, x, meta)
                f2 = _TRITON_ENSEMBLE_EXECUTOR.submit(self.model2.infer_preprocessed, x, meta)
                try:
                    model1_preds = f1.result()
                    model2_preds = f2.result()
                except Exception as e:
                    self._parallel_disabled = True
                    if os.getenv("REALTIME_DEBUG", "").strip():
                        print(f"[EnsembleDetector] parallel Triton inference failed; retrying sequential. err={e!r}")
                    model1_preds = self.model1.infer_preprocessed(x, meta)
                    model2_preds = self.model2.infer_preprocessed(x, meta)
            else:
                model1_preds = self.model1.infer_preprocessed(x, meta)
                model2_preds = self.model2.infer_preprocessed(x, meta)
        else:
            model1_preds = self.model1(img)  # Already filtered to 'person' in YoloDetector
            model2_preds = self.model2(img)

        # Prepare for WBF
        if len(model1_preds) > 0:
            # Ensure tensor is on CPU before converting to numpy
            if isinstance(model1_preds, torch.Tensor):
                if model1_preds.device.type != 'cpu':
                    model1_preds = model1_preds.cpu()
            yolo_boxes = model1_preds[:, :4].numpy()
            yolo_scores = model1_preds[:, 4].numpy()
            yolo_boxes_normalized = yolo_boxes / np.array([orig_w, orig_h, orig_w, orig_h])
            yolo_labels = np.zeros(len(yolo_scores))  # Class 0 for 'person'
        else:
            yolo_boxes_normalized = np.array([])
            yolo_scores = np.array([])
            yolo_labels = np.array([])
    
        if len(model2_preds) > 0:
            # Ensure tensor is on CPU before converting to numpy
            if isinstance(model2_preds, torch.Tensor):
                if model2_preds.device.type != 'cpu':
                    model2_preds = model2_preds.cpu()
            other_boxes = model2_preds[:, :4].numpy()
            other_scores = model2_preds[:, 4].numpy()
            other_boxes_normalized = other_boxes / np.array([orig_w, orig_h, orig_w, orig_h])
            other_labels = np.zeros(len(other_scores))  # Class 0 for 'person'
        else:
            other_boxes_normalized = np.array([])
            other_scores = np.array([])
            other_labels = np.array([])

        # Weighted Box Fusion
        topk = int(os.getenv("REALTIME_WBF_TOPK", "200"))
        skip_thr_env = os.getenv("REALTIME_WBF_SKIP_BOX_THR", "").strip()
        if skip_thr_env:
            skip_thr = float(skip_thr_env)
        else:
            # Default: keep boxes down to half the final threshold (keeps recall while shrinking WBF work).
            skip_thr = max(0.0, float(self.conf_thresh) * 0.5)

        def _prefilter(boxes_n: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if scores.size == 0:
                return boxes_n, scores, labels
            if float(skip_thr) > 0.0:
                m = scores >= float(skip_thr)
                boxes_n = boxes_n[m]
                scores = scores[m]
                labels = labels[m]
            if int(topk) > 0 and scores.size > int(topk):
                # keep top-K by score
                idx = np.argpartition(-scores, int(topk) - 1)[: int(topk)]
                idx = idx[np.argsort(scores[idx])[::-1]]
                boxes_n = boxes_n[idx]
                scores = scores[idx]
                labels = labels[idx]
            return boxes_n, scores, labels

        yolo_boxes_normalized, yolo_scores, yolo_labels = _prefilter(yolo_boxes_normalized, yolo_scores, yolo_labels)
        other_boxes_normalized, other_scores, other_labels = _prefilter(other_boxes_normalized, other_scores, other_labels)
        boxes_list = [yolo_boxes_normalized, other_boxes_normalized]
        scores_list = [yolo_scores, other_scores]
        labels_list = [yolo_labels, other_labels]  # Use actual class labels
        weights = [self.model1_weight, self.model2_weight]

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=self.iou_thresh,
            skip_box_thr=float(skip_thr),
        )

        # Filter to only 'person' (class 0) after WBF
        person_mask = labels == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        # Apply confidence threshold
        conf_mask = scores > self.conf_thresh
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]

        # Scale back to original resolution
        if len(boxes) > 0:
            boxes = boxes * np.array([orig_w, orig_h, orig_w, orig_h])
            annotations = torch.tensor(np.hstack((boxes, scores[:, np.newaxis])), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)

        # Keep tensor on CPU to avoid device conversion issues later
        if annotations.device.type != 'cpu':
            annotations = annotations.cpu()
        
        return annotations