"""科技核心/兑换站关键点检测 (tc): 单阶段 yolo-pose, OpenVINO 后端。

"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .letterbox import letterbox, to_blob
from .types import DetectionResult, Instance

BOX_COLUMNS = 6  # x1, y1, x2, y2, score, class


class TcPoseDetector:
    """一次前向拿到框 + 关键点。延迟最低, 兑换站默认方案。"""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self._model_path = str(config["model"])
        self._device = str(config.get("device", "CPU")).upper()
        self._score_threshold = float(config.get("conf_threshold", 0.5))
        self._kp_threshold = float(config.get("keypoint_score_threshold", 0.0))
        self._max_instances = int(config.get("max_instances", 16))
        self._schema = str(config.get("schema", "exchange_station"))
        self._class_names = {int(k): str(v) for k, v in (config.get("class_names") or {}).items()}
        self._keypoint_names: List[str] = [str(v) for v in (config.get("keypoint_names") or [])]
        self._input_w = 640
        self._input_h = 640
        self._compiled = None
        self._output = None

    # ---------- 生命周期 ----------
    def open(self) -> None:
        from openvino import Core

        core = Core()
        model = core.read_model(self._model_path)
        self._compiled = core.compile_model(model, self._device)
        self._output = self._compiled.outputs[0]
        shape = [int(v) for v in self._compiled.inputs[0].shape]
        if len(shape) == 4 and shape[2] and shape[3]:
            self._input_h, self._input_w = shape[2], shape[3]
        # 类别名: config 优先, 否则读 IR 内嵌 rt_info["model_info"]["labels"]
        if not self._class_names:
            self._class_names = self._labels_from_rt_info(model)

    @staticmethod
    def _labels_from_rt_info(model) -> Dict[int, str]:
        try:
            labels = model.get_rt_info(["model_info", "labels"]).astype(str)
            return {i: name for i, name in enumerate(str(labels).split())}
        except Exception:
            return {}

    def close(self) -> None:
        self._compiled = None
        self._output = None

    # ---------- 元信息 ----------
    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    @property
    def keypoint_names(self) -> List[str]:
        return list(self._keypoint_names)

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def description(self) -> str:
        return f"tc yolo-pose {self._model_path} @{self._device}"

    # ---------- 推理 ----------
    def detect(self, image: np.ndarray) -> DetectionResult:
        if self._compiled is None:
            raise RuntimeError("TcPoseDetector 未 open()")
        started = time.perf_counter()
        padded, info = letterbox(image, (self._input_w, self._input_h))
        blob = to_blob(padded, swap_rb=True)
        output = np.asarray(self._compiled([blob])[self._output], dtype=np.float32)
        if output.ndim == 3:
            output = output[0]  # [300, 42]

        keypoint_count = (output.shape[1] - BOX_COLUMNS) // 3
        scores = output[:, 4]
        keep = np.nonzero(scores >= self._score_threshold)[0]
        keep = keep[np.argsort(-scores[keep])][: self._max_instances]

        names = self._keypoint_names or [str(i) for i in range(keypoint_count)]
        instances: List[Instance] = []
        for row_index in keep:
            row = output[row_index]
            class_id = int(round(float(row[5])))
            keypoints = row[BOX_COLUMNS:].reshape(keypoint_count, 3).astype(np.float64)
            keypoints[:, :2] = info.undo_points(keypoints[:, :2])
            if self._kp_threshold > 0.0:
                keypoints[keypoints[:, 2] < self._kp_threshold, 2] = 0.0
            instances.append(
                Instance(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    score=float(row[4]),
                    bbox=info.undo_boxes(row[:4]),
                    keypoints=keypoints,
                    keypoint_names=names,
                )
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionResult(instances=instances, inference_ms=elapsed_ms,
                               stage_ms={"pose": elapsed_ms})
