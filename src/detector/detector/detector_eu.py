"""能量单元 / 矿石检测 (eu): 单阶段 yolo detect, 双后端。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .letterbox import letterbox, to_blob
from .types import DetectionResult, Instance


class EuDetector:
    """单类矿石目标检测, 只出框 (无关键点)。OpenVINO 或 ultralytics 后端。"""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self._model_path = str(config["model"])
        self._device = str(config.get("device", "cpu"))
        self._conf = float(config.get("conf_threshold", 0.5))
        self._iou = float(config.get("iou_threshold", 0.45))
        self._max_instances = int(config.get("max_instances", 16))
        self._schema = str(config.get("schema", "energy_unit"))
        self._names: Dict[int, str] = {
            int(k): str(v) for k, v in (config.get("class_names") or {}).items()
        }
        # 后端相关句柄 (二选一)
        self._backend = ""
        self._model = None          # ultralytics
        self._compiled = None       # openvino
        self._output = None
        self._input_w = 640
        self._input_h = 640

    # ---------- 生命周期 ----------
    def open(self) -> None:
        xml = self._find_ir_xml(self._model_path)
        if xml is not None:
            self._open_openvino(xml)
        else:
            self._open_ultralytics()

    @staticmethod
    def _find_ir_xml(model_path: str) -> Optional[str]:
        """判定是否 OpenVINO IR: 返回 .xml 路径, 否则 None (走 ultralytics)。"""
        path = Path(model_path)
        if path.suffix == ".xml" and path.is_file():
            return str(path)
        if path.is_dir():
            best = path / "best.xml"
            if best.is_file():
                return str(best)
            xmls = sorted(path.glob("*.xml"))
            if xmls:
                return str(xmls[0])
        return None

    def _open_openvino(self, xml_path: str) -> None:
        from openvino import Core

        core = Core()
        model = core.read_model(xml_path)
        self._compiled = core.compile_model(model, self._device.upper())
        self._output = self._compiled.outputs[0]
        shape = [int(v) for v in self._compiled.inputs[0].shape]
        if len(shape) == 4 and shape[2] and shape[3]:
            self._input_h, self._input_w = shape[2], shape[3]
        self._backend = "openvino"
        # 类别名: config 优先, 否则读同目录 metadata.yaml 的 names
        if not self._names:
            self._names = self._names_from_metadata(xml_path)

    def _open_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "eu 检测 (.pt) 需要 ultralytics (含 torch), 当前环境未安装。\n"
                "  pip install ultralytics\n"
                "或把 .pt 导出成 OpenVINO IR (yolo export ... format=openvino),\n"
                "再在 detector.yaml 里把 eu.model 指向该 _openvino_model 目录即可纯 OpenVINO 运行。"
            ) from exc

        task = None if self._model_path.endswith(".pt") else "detect"
        self._model = YOLO(self._model_path, task=task)
        self._names = {int(k): str(v) for k, v in dict(self._model.names).items()}
        self._backend = "ultralytics"
        # 预热一次, 尽早暴露加载/设备错误
        self._model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8),
                            device=self._device, verbose=False)

    @staticmethod
    def _names_from_metadata(xml_path: str) -> Dict[int, str]:
        """读 IR 目录内 metadata.yaml 的 names (ultralytics 导出时附带)。"""
        try:
            import yaml
            meta = Path(xml_path).parent / "metadata.yaml"
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            return {int(k): str(v) for k, v in (data.get("names") or {}).items()}
        except Exception:
            return {}

    def close(self) -> None:
        self._model = None
        self._compiled = None
        self._output = None

    # ---------- 元信息 ----------
    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    @property
    def keypoint_names(self) -> List[str]:
        return []

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def description(self) -> str:
        return f"eu yolo-detect[{self._backend}] {self._model_path} @{self._device}"

    # ---------- 推理 ----------
    def detect(self, image: np.ndarray) -> DetectionResult:
        if self._backend == "openvino":
            return self._detect_openvino(image)
        if self._backend == "ultralytics":
            return self._detect_ultralytics(image)
        raise RuntimeError("EuDetector 未 open()")

    def _detect_openvino(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        padded, info = letterbox(image, (self._input_w, self._input_h))
        blob = to_blob(padded, swap_rb=True)
        output = np.asarray(self._compiled([blob])[self._output], dtype=np.float32)
        if output.ndim == 3:
            output = output[0]                 # [4+nc, N]
        if output.shape[0] < output.shape[1]:  # -> [N, 4+nc] (锚点数远大于通道数)
            output = output.T

        boxes_xywh = output[:, :4]
        cls_scores = output[:, 4:]
        class_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

        keep = confs >= self._conf
        boxes_xywh, confs, class_ids = boxes_xywh[keep], confs[keep], class_ids[keep]

        instances: List[Instance] = []
        if boxes_xywh.shape[0] > 0:
            # xywh(中心) -> xyxy (letterbox 640 空间)
            cx, cy, w, h = boxes_xywh.T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            selected = self._nms(xyxy, confs, class_ids, self._iou)
            selected = selected[np.argsort(-confs[selected])][: self._max_instances]
            for i in selected:
                class_id = int(class_ids[i])
                instances.append(
                    Instance(
                        class_id=class_id,
                        class_name=self._names.get(class_id, str(class_id)),
                        score=float(confs[i]),
                        bbox=info.undo_boxes(xyxy[i]),  # 反算回原图像素
                        keypoints=None,
                    )
                )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionResult(instances=instances, inference_ms=elapsed_ms,
                               stage_ms={"detect": elapsed_ms})

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
             class_ids: np.ndarray, iou_thr: float) -> np.ndarray:
        """按类分离的贪心 NMS (纯 numpy)。返回保留下来的行索引。"""
        # 给不同类别加大偏移, 使跨类框不互相抑制 (标准做法)
        if boxes_xyxy.shape[0] == 0:
            return np.empty(0, dtype=int)
        offset = class_ids.astype(np.float64) * (boxes_xyxy.max() + 1.0)
        boxes = boxes_xyxy + offset[:, None]
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return np.asarray(keep, dtype=int)

    def _detect_ultralytics(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        results = self._model.predict(
            source=image, conf=self._conf, iou=self._iou,
            device=self._device, verbose=False,
        )
        instances: List[Instance] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            order = np.argsort(-confs)[: self._max_instances]
            for i in order:
                class_id = int(cls[i])
                instances.append(
                    Instance(
                        class_id=class_id,
                        class_name=self._names.get(class_id, str(class_id)),
                        score=float(confs[i]),
                        bbox=np.asarray(xyxy[i], dtype=np.float64),
                        keypoints=None,
                    )
                )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionResult(instances=instances, inference_ms=elapsed_ms,
                               stage_ms={"detect": elapsed_ms})
