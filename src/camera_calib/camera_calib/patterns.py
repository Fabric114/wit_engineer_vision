"""平面标定板: 棋盘格 / ChArUco / 对称圆点板。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


@dataclass
class Detection:
    """一次检测结果。object_points/image_points 形状为 (N,1,3)/(N,1,2), 与 cv2 标定接口一致。"""

    found: bool
    count: int
    object_points: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 3), np.float32))
    image_points: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 2), np.float32))
    extra: dict = field(default_factory=dict)


class Pattern(abc.ABC):
    """标定板抽象: 知道自己的物理尺寸, 会在灰度图里找自己的特征点。"""

    name = "pattern"

    @property
    @abc.abstractmethod
    def description(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def board_size_m(self) -> tuple:
        """(宽, 高) 米, 用于视角签名。"""

    @abc.abstractmethod
    def detect(self, gray: np.ndarray) -> Detection:
        ...

    def draw(self, image: np.ndarray, detection: Detection) -> None:
        """默认画法: 把检测到的点画成角点。"""
        if detection.found:
            cv2.drawChessboardCorners(image, (detection.count, 1), detection.image_points, True)


def _grid_object_points(columns: int, rows: int, spacing_m: float) -> np.ndarray:
    """规则网格的 3D 点, Z=0, 原点在第一个点。"""
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * float(spacing_m)
    return points.reshape(-1, 1, 3)


class ChessboardPattern(Pattern):
    """普通黑白棋盘格。columns/rows 是**内部角点数** (格子数 - 1)。"""

    name = "chessboard"

    def __init__(self, columns: int = 9, rows: int = 6, square_length_m: float = 0.025):
        self.columns = int(columns)
        self.rows = int(rows)
        self.square_length_m = float(square_length_m)
        self._object_points = _grid_object_points(self.columns, self.rows, self.square_length_m)

    @property
    def description(self) -> str:
        return (
            f"棋盘格 内部角点 {self.columns}x{self.rows}, "
            f"格边长 {self.square_length_m * 1000:.1f}mm"
        )

    @property
    def board_size_m(self) -> tuple:
        return ((self.columns - 1) * self.square_length_m, (self.rows - 1) * self.square_length_m)

    def detect(self, gray: np.ndarray) -> Detection:
        size = (self.columns, self.rows)
        corners = None
        # SB 版本对模糊/大畸变更鲁棒, 且自带亚像素; 老版本 OpenCV 没有则退回经典实现
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray, size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
        else:
            found = False
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray, size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA
                )
        if not found or corners is None:
            return Detection(found=False, count=0)
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        return Detection(
            found=True,
            count=len(corners),
            object_points=self._object_points.copy(),
            image_points=corners,
        )

    def draw(self, image: np.ndarray, detection: Detection) -> None:
        if detection.found:
            cv2.drawChessboardCorners(
                image, (self.columns, self.rows), detection.image_points, True
            )


class CharucoPattern(Pattern):
    """ChArUco 板: 允许部分遮挡/出画, 采样比棋盘格宽松, 标定更容易凑够视角。"""

    name = "charuco"

    def __init__(
        self,
        squares_x: int = 14,
        squares_y: int = 9,
        square_length_m: float = 0.020,
        marker_length_m: float = 0.015,
        dictionary: str = "DICT_5X5_1000",
    ):
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(f"不支持的 aruco 字典: {dictionary}")
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_length_m = float(square_length_m)
        self.marker_length_m = float(marker_length_m)
        self.dictionary_name = dictionary
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary]),
        )
        self._detector = self._build_detector()

    def _build_detector(self):
        detector_parameters = cv2.aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_parameters.cornerRefinementMaxIterations = 80
        detector_parameters.cornerRefinementMinAccuracy = 0.001
        detector_parameters.adaptiveThreshWinSizeMin = 3
        detector_parameters.adaptiveThreshWinSizeMax = 53
        detector_parameters.adaptiveThreshWinSizeStep = 4
        charuco_parameters = cv2.aruco.CharucoParameters()
        charuco_parameters.minMarkers = 2
        charuco_parameters.tryRefineMarkers = True
        return cv2.aruco.CharucoDetector(self.board, charuco_parameters, detector_parameters)

    @property
    def description(self) -> str:
        return (
            f"ChArUco {self.squares_x}x{self.squares_y}, 格 {self.square_length_m * 1000:.1f}mm, "
            f"码 {self.marker_length_m * 1000:.1f}mm, {self.dictionary_name}"
        )

    @property
    def board_size_m(self) -> tuple:
        return (self.squares_x * self.square_length_m, self.squares_y * self.square_length_m)

    def detect(self, gray: np.ndarray) -> Detection:
        corners, ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        count = 0 if ids is None else len(ids)
        if count < 4:
            return Detection(found=False, count=count)
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        return Detection(
            found=True,
            count=count,
            object_points=np.asarray(object_points, dtype=np.float32),
            image_points=np.asarray(image_points, dtype=np.float32),
            extra={"corners": corners, "ids": ids,
                   "marker_corners": marker_corners, "marker_ids": marker_ids},
        )

    def draw(self, image: np.ndarray, detection: Detection) -> None:
        marker_ids = detection.extra.get("marker_ids")
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(image, detection.extra["marker_corners"], marker_ids)
        ids = detection.extra.get("ids")
        if ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, detection.extra["corners"], ids)


class CircleGridPattern(Pattern):
    """对称圆点板 (如 HC7-200-10)。"""

    name = "circle_grid"

    def __init__(self, columns: int = 7, rows: int = 7, spacing_m: float = 0.020):
        self.columns = int(columns)
        self.rows = int(rows)
        self.spacing_m = float(spacing_m)
        self._object_points = _grid_object_points(self.columns, self.rows, self.spacing_m)

    @property
    def description(self) -> str:
        return f"对称圆点板 {self.columns}x{self.rows}, 间距 {self.spacing_m * 1000:.1f}mm"

    @property
    def board_size_m(self) -> tuple:
        return ((self.columns - 1) * self.spacing_m, (self.rows - 1) * self.spacing_m)

    def detect(self, gray: np.ndarray) -> Detection:
        found, centers = cv2.findCirclesGrid(
            gray, (self.columns, self.rows),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
        )
        if not found or centers is None:
            return Detection(found=False, count=0)
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 1, 2)
        return Detection(
            found=True,
            count=len(centers),
            object_points=self._object_points.copy(),
            image_points=centers,
        )

    def draw(self, image: np.ndarray, detection: Detection) -> None:
        if detection.found:
            cv2.drawChessboardCorners(
                image, (self.columns, self.rows), detection.image_points, True
            )


def create_pattern(board_type: str, parameters: Optional[dict] = None) -> Pattern:
    """数据驱动: 换板子只改 config/intrinsic.yaml 的 board_type 与尺寸。"""
    parameters = parameters or {}
    key = str(board_type).strip().lower()
    if key in ("chessboard", "checkerboard", "棋盘格"):
        return ChessboardPattern(
            columns=parameters.get("chessboard_columns", 9),
            rows=parameters.get("chessboard_rows", 6),
            square_length_m=parameters.get("chessboard_square_length_m", 0.025),
        )
    if key in ("charuco", "charuco_300"):
        return CharucoPattern(
            squares_x=parameters.get("charuco_squares_x", 14),
            squares_y=parameters.get("charuco_squares_y", 9),
            square_length_m=parameters.get("charuco_square_length_m", 0.020),
            marker_length_m=parameters.get("charuco_marker_length_m", 0.015),
            dictionary=parameters.get("charuco_dictionary", "DICT_5X5_1000"),
        )
    if key in ("circle_grid", "circles", "hc7_200_10"):
        return CircleGridPattern(
            columns=parameters.get("circle_columns", 7),
            rows=parameters.get("circle_rows", 7),
            spacing_m=parameters.get("circle_spacing_m", 0.020),
        )
    raise ValueError(f"不支持的标定板: {board_type}")
