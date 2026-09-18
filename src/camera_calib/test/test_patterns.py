"""标定板检测的纯算法测试: 合成一张棋盘格图, 看能不能检出正确数量的角点。"""

import cv2
import numpy as np
import pytest

from camera_calib.patterns import ChessboardPattern, create_pattern


def render_chessboard(columns: int, rows: int, square_px: int = 60, margin_px: int = 80):
    """画一张正视的棋盘格。columns/rows 是内部角点数, 所以格子数要 +1。"""
    squares_x, squares_y = columns + 1, rows + 1
    width = squares_x * square_px + 2 * margin_px
    height = squares_y * square_px + 2 * margin_px
    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2 == 0:
                continue
            y0 = margin_px + row * square_px
            x0 = margin_px + column * square_px
            image[y0:y0 + square_px, x0:x0 + square_px] = 0
    return image


def test_chessboard_detects_all_inner_corners():
    pattern = ChessboardPattern(columns=9, rows=6, square_length_m=0.025)
    gray = render_chessboard(9, 6)

    detection = pattern.detect(gray)

    assert detection.found
    assert detection.image_points.shape == (54, 1, 2)
    assert detection.object_points.shape == (54, 1, 3)


def test_chessboard_object_points_match_square_length():
    pattern = ChessboardPattern(columns=4, rows=3, square_length_m=0.03)
    detection = pattern.detect(render_chessboard(4, 3))

    assert detection.found
    points = detection.object_points.reshape(-1, 3)
    # 第 0 点在原点, 同一行相邻点间距 = 格边长, Z 恒为 0
    assert np.allclose(points[0], [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(points[1] - points[0]), 0.03)
    assert np.allclose(points[:, 2], 0.0)


def test_chessboard_detection_fails_on_blank_image():
    pattern = ChessboardPattern()
    detection = pattern.detect(np.full((480, 640), 127, dtype=np.uint8))

    assert not detection.found
    assert len(detection.image_points) == 0


def test_board_size_spans_outer_corners():
    """board_size_m 是角点张成的范围 (给视角指纹用), 不含最外圈半格。"""
    pattern = ChessboardPattern(columns=9, rows=6, square_length_m=0.025)

    width_m, height_m = pattern.board_size_m

    assert np.isclose(width_m, 8 * 0.025)
    assert np.isclose(height_m, 5 * 0.025)


@pytest.mark.parametrize("board_type", ["chessboard", "Chessboard", "棋盘格"])
def test_create_pattern_accepts_aliases(board_type):
    pattern = create_pattern(board_type, {"chessboard_columns": 7, "chessboard_rows": 5})

    assert isinstance(pattern, ChessboardPattern)
    assert "7x5" in pattern.description


def test_create_pattern_rejects_unknown_type():
    with pytest.raises(ValueError):
        create_pattern("nonexistent_board", {})


def test_draw_marks_corners_without_touching_input_shape():
    pattern = ChessboardPattern(columns=9, rows=6)
    gray = render_chessboard(9, 6)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    before = canvas.copy()

    pattern.draw(canvas, pattern.detect(gray))

    assert canvas.shape == before.shape
    assert not np.array_equal(canvas, before)
