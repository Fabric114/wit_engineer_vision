"""离线内参标定 (纯 CLI, 不用 ROS): 拿一堆图片直接标。

    ros2 run camera_calib offline_calib --images ~/calib_shots --output ~/calib_out
    # 或直接 python3 -m camera_calib.offline_calib --images ...
录像也行: 先 ffmpeg 抽帧, 再指到抽帧目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from camera_calib.intrinsic import SampleCollector, calibrate, save_camera_info, save_report
from camera_calib.patterns import create_pattern

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _load_config(filename: str) -> dict:
    """读工作区根 config/ (wit_engineer_vision/config/) 下的 YAML。

    向上查找同时含 config/ 与 src/ 的工作区根; 源码运行与拷贝安装下都能命中。
    """
    here = Path(__file__).resolve()
    root = next(
        (p for p in here.parents if (p / "config").is_dir() and (p / "src").is_dir()),
        here.parents[3],
    )
    with (root / "config" / filename).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线相机内参标定")
    parser.add_argument("--images", required=True, help="图片目录")
    parser.add_argument("--output", default="", help="输出目录, 默认 <images>/calibration")
    parser.add_argument("--board", default="", help="覆盖 config/intrinsic.yaml 的 board_type")
    args = parser.parse_args(argv)

    config = _load_config("intrinsic.yaml")
    if args.board:
        config["board_type"] = args.board
    pattern = create_pattern(config.get("board_type", "chessboard"), config)
    collector = SampleCollector(
        pattern,
        min_points=config.get("min_points", 20),
        min_sharpness=config.get("min_sharpness", 80.0),
        min_coverage=config.get("min_coverage", 0.035),
        min_signature_distance=config.get("min_signature_distance", 0.05),
        max_samples=config.get("max_samples", 300),
    )

    image_directory = Path(args.images).expanduser()
    files = sorted(
        path for path in image_directory.glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        print(f"目录里没有图片: {image_directory}", file=sys.stderr)
        return 2

    print(f"标定板: {pattern.description}")
    image_size = None
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        added, reason = collector.try_add(image, label=path.name)
        print(f"  {path.name}: {'采用' if added else '跳过'} ({reason})")

    if image_size is None:
        print("没有可读图片", file=sys.stderr)
        return 2

    try:
        result = calibrate(
            collector.samples, image_size,
            min_samples=int(config.get("min_samples", 15)),
            max_view_error_px=float(config.get("max_view_error_px", 0.8)),
        )
    except ValueError as exc:
        print(f"标定失败: {exc}", file=sys.stderr)
        return 1

    output_directory = Path(args.output).expanduser() if args.output else image_directory / "calibration"
    camera_info_path = save_camera_info(
        result, image_size, str(config.get("camera_name", "camera")),
        output_directory / "camera_info.yaml",
    )
    report_path = save_report(
        result, collector.samples, image_size, pattern.description,
        output_directory / "calibration_report.json",
    )
    print(
        f"RMS={result.rms:.4f}px, 用了 {len(result.kept)}/{len(collector.samples)} 张\n"
        f"内参: {camera_info_path}\n报告: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
