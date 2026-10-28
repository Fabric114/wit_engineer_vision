"""从同步记录的 FK / 标定板 PnP 位姿生成相机到 arm_base 的外参。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from camera_calib.hand_eye import (
    calibrate_eye_to_hand,
    extrinsic_from_measured_board,
    eye_to_hand_residuals,
    save_extrinsic,
)


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / "config").is_dir() and (p / "src").is_dir()),
        Path.cwd(),
    )


def _transform(value, name: str) -> np.ndarray:
    if not isinstance(value, dict) or value.get("rows") != 4 or value.get("cols") != 4:
        raise ValueError(f"{name} 必须包含 rows: 4、cols: 4 和 16 个 data 值")
    matrix = np.asarray(value.get("data"), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{name} 需要 16 个 data 值")
    matrix = matrix.reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError(f"{name} 不是有效的刚体变换")
    return matrix


def _base_from_board(config: dict) -> np.ndarray:
    if config.get("base_from_board_confirmed") is not True:
        raise ValueError("方法 A 需实测 base_from_board，并设置 base_from_board_confirmed: true")
    pose = config["base_from_board"]
    xyz = np.asarray(pose["xyz_m"], dtype=np.float64)
    rpy = np.deg2rad(np.asarray(pose["rpy_deg"], dtype=np.float64))
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
        raise ValueError("base_from_board 的 xyz_m / rpy_deg 均需 3 个有限数值")
    roll, pitch, yaw = rpy
    rz = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]
    ry = cv2.Rodrigues(np.array([0.0, pitch, 0.0]))[0]
    rx = cv2.Rodrigues(np.array([roll, 0.0, 0.0]))[0]
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def _rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def _apply_extrinsic_file(config_path: Path, result_path: Path) -> None:
    content = config_path.read_text(encoding="utf-8")
    replacement = json.dumps(os.path.relpath(result_path, config_path.parent), ensure_ascii=False)
    updated, count = re.subn(r'^extrinsic_file:[^\n]*$', f'extrinsic_file: {replacement}', content, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"{config_path} 缺少唯一的 extrinsic_file 配置项")
    config_path.write_text(updated, encoding="utf-8")


def calibrate(samples_path: Path, config_path: Path, extrinsic_config: Path, output: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    document = yaml.safe_load(samples_path.read_text(encoding="utf-8")) or {}
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("采样文件需要 samples 列表")
    method = config.get("method", "B")
    if method == "A":
        if len(samples) != 1:
            raise ValueError("方法 A 需要恰好 1 组同步采样")
        camera_board = _transform(samples[0]["camera_from_board"], "camera_from_board[0]")
        result = extrinsic_from_measured_board(_base_from_board(config), camera_board)
        residuals = None
    elif method == "B":
        required = max(3, int(config.get("poses_required", 12)))
        if len(samples) < required:
            raise ValueError(f"方法 B 至少需要 {required} 组同步采样，当前 {len(samples)} 组")
        base_grippers = [_transform(s["base_from_gripper"], f"base_from_gripper[{i}]") for i, s in enumerate(samples)]
        camera_boards = [_transform(s["camera_from_board"], f"camera_from_board[{i}]") for i, s in enumerate(samples)]
        min_degrees = float(config.get("min_rotation_between_poses_deg", 8.0))
        if not np.isfinite(min_degrees) or min_degrees <= 0:
            raise ValueError("min_rotation_between_poses_deg 必须为正数")
        distinct = []
        for pose in base_grippers:
            if all(_rotation_degrees(pose, previous) >= min_degrees for previous in distinct):
                distinct.append(pose)
        if len(distinct) < required:
            raise ValueError(f"姿态旋转差至少 {min_degrees:g} 度的独立样本不足 {required} 组")
        result = calibrate_eye_to_hand(base_grippers, camera_boards)
        residuals = eye_to_hand_residuals(result, base_grippers, camera_boards)
    else:
        raise ValueError(f"未知手眼标定方法: {method}")

    if not np.isfinite(result).all() or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("标定求解未得到有效刚体变换")

    configured_output = config.get("output_path") or ""
    result_path = output or (Path(configured_output) if configured_output else Path("calibration") / datetime.now().strftime("%Y%m%d_%H%M%S") / "extrinsic.yaml")
    if not result_path.is_absolute():
        result_path = Path.cwd() / result_path
    result_path = result_path.resolve()
    if result_path == extrinsic_config.resolve():
        raise ValueError("输出路径不能覆盖 camera_extrinsic.yaml")
    # 先确认配置可更新，再写结果，避免留下无法使用的标定产物。
    if len(re.findall(r'^extrinsic_file:', extrinsic_config.read_text(encoding="utf-8"), re.MULTILINE)) != 1:
        raise ValueError("camera_extrinsic.yaml 缺少唯一的 extrinsic_file 配置项")
    save_extrinsic(result, result_path)
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_workspace_root(), capture_output=True, text=True, check=False).stdout.strip()
    report = {
        "method": method,
        "date": datetime.now().astimezone().isoformat(),
        "camera_sn": document.get("camera_sn", ""),
        "git_hash": git_hash,
        "samples": len(samples),
        "residuals": residuals,
    }
    result_path.with_name("extrinsic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _apply_extrinsic_file(extrinsic_config, result_path)
    return result_path


def main(argv=None) -> int:
    root = _workspace_root()
    parser = argparse.ArgumentParser(description="离线手眼外参标定（输入同步 FK / 标定板 PnP 位姿）")
    parser.add_argument("--samples", required=True, type=Path, help="采样 YAML 文件")
    parser.add_argument("--config", type=Path, default=root / "config/hand_eye.yaml")
    parser.add_argument("--extrinsic-config", type=Path, default=root / "config/camera_extrinsic.yaml")
    parser.add_argument("--output", type=Path, help="输出 extrinsic.yaml 路径")
    args = parser.parse_args(argv)
    try:
        path = calibrate(args.samples, args.config, args.extrinsic_config, args.output)
    except (KeyError, OSError, TypeError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    print(f"外参已保存: {path}")
    print(f"外参配置已更新: {args.extrinsic_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
