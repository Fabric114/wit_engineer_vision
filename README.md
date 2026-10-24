# wit_engineer_vision

武汉工程大学 27 赛季重装机器人（工程 + 英雄）的辅助装配上位机：识别兑换站 → 规划接近 → 约束插入。

## 测试

`colcon test` 收集不到这些用例，禁止用它判断结果。在仓库根目录直接跑 pytest：

```bash
source /opt/ros/humble/setup.bash

# 纯算法包（不需要 source install）
for pkg in solver shm camera_calib; do
  PYTHONPATH=src/$pkg${PYTHONPATH:+:$PYTHONPATH} \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/$pkg/test -q
done

# sim 需要自己的 PYTHONPATH
PYTHONPATH=src/runtime/sim${PYTHONPATH:+:$PYTHONPATH} \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/runtime/sim/test -q

# planning：10 个 node 用例需要先 source ROS + install/setup.bash
source install/setup.bash
PYTHONPATH=src/planning${PYTHONPATH:+:$PYTHONPATH} \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/planning/test -q
```

基线（2026-10-24 实跑）：

| 包 | 不 source install | source ROS + install | 说明 |
|---|---:|---:|---|
| planning | 86 passed, 1 skipped | 105 passed | 10 个 node 用例需可写 `~/.ros` |
| solver | 53 passed, 1 skipped | 53 passed, 1 skipped | skip 是真实相机门 |
| shm | 9 passed | 9 passed | |
| camera_calib | 28 passed | 28 passed | |
| sim | 23 passed | 23 passed | 需 `PYTHONPATH=src/runtime/sim` |
| **合计** | **199 passed, 1 skipped** | **218 passed, 1 skipped** | |

- 不要用 `-p no:anyio`：当前环境实测它不改变收集结果，且会和 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 的用法混在一起。
- 缺 `rclpy` / `interfaces` 时，`test_planning_node_collision.py` 只跳过该文件（1 skipped），不会中止整个目录。
- 问题清单与优先级见 [todo.md](todo.md)。
