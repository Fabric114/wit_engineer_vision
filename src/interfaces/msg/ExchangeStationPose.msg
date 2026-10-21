# 兑换站 (能量单元装配目标) 位姿, 已由 PnP 解算并变换到 arm_base 坐标系。
# pose = T_arm_base_station: 站体系原点 = ring 点 = 插到底时的 TCP, +x = 退出方向。
# 视觉链上 station_model 与 station_assembly 是同一个系 (2026-10-20 实测, 见
# config/keypoint_schema.yaml 注释与 docs/2026-10-20-workspace-audit.md §4),
# 因此这里不再区分两个子坐标系。
std_msgs/Header header      # frame_id = "arm_base"
geometry_msgs/Pose pose
float32 confidence          # 目前只有 sim 填 (真值 1.0), 无消费者
uint8 station_id            # 目前只有 sim 填 0, 无消费者
bool valid                  # PnP 收敛且重投影/置信度过门限 (sim 恒 true)
