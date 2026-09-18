# 兑换站 (能量单元装配目标) 位姿, 已由 PnP 解算并变换到 arm_base 坐标系
std_msgs/Header header      # frame_id = "arm_base"
geometry_msgs/Pose pose
float32 confidence
uint8 station_id
bool valid                  # PnP 是否收敛且通过双目/IMU 校验
