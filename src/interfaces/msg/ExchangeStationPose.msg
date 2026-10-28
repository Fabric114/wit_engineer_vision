# 兑换站 (能量单元装配目标) 位姿
# pose = T_arm_base_station: 站体系原点 = ring 点 = 插到底时的 TCP, +x = 退出方向。。
std_msgs/Header header      # frame_id = "arm_base"
geometry_msgs/Pose pose
float32 confidence          
uint8 station_id            
bool valid                  # PnP 收敛且重投影/置信度过门限
