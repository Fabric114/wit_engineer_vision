# 装配段请求: 按 path_name 规划一段固定的装配流形路径 (同 RM2026 Type3PlanningRequest)
uint8 PATH_SLIDE=1                 # 插到底 -> 滑动 slide_z_mag
uint8 PATH_P=2                     # 滑动终点 -> p=90°, 同时验证 Q ±90° 可转并返回 Q 缓存

std_msgs/Header header
string request_id                  # 请求方生成 (uuid hex), planning_node 原样抄进结果
geometry_msgs/PoseStamped station_pose_arm_base  # 兑换站位姿 (插入后正解获得), frame_id 必须 arm_base
sensor_msgs/JointState initial_joint_state       # 按 name 取 J1..J6, 多余关节忽略; header.stamp = 关节角采样时刻
uint8 path_name                    # PATH_SLIDE / PATH_P; 其它值判失败 (INVALID_REQUEST)
