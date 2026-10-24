# 接近段请求: 目标 = 沿站轴退开 insert_slide_mag 的待插位, 由 planning_node 扫 roll 自选 (同 RM2026)
std_msgs/Header header
string request_id                  # 请求方生成 (uuid hex), planning_node 原样抄进结果
sensor_msgs/JointState initial_joint_state       # 按 name 取 J1..J6, 多余关节忽略; header.stamp = 关节角采样时刻
geometry_msgs/PoseStamped station_pose_arm_base  # 兑换站体位姿, frame_id 必须 arm_base; 摆碰撞场景用
