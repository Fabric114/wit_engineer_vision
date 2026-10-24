# 接近段请求
std_msgs/Header header
string request_id                  # 请求方生成 (uuid hex), planning_node 原样抄进结果
sensor_msgs/JointState initial_joint_state       # 按 name 取 J1..J6, 多余关节忽略; header.stamp = 关节角采样时刻
geometry_msgs/Pose target_pose     # arm_base 系, 兑换站前方接近点
geometry_msgs/PoseStamped station_pose_arm_base  # 兑换站体位姿, frame_id 必须 arm_base; 摆碰撞场景用
float32 timeout                    # 规划超时 (s)
