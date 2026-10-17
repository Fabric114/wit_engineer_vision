# 规划请求: 从当前关节角规划一条到接近点的无碰撞关节轨迹
std_msgs/Header header
float32[6] start_joint
geometry_msgs/Pose target_pose     # arm_base 系, 兑换站前方接近点
geometry_msgs/Pose station_pose    # arm_base 系, 兑换站体位姿 (约定同 AssemblyRequest.station_pose); 只用来摆碰撞场景, 碰撞开启时必填
float32 timeout                    # 规划超时 (s)
