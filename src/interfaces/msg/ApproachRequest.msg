# 接近段请求
std_msgs/Header header
string request_id                  # 请求方生成 (uuid hex), planning_node 原样抄进结果
float32[6] start_joint
geometry_msgs/Pose target_pose     # arm_base 系, 兑换站前方接近点
geometry_msgs/Pose station_pose    # arm_base 系, 兑换站体位姿 (约定同 AssemblyRequest.station_pose); 只用来摆碰撞场景, 碰撞开启时必填
float32 timeout                    # 规划超时 (s)
