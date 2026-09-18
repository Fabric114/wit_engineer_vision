# 约束装配请求: 在贴合流形上生成插入/滑动/旋转的关节轨迹
std_msgs/Header header
float32[6] start_joint
geometry_msgs/Pose station_pose    # 兑换站位姿 (arm_base 系)
AssemblyState goal                 # 目标流形状态
