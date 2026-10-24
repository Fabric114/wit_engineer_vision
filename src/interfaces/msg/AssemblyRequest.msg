# 装配段请求
std_msgs/Header header
string request_id
sensor_msgs/JointState initial_joint_state       # 按 name 取 J1..J6, 多余关节忽略; header.stamp = 关节角采样时刻
geometry_msgs/PoseStamped station_pose_arm_base  # 兑换站位姿 (插入后正解获得), frame_id 必须 arm_base
AssemblyState goal                 # 目标流形状态
bool use_start                     # false: 起点 = 沿站轴退开 planning.type3.exchange_trajectory.insert_slide_mag 的待插状态; true: 用下面的 start
AssemblyState start                # 起始流形状态 (use_start=true 时生效), 如"已插到底"后接滑动段
