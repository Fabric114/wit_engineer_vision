# 装配段请求
std_msgs/Header header
string request_id                  # 请求方生成 (uuid hex), planning_node 原样抄进结果
float32[6] start_joint
geometry_msgs/Pose station_pose    # 兑换站位姿 (arm_base 系)
AssemblyState goal                 # 目标流形状态
bool use_start                     # false: 起点 = 沿站轴退开 planning.type3.exchange_trajectory.insert_slide_mag 的待插状态; true: 用下面的 start
AssemblyState start                # 起始流形状态 (use_start=true 时生效), 如"已插到底"后接滑动段
