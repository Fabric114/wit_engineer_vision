# 装配段请求
uint8 PATH_SLIDE=1                 # 插到底 -> 滑动 slide_z_mag
uint8 PATH_P=2                     # 滑动终点 -> p=90°, 同时验证 Q ±90° 可转并返回 Q 缓存

std_msgs/Header header
string request_id
geometry_msgs/PoseStamped station_pose_arm_base  # 兑换站位姿 (插入后正解获得), frame_id 必须 arm_base
sensor_msgs/JointState initial_joint_state       # J1-J6时间戳header.stamp
uint8 path_name                    # PATH_SLIDE / PATH_P
