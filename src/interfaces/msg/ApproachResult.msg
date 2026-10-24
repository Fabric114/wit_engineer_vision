# 接近段规划结果
std_msgs/Header header
string request_id                  # = 对应请求的 request_id; 请求方只认自己最新发出的那个, 其余丢弃
bool success
string message
uint32 repaired_nodes          # best_ik 修复节点数; 接近段恒 0
string error_code              # 失败分类; 成功为空串: INVALID_REQUEST / IK_UNREACHABLE / COLLISION / TRAJECTORY_INVALID / PLANNER_FAILED / INTERNAL
trajectory_msgs/JointTrajectory trajectory
