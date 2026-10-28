# 接近段规划结果
std_msgs/Header header
string request_id                  # 对应请求的 request_id
bool success
string message
float64 cost                   # 终点相对起点的运动代价; 失败为 inf
uint32 repaired_nodes          # best_ik 修复节点数; 接近段恒 0
string error_code              # 失败分类
trajectory_msgs/JointTrajectory trajectory
