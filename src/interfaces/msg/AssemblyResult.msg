# 装配段规划结果
std_msgs/Header header
string request_id                  # = 对应请求的 request_id; 请求方只认自己最新发出的那个, 其余丢弃
bool success
string message
uint32 repaired_nodes          # best_ik 修复节点数; 0 = 严格 Viterbi 解
string error_code              # 失败分类; 成功为空串: INVALID_REQUEST / IK_UNREACHABLE / COLLISION / TRAJECTORY_INVALID / PLANNER_FAILED / INTERNAL
float64 cost                   # Viterbi 图代价; 失败为 inf
uint8 path_name                # = 请求的 path_name (失败也回显)
float64 selected_roll          # 终点 roll (rad)
int32 selected_ik_branch       # 所用 IK 分支; 失败为 -1
trajectory_msgs/JointTrajectory trajectory
# Q 缓存: 只有 PATH_P 成功时填。 q_sweep_s = q / 90°, 从 -1 严格递增到 +1;
# q_sweep_trajectory 是查找表不是轨迹: 第 i 点 = q_sweep_s[i] 处的关节角, time_from_start = i 秒,
# 速度/加速度全 0, 不得直接播放 (执行侧按 s 对每个关节插值)。
float64[] q_sweep_s
trajectory_msgs/JointTrajectory q_sweep_trajectory
