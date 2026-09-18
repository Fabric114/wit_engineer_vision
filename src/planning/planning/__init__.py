"""planning: 运动规划纯算法库 (零 ROS 依赖), 重构自 RM2026 arm_exchange_core。

子模块 (待按 RM2026 填充):
- collision: 碰撞检测 (python-fcl/hpp-fcl), 加载臂+场地+兑换站几何, is_valid(q)。
- joint_path: 接近段无碰撞规划 (OMPL BIT*), 起止关节角 -> 关节路径。
- assembly_manifold: 约束装配段, AssemblyState -> 贴合面末端位姿序列 -> IK -> 关节轨迹。

供 arm_host/planning_node 调用; 依赖 task 的运动学 (IK/FK)。
"""
