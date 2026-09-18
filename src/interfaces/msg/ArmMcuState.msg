# MCU -> Host: 机械臂当前状态 (仿真后端与真机 firmware bridge 发布同一消息)
std_msgs/Header header
float32[6] joint_pos       # rad
float32[6] joint_vel       # rad/s
uint8 gripper_state        # 0=open 1=closed 2=moving
float32 gripper_force      # N, 无力传感器时为 0
uint16 status_code         # 位标志: 使能/限位/故障
