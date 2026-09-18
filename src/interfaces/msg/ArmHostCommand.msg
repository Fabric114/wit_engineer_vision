# Host -> MCU: 关节指令
std_msgs/Header header
float32[6] joint_target    # rad
float32[6] joint_vel_ff    # rad/s, 前馈, 可为 0
uint8 gripper_cmd          # 0=open 1=close 2=hold
uint16 control_mode        # 0=idle 1=position 2=trajectory
