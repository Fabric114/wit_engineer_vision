# 插入时操作手微调指令
std_msgs/Header header
uint8 command              # 0=IDLE 1=START 2=ADVANCE 3=WITHDRAW 4=ADJUST
uint8 level                # 装配档位/预设 (不同兑换站高度或姿态)
float32 dx                 # ADJUST 时的末端微调量 (m), arm_base 系
float32 dy
float32 dz
