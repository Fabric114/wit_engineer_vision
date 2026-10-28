# 视觉检测出的关键点观测
std_msgs/Header header
string schema          
uint16 class_id
float32[] u            # 像素 x
float32[] v            # 像素 y
float32[] confidence   # 每个关键点置信度
