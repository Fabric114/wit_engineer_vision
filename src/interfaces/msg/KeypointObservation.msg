# 视觉检测出的关键点观测 (2D 像素 + 置信度), 数据驱动: schema 决定点的语义
std_msgs/Header header
string schema          # 对应 pnp_solver 包 config/keypoint_schema.yaml 里的一个条目, 如 "exchange_station"
                       # 未来加 "launcher" 只需新增 schema + 权重, 不改消息
uint16 class_id
float32[] u            # 像素 x
float32[] v            # 像素 y
float32[] confidence   # 每个关键点置信度
