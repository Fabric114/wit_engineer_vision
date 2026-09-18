"""关键点 <-> 3D 模型点映射 (数据驱动 schema).

按 class_id 取该类用到的关键点子集, 再按名字查 3D 点 —— 名字来自模型
metadata.yaml 的 kpt_names, 3D 点来自 config/keypoint_schema.yaml。

参考: BIT pnp_solver_pkg/keypoint_mapping.py
"""
