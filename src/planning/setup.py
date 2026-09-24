from setuptools import find_packages, setup

package_name = "planning"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # 包内数据: 规划配置真值 + 碰撞网格资产 (importlib.resources 按包内路径读取)
    package_data={package_name: ["config/*.yaml", "assets/collision/*.obj"]},
    include_package_data=True,
    # numpy/scipy 为硬依赖; OMPL(type2) 与 hpp-fcl(collision) 是系统级可选依赖,
    # 不经 pip 安装, 未装时仅这两个模块不可用, 其余运动学/type3 正常。
    install_requires=["setuptools", "numpy", "scipy", "pyyaml"],
    zip_safe=True,
    maintainer="sxh",
    maintainer_email="shaozi2233@gmail.com",
    description="纯算法库: 碰撞检测 / 关节路径 (OMPL) / 装配流形约束",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 薄 ROS 节点; 库本体 (planning/*.py 其余文件) 保持零 ROS 依赖
            "planning_node = planning.planning_node:main",
        ],
    },
)
