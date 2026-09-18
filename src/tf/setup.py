from setuptools import find_packages, setup

package_name = "tf"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sxh",
    maintainer_email="shaozi2233@gmail.com",
    description="坐标变换: 相机系位姿 -> arm_base 系 ExchangeStationPose + 静态 TF 广播",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_tf_node = tf.vision_tf_node:main",
        ],
    },
)
