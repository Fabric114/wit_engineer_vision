from setuptools import find_packages, setup

package_name = "detector"

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
    description="关键点检测 (OpenVINO): RM2026 yolopose 单阶段 或 yolo+litehrnet 两阶段, 出 KeypointObservationArray",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = detector.detector_node:main",
        ],
    },
)
