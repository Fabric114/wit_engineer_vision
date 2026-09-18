from setuptools import find_packages, setup

package_name = "ros2_camera_pkg"

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
    description="相机驱动: 海康(MVS)/大恒(gxipy) 后端可换, 出 /camera/image_raw + /camera/camera_info",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = ros2_camera_pkg.camera_node:main",
        ],
    },
)
