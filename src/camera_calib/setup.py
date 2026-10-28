from setuptools import find_packages, setup

package_name = "camera_calib"

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
    description="相机内参标定与 camera->arm_base 离线手眼外参标定",
    license="MIT",
    entry_points={
        "console_scripts": [
            "intrinsic_calib_node = camera_calib.intrinsic_calib_node:main",
            "offline_calib = camera_calib.offline_calib:main",
            "hand_eye_calib = camera_calib.hand_eye_calib:main",
        ],
    },
)
