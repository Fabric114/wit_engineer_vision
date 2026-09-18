from setuptools import find_packages, setup

package_name = "solver"

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
    description="PnP 解算: 关键点 schema 映射 + solvePnPRansac + 滤波, 出相机系兑换站 6D 位姿",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pnp_node = solver.pnp_node:main",
        ],
    },
)
