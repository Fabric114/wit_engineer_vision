from glob import glob
from setuptools import find_packages, setup

package_name = "sim"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/model", glob("model/*.xml")),
        # scene.xml 会 include 描述文件, 描述文件又引 mesh, 三者都得装进 share
        ("share/" + package_name + "/model/rm26_engineer_description",
         glob("model/rm26_engineer_description/*.xml")),
        ("share/" + package_name + "/model/rm26_engineer_description/meshes",
         glob("model/rm26_engineer_description/meshes/*")),
    ],
    install_requires=["setuptools", "mujoco", "numpy"],
    zip_safe=True,
    maintainer="sxh",
    maintainer_email="shaozi2233@gmail.com",
    description="MuJoCo 仿真后端",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = sim.sim_node:main",
        ],
    },
)
