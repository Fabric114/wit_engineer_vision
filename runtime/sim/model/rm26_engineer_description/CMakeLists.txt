
cmake_minimum_required(VERSION 3.8)

project(rm26_engineer_description)

# 使用 ROS 2 ament_cmake 构建系统
find_package(ament_cmake REQUIRED)

# 安装 URDF、STL 模型及配置文件
install(
  DIRECTORY
    urdf
    meshes
    config
    launch
  DESTINATION share/${PROJECT_NAME}
)

# 注册 ROS 2 Package
ament_package()
