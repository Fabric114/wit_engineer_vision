cmake_minimum_required(VERSION 3.8)
project(interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # 感知层 (由 vision (Python) 发布)
  "msg/KeypointObservation.msg"
  "msg/ExchangeStationPose.msg"
  # MCU <-> Host 契约 (换真机只换底层驱动, 这些 topic 不变)
  "msg/ArmMcuState.msg"
  "msg/ArmHostCommand.msg"
  # 操作手在环
  "msg/OperatorCommand.msg"
  # 规划通道 (request/result)
  "msg/AssemblyState.msg"
  "msg/ApproachRequest.msg"
  "msg/ApproachResult.msg"
  "msg/AssemblyRequest.msg"
  "msg/AssemblyResult.msg"
  DEPENDENCIES std_msgs geometry_msgs trajectory_msgs
)

ament_package()
