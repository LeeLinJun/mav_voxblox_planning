cmake_minimum_required(VERSION 2.8.3)
project(voxblox_mpc_mpnet_planner) # project(voxblox_mpc_mpnet_planner LANGUAGES CXX CUDA)


find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

# list(APPEND CMAKE_PREFIX_PATH /root/catkin_ws/src/mav_voxblox_planning/voxblox_mpc_mpnet_planner/deps/libtorch)
set(TORCH_LIB "/root/catkin_ws/src/mav_voxblox_planning/voxblox_mpc_mpnet_planner/deps/libtorch1.7")
set(CMAKE_PREFIX_PATH "${TORCH_LIB}/libtorch")
# list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB}/libtorch")


find_package(Torch REQUIRED)
# SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_definitions(-std=c++14)
add_definitions(-O3)

include_directories(
    "include"
    ## torch
    ${TORCH_INCLUDE_DIRS}
    )

set(PLANNING_UTILS
    src/nearest_neighbors/graph_nearest_neighbors.cpp
    src/utilities/timer.cpp
    src/utilities/random.cpp
)

# for mpc and cem
set(CEM_MPC_SOURCE
    src/systems/quadrotor_voxblox.cpp
    src/trajectory_optimizers/cem.cpp
    )

set(MPC_MPNET_SOURCE
    ${PLANNING_UTILS}
    ${CEM_MPC_SOURCE}
    src/motion_planners/sst.cpp
    src/networks/mpnet.cpp
    src/networks/mpnet_cost.cpp
    src/motion_planners/mpc_mpnet.cpp

    src/voxblox_mpc_mpnet.cpp
    src/voxblox_mpc_mpnet_planner.cpp)

cs_add_library(${PROJECT_NAME}
    ${MPC_MPNET_SOURCE}
)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})


cs_add_executable(voxblox_mpc_mpnet_node
    src/mpc_mpnet_planner_node.cpp
    )
target_link_libraries(voxblox_mpc_mpnet_node ${PROJECT_NAME})

# cs_add_executable(test_quadrotor_voxblox
#     src/systems/quadrotor_voxblox.cpp
#     src/tests/test_quadrotor_voxblox.cpp
#     )

cs_install()
cs_export()
