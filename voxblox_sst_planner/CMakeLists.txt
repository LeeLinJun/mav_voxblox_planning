cmake_minimum_required(VERSION 2.8.3)
project(voxblox_sst_planner)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

###########
# INCLUDE #
###########
include_directories(
    include
    ${OMPL_INCLUDE_DIRS}
 )



#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
  src/voxblox_ompl_sst.cpp
  src/voxblox_sst_planner.cpp
)
target_link_libraries(${PROJECT_NAME} ${OMPL_LIBRARIES} ${OMPLAPP_LIBRARIES})

############
# BINARIES #
############
cs_add_executable(voxblox_sst_planner_node
  src/voxblox_sst_planner_node.cpp
)
target_link_libraries(voxblox_sst_planner_node ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export(LIBRARIES ${OMPL_LIBRARIES})
