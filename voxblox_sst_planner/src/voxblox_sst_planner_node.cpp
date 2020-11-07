#include "voxblox_sst_planner/voxblox_sst_planner.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "voxblox_sst_planner");
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  FLAGS_alsologtostderr = true;

  mav_planning::VoxbloxSstPlanner node(nh, nh_private);
  ROS_INFO("Initialized SST Planner Voxblox node.");

  ros::spin();
  return 0;
}
