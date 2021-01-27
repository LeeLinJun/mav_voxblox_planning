#include "voxblox_mpc_mpnet_planner.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "voxblox_mpc_mpnet_planner");
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  FLAGS_alsologtostderr = true;

  mav_planning::VoxbloxMPCMPNetPlanner node(nh, nh_private);
  ROS_INFO("Initialized MPCMpnet Planner Voxblox node.");

  ros::spin();
  return 0;
}
