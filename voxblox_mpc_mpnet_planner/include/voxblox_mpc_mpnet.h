#ifndef VOXBLOX_MPC_MPNET_PLANNER_VOXBLOX_MPC_MPNET_H_
#define VOXBLOX_MPC_MPNET_PLANNER_VOXBLOX_MPC_MPNET_H_

#include <mav_msgs/conversions.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <ros/ros.h>

#include "utilities/timer.hpp"

#include "systems/quadrotor_voxblox.hpp"
#include "motion_planners/mpc_mpnet.hpp"
#include "networks/mpnet_cost.hpp"

namespace mav_planning {

class VoxbloxMPCMPNet {
 public:
  enum RrtPlannerType {
    kMPNetPath = 0,
    kMPNetTree
  };

  VoxbloxMPCMPNet(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  virtual ~VoxbloxMPCMPNet() {
    system_.reset();
    cem_.reset();
    mpnet_.reset();
    planner_.reset();
  }

  inline void setRobotRadius(double robot_radius) {
    robot_radius_ = robot_radius;
  }
  void setBounds(const Eigen::Vector3d& lower_bound,
                 const Eigen::Vector3d& upper_bound);

  // Both are expected to be OWNED BY ANOTHER OBJECT that shouldn't go out of
  // scope while this object exists.
  void setTsdfLayer(voxblox::Layer<voxblox::TsdfVoxel>* tsdf_layer);
  void setEsdfLayer(voxblox::Layer<voxblox::EsdfVoxel>* esdf_layer);

  inline void setOptimistic(bool optimistic) { optimistic_ = optimistic; }
  bool getOptimistic() const { return optimistic_; }

  double getNumSecondsToPlan() const { return num_seconds_to_plan_; }
  void setNumSecondsToPlan(double num_seconds) {
    num_seconds_to_plan_ = num_seconds;
  }

  RrtPlannerType getPlanner() const { return planner_type_; }
  void setPlanner(RrtPlannerType planner) { planner_type_ = planner; }

  // Only call this once, only call this after setting all settings correctly.
  void setupProblem();

  // Fixed start and end locations, returns list of waypoints between.
  bool getPathBetweenWaypoints(
      const mav_msgs::EigenTrajectoryPoint& start,
      const mav_msgs::EigenTrajectoryPoint& goal,
      mav_msgs::EigenTrajectoryPoint::Vector* solution);

  void solutionPathToTrajectoryPoints(
      std::vector<std::vector<double>>& path,
      mav_msgs::EigenTrajectoryPointVector* trajectory_points) const;

 protected:
  void setupFromStartAndGoal(const mav_msgs::EigenTrajectoryPoint& start,
                             const mav_msgs::EigenTrajectoryPoint& goal);
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Setup the problem in SST.
  RrtPlannerType planner_type_;
  double num_seconds_to_plan_;
  bool simplify_solution_;
  double robot_radius_;
  bool verbose_;

  // Whether the planner is optimistic (true) or pessimistic (false) about
  // how unknown space is handled.
  // Optimistic uses the TSDF for collision checking, while pessimistic uses
  // the ESDF. Be sure to set the maps accordingly.
  bool optimistic_;

  // Whether to trust an approximate solution (i.e., not necessarily reaching
  // the exact goal state).
  bool trust_approx_solution_;

  // Planning bounds, if set.
  Eigen::Vector3d lower_bound_;
  Eigen::Vector3d upper_bound_;

  // NON-OWNED pointers to the relevant layers. TSDF only used if optimistic,
  // ESDF only used if pessimistic.
  voxblox::Layer<voxblox::TsdfVoxel>* tsdf_layer_;
  voxblox::Layer<voxblox::EsdfVoxel>* esdf_layer_;

  double voxel_size_;

  int min_control_duration_;
  int max_control_duration_;
  double step_size_;
  double goal_radius_;

  double min_rotate_control_;
  double max_rotate_control_;
  double min_z_control_;
  double max_z_control_;
  double min_vel_;
  double max_vel_;
  double min_omega_;
  double max_omega_;

  int random_seed_;
  double sst_delta_near_;
  double sst_delta_drain_;
  int shm_max_step_;

  std::unique_ptr<enhanced_system_t> system_;
  std::function<double(const double*, const double*, unsigned int)> distance_computer_;
  std::unique_ptr<trajectory_optimizers::CEM> cem_;
  std::unique_ptr<networks::mpnet_cost_t> mpnet_;
  std::unique_ptr<mpc_mpnet_t> planner_;
  torch::Tensor obs_tensor;
  torch::NoGradGuard no_grad;

  sys_timer_t timer_;

  int cem_ns_, cem_nt_, cem_ne_, cem_max_it_;
  double cem_converge_r_, cem_mu_t_, cem_std_t_, cem_t_max_,
    cem_mean_controlz_, cem_mean_controlr_,
    cem_std_controlz_, cem_std_controlr_,
    cem_opt_step_size_;

  int mpnet_num_sample_;
  std::string mpnet_device_id_, 
    mpnet_path_,
    mpnet_dnet_path_;
  std::vector<float> obs_vec_;

  torch::Tensor obs_tensor_;
  
};

}  // namespace mav_planning

#endif  