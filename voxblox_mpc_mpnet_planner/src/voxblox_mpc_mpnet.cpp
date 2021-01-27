#include <iostream>
#include <fstream>
#include <string>
#include "voxblox_mpc_mpnet.h"

namespace mav_planning {

VoxbloxMPCMPNet::VoxbloxMPCMPNet(const ros::NodeHandle& nh,
                                 const ros::NodeHandle& nh_private)
    : nh_(nh),
    nh_private_(nh_private),
    num_seconds_to_plan_(2.5),
    simplify_solution_(false),
    robot_radius_(0.25),
    verbose_(false),
    optimistic_(false),
    trust_approx_solution_(false),
    lower_bound_(Eigen::Vector3d::Zero()),
    upper_bound_(Eigen::Vector3d::Zero()),
    step_size_(0.02),
    goal_radius_(2.5),

    min_rotate_control_(-1),
    max_rotate_control_(1),
    min_z_control_(-12),
    max_z_control_(-8),
    min_vel_(-1),
    max_vel_(1),
    min_omega_(-1),
    max_omega_(1),

    random_seed_(0),
    sst_delta_near_(0.1),
    sst_delta_drain_(0.1),
    shm_max_step_(100),

    cem_ns_(32), 
    cem_nt_(1),
    cem_ne_(4), 
    cem_max_it_(5),
    cem_converge_r_(1e-10),

    cem_mu_t_(0.25),
    cem_std_t_(0.25),
    cem_t_max_(1),
    cem_mean_controlz_(-10),
    cem_mean_controlr_(0),
    cem_std_controlz_(5),
    cem_std_controlr_(1),
    cem_opt_step_size_(1),

    mpnet_num_sample_(1),
    mpnet_device_id_("cuda:0"),
    mpnet_path_(""),
    mpnet_dnet_path_("") {
  nh_private_.param("robot_radius", robot_radius_, robot_radius_);
  nh_private_.param("num_seconds_to_plan", num_seconds_to_plan_,
                    num_seconds_to_plan_);
  nh_private_.param("simplify_solution", simplify_solution_,
                    simplify_solution_);
  nh_private_.param("trust_approx_solution", trust_approx_solution_,
                    trust_approx_solution_);
  nh_private_.param("verbose", verbose_, verbose_);

  nh_private_.param("step_size", step_size_, step_size_);
  nh_private_.param("goal_radius", goal_radius_, goal_radius_);

  nh_private_.param("min_rotate_control", min_rotate_control_, min_rotate_control_);
  nh_private_.param("max_rotate_control", max_rotate_control_, max_rotate_control_);
  nh_private_.param("min_z_control", min_z_control_, min_z_control_);
  nh_private_.param("max_z_control", max_z_control_, max_z_control_);
  nh_private_.param("min_vel", min_vel_, min_vel_);
  nh_private_.param("max_vel", max_vel_, max_vel_);
  nh_private_.param("min_omega", min_omega_, min_omega_);
  nh_private_.param("max_omega", max_omega_, max_omega_);

  nh_private_.param("random_seed", random_seed_, random_seed_);
  nh_private_.param("sst_delta_near", sst_delta_near_, sst_delta_near_);
  nh_private_.param("sst_delta_drain", sst_delta_drain_, sst_delta_drain_);
  nh_private_.param("shm_max_step", shm_max_step_, shm_max_step_);

  nh_private_.param("cem_ns", cem_ns_, cem_ns_);
  nh_private_.param("cem_nt", cem_nt_, cem_nt_);
  nh_private_.param("cem_ne", cem_ne_, cem_ne_);
  nh_private_.param("cem_max_it", cem_max_it_, cem_max_it_);
  nh_private_.param("cem_converge_r", cem_converge_r_, cem_converge_r_);

  nh_private_.param("cem_mu_t", cem_mu_t_, cem_mu_t_);
  nh_private_.param("cem_std_t", cem_std_t_, cem_std_t_);
  nh_private_.param("cem_t_max", cem_t_max_, cem_t_max_);
  
  nh_private_.param("cem_mean_controlz", cem_mean_controlz_, cem_mean_controlz_);
  nh_private_.param("cem_mean_controlr", cem_mean_controlr_, cem_mean_controlr_);
  nh_private_.param("cem_std_controlz", cem_std_controlz_, cem_std_controlz_);
  nh_private_.param("cem_std_controlr", cem_std_controlr_, cem_std_controlr_);
  nh_private_.param("cem_opt_step_size", cem_opt_step_size_, cem_opt_step_size_);

  nh_private_.param("mpnet_num_sample", mpnet_num_sample_, mpnet_num_sample_);
  nh_private_.param("mpnet_device_id", mpnet_device_id_, mpnet_device_id_);
  nh_private_.param("mpnet_path", mpnet_path_, mpnet_path_);
  nh_private_.param("mpnet_dnet_path", mpnet_dnet_path_, mpnet_dnet_path_);

  distance_computer_ = quadrotor_voxblox_t::distance;

  std::ifstream infile;
  obs_vec_.clear();
  infile.open("/root/catkin_ws/src/mav_voxblox_planning/voxblox_mpc_mpnet_planner/model/dataset/data/voxel.txt");
  
  if(infile.is_open()){
    std::string line;
    while(std::getline(infile, line)){
      std::stringstream ss(line);
      for (int i; ss >> i;) {
        obs_vec_.push_back((float)i);    
        if (ss.peek() == ' ')
            ss.ignore();
      }
    }
  }
  // std::cout << "tensor size:" << obs_vec_.size() << std::endl;
  obs_tensor_ = torch::from_blob(obs_vec_.data(), {1, 20, 100, 100}).to(torch::Device(mpnet_device_id_));
}

void VoxbloxMPCMPNet::setBounds(const Eigen::Vector3d& lower_bound,
                               const Eigen::Vector3d& upper_bound) {
  lower_bound_ = lower_bound;
  upper_bound_ = upper_bound;
}

void VoxbloxMPCMPNet::setTsdfLayer(
    voxblox::Layer<voxblox::TsdfVoxel>* tsdf_layer) {
  tsdf_layer_ = tsdf_layer;
  CHECK_NOTNULL(tsdf_layer_);
  voxel_size_ = tsdf_layer_->voxel_size();
}

void VoxbloxMPCMPNet::setEsdfLayer(
    voxblox::Layer<voxblox::EsdfVoxel>* esdf_layer) {
  esdf_layer_ = esdf_layer;
  CHECK_NOTNULL(esdf_layer_);
  voxel_size_ = esdf_layer_->voxel_size();
}

void VoxbloxMPCMPNet::setupProblem() {
  if (optimistic_) {
    CHECK_NOTNULL(tsdf_layer_);
    ROS_INFO("setting up tsdf layer, this should not be called!!!");
    // problem_setup_.setTsdfVoxbloxCollisionChecking(robot_radius_, tsdf_layer_);
  } else {
    CHECK_NOTNULL(esdf_layer_);
    ROS_INFO("setting up esdf layer");
    // problem_setup_.setEsdfVoxbloxCollisionChecking(robot_radius_, esdf_layer_);
  }
  // problem_setup_.setDefaultObjective();

  // problem_setup_.setDefaultPlanner();
  // // This is a fraction of the space extent! Not actual metric units. For
  // // mysterious reasons. Thanks OMPL!
  // double validity_checking_resolution = 0.01;
  // if ((upper_bound_ - lower_bound_).norm() > 1e-3) {
  //   // If bounds are set, set this to approximately one voxel.
  //   validity_checking_resolution =
  //       voxel_size_ / (upper_bound_ - lower_bound_).norm() / 2.0;
  // }
  // // problem_setup_.setStateValidityCheckingResolution(
  // //     validity_checking_resolution);
}

bool VoxbloxMPCMPNet::getPathBetweenWaypoints(
    const mav_msgs::EigenTrajectoryPoint& start,
    const mav_msgs::EigenTrajectoryPoint& goal,
    mav_msgs::EigenTrajectoryPointVector* solution) {
  setupFromStartAndGoal(start, goal);


  std::vector<std::vector<double>> path;
  std::vector<std::vector<double>> controls; 
  std::vector<double> costs;
  double return_states[13];
  // Solvin' time!
  timer_.reset();
  timer_.measure();
  // std::cout << timer_.measure() << std::endl;

  while(timer_.measure() <= num_seconds_to_plan_) {
    planner_ -> deep_smp_step(system_.get(), step_size_, obs_tensor_, false, 0, false, false, return_states, 0.05);
    for(int i = 0; i<13; i++){
      std::cout << return_states[i] << ",";
    }
    std::cout << std::endl;

    path.clear();
    controls.clear();
    costs.clear();
    planner_ -> get_solution(path, controls, costs);
    if(!path.empty()){
      break;
    }
  }
  if (!path.empty()) {
    solutionPathToTrajectoryPoints(path, solution);
    return true;
  }

  ROS_WARN("MPC-MPNet planning failed.");

  return false;
}

void VoxbloxMPCMPNet::setupFromStartAndGoal(
    const mav_msgs::EigenTrajectoryPoint& start,
    const mav_msgs::EigenTrajectoryPoint& goal) {
      system_.reset(new quadrotor_voxblox_t(
        -20, 5,
        -5, 25,
        0, 2,
        min_vel_, max_vel_,
        min_omega_, max_omega_,
        min_z_control_, max_z_control_,
        min_rotate_control_, max_rotate_control_,
        step_size_,
        esdf_layer_,
        robot_radius_));     

      // MPC-MPNet part
      mpnet_.reset(
        new networks::mpnet_cost_t(
          mpnet_path_, 
          mpnet_dnet_path_, //cost_predictor_weight_path, 
          "", //cost_to_go_predictor_weight_path,
          mpnet_num_sample_,
          mpnet_device_id_,
          0/*refine_lr*/,
          true/*normalize*/)
      );
     
      double cem_mean_control[system_->get_control_dimension()] = {cem_mean_controlz_, cem_mean_controlr_, cem_mean_controlr_, cem_mean_controlr_};
      double cem_std_control[system_->get_control_dimension()] = {cem_std_controlz_, cem_std_controlr_, cem_std_controlr_, cem_std_controlr_};
      double loss_weights[system_->get_state_dimension()];
      for (unsigned int i = 0; i < system_->get_state_dimension(); i++) {
        if(i < 3){
          loss_weights[i] = 1;
        } else {
          loss_weights[i] = 0.1;
        }
      }
      cem_.reset(
        new trajectory_optimizers::CEM(
            system_.get(), cem_ns_, cem_nt_,               
            cem_ne_, cem_converge_r_, 
            cem_mean_control, cem_std_control, 
            cem_mu_t_, cem_std_t_, cem_t_max_, 
            step_size_, loss_weights, cem_max_it_, false, cem_opt_step_size_)
      );

      const double start_state[system_ -> get_state_dimension()] = {
        start.position_W.x(), start.position_W.y(), start.position_W.z(), /*r0*/
        0, 0, 0, 1,/*q0*/ 0, 0, 0,/*v0*/ 0, 0, 0 /*w0*/
      };
      const double goal_state[system_ -> get_state_dimension()] = {
        goal.position_W.x(), goal.position_W.y(), goal.position_W.z(), /*r0*/
        0, 0, 0, 1,/*q0*/ 0, 0, 0,/*v0*/ 0, 0, 0 /*w0*/
      };
      planner_.reset(
        new mpc_mpnet_t(
          start_state, goal_state, goal_radius_,
          system_->get_state_bounds(),
          system_->get_control_bounds(),
          distance_computer_,
          random_seed_,
          sst_delta_near_, sst_delta_drain_, 
          cem_.get(),
          mpnet_.get(),
          1, // Np=1
          shm_max_step_));

}

void VoxbloxMPCMPNet::solutionPathToTrajectoryPoints(
    std::vector<std::vector<double>>& path,
    mav_msgs::EigenTrajectoryPointVector* trajectory_points) const {
  CHECK_NOTNULL(trajectory_points);
  trajectory_points->clear();
  trajectory_points->reserve(path.size());


  for (std::vector<double> state : path) {
    Eigen::Vector3d mav_position(
      state[0], state[1], state[2]);
    mav_msgs::EigenTrajectoryPoint mav_trajectory_point;
    mav_trajectory_point.position_W = mav_position;
    trajectory_points->emplace_back(mav_trajectory_point);
  }
}

}  // namespace mav_planning
