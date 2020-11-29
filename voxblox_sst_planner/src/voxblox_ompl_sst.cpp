#include "voxblox_sst_planner/voxblox_ompl_sst.h"

namespace mav_planning {

VoxbloxOmplSst::VoxbloxOmplSst(const ros::NodeHandle& nh,
                               const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      planner_type_(kSst),
      num_seconds_to_plan_(2.5),
      simplify_solution_(true),
      robot_radius_(1.0),
      verbose_(false),
      optimistic_(false),
      trust_approx_solution_(false),
      lower_bound_(Eigen::Vector3d::Zero()),
      upper_bound_(Eigen::Vector3d::Zero()),
      minControlDuration_(20),
      maxControlDuration_(100),
      stepSize_(0.02),
      goalRadius_(2.5),
      
      minRotateControl_(-1),
      maxRotateControl_(1),
      minZControl_(-12),
      maxZControl_(-8),
      minVel_(-1),
      maxVel_(1),
      minOmega_(-1),
      maxOmega_(1) {
  nh_private_.param("robot_radius", robot_radius_, robot_radius_);
  nh_private_.param("num_seconds_to_plan", num_seconds_to_plan_,
                    num_seconds_to_plan_);
  nh_private_.param("simplify_solution", simplify_solution_,
                    simplify_solution_);
  nh_private_.param("trust_approx_solution", trust_approx_solution_,
                    trust_approx_solution_);
  nh_private_.param("verbose", verbose_, verbose_);

  nh_private_.param("minControlDuration", minControlDuration_, 
                    minControlDuration_);
  nh_private_.param("maxControlDuration", maxControlDuration_,
                    maxControlDuration_);
  nh_private_.param("stepSize", stepSize_, stepSize_);
  nh_private_.param("goalRadius", goalRadius_, goalRadius_);

  nh_private_.param("minRotateControl", minRotateControl_, minRotateControl_);
  nh_private_.param("maxRotateControl", maxRotateControl_, maxRotateControl_);
  nh_private_.param("minZControl", minZControl_, minZControl_);
  nh_private_.param("maxZControl", maxZControl_, maxZControl_);
  nh_private_.param("minVel", minVel_, minVel_);
  nh_private_.param("maxVel", maxVel_, maxVel_);
  nh_private_.param("minOmega", minOmega_, minOmega_);
  nh_private_.param("maxOmega", maxOmega_, maxOmega_);


}

void VoxbloxOmplSst::setBounds(const Eigen::Vector3d& lower_bound,
                               const Eigen::Vector3d& upper_bound) {
  lower_bound_ = lower_bound;
  upper_bound_ = upper_bound;
}

void VoxbloxOmplSst::setTsdfLayer(
    voxblox::Layer<voxblox::TsdfVoxel>* tsdf_layer) {
  tsdf_layer_ = tsdf_layer;
  CHECK_NOTNULL(tsdf_layer_);
  voxel_size_ = tsdf_layer_->voxel_size();
}

void VoxbloxOmplSst::setEsdfLayer(
    voxblox::Layer<voxblox::EsdfVoxel>* esdf_layer) {
  esdf_layer_ = esdf_layer;
  CHECK_NOTNULL(esdf_layer_);
  voxel_size_ = esdf_layer_->voxel_size();
}

void VoxbloxOmplSst::setupProblem() {
  if (optimistic_) {
    CHECK_NOTNULL(tsdf_layer_);
    ROS_INFO("setting up tsdf layer");
    problem_setup_.setTsdfVoxbloxCollisionChecking(robot_radius_, tsdf_layer_);
  } else {
    CHECK_NOTNULL(esdf_layer_);
    ROS_INFO("setting up esdf layer");

    problem_setup_.setEsdfVoxbloxCollisionChecking(robot_radius_, esdf_layer_);
  }
  problem_setup_.setDefaultObjective();

  problem_setup_.setDefaultPlanner();

  if (lower_bound_ != upper_bound_) {
    ompl::base::RealVectorBounds r3bounds(3), velbounds(6), controlbounds(4);
    ROS_INFO("Map Lower Bounds: %f\t%f\t%f", 
           lower_bound_.x(),
           lower_bound_.y(),
           lower_bound_.z());
    ROS_INFO("Map Upper Bounds: %f\t%f\t%f", 
           upper_bound_.x(),
           upper_bound_.y(),
           upper_bound_.z());

    r3bounds.setLow(0, lower_bound_.x());
    r3bounds.setHigh(0, upper_bound_.x());
    r3bounds.setLow(1, lower_bound_.y());
    r3bounds.setHigh(1, upper_bound_.y());

    r3bounds.setLow(2, 0.0);
    r3bounds.setHigh(2, 1.75);
    
   
    problem_setup_.getStateSpace()->as<ompl::base::CompoundStateSpace>()->as<ompl::base::SE3StateSpace>(0)->setBounds(r3bounds);

    velbounds.setLow(0, minVel_);
    velbounds.setHigh(0, maxVel_);
    velbounds.setLow(1, minVel_);
    velbounds.setHigh(1, maxVel_);
    velbounds.setLow(2, minVel_);
    velbounds.setHigh(2, maxVel_);

    velbounds.setLow(3, minOmega_);
    velbounds.setHigh(3, maxOmega_);
    velbounds.setLow(4, minOmega_);
    velbounds.setHigh(4, maxOmega_);
    velbounds.setLow(5, minOmega_);
    velbounds.setHigh(5, maxOmega_);

    problem_setup_.getStateSpace()->as<ompl::base::CompoundStateSpace>()->as<ompl::base::RealVectorStateSpace>(1)->setBounds(velbounds);

    controlbounds.setLow(minRotateControl_);
    controlbounds.setHigh(maxRotateControl_);
    controlbounds.setLow(0, minZControl_);
    controlbounds.setHigh(0, maxZControl_);
    problem_setup_.getControlSpace()->as<ompl::control::RealVectorControlSpace>()->setBounds(controlbounds);
    problem_setup_.getSpaceInformation()->setMinMaxControlDuration(minControlDuration_, maxControlDuration_);
    problem_setup_.getSpaceInformation()->setPropagationStepSize(stepSize_);

  }

  // This is a fraction of the space extent! Not actual metric units. For
  // mysterious reasons. Thanks OMPL!
  double validity_checking_resolution = 0.01;
  if ((upper_bound_ - lower_bound_).norm() > 1e-3) {
    // If bounds are set, set this to approximately one voxel.
    validity_checking_resolution =
        voxel_size_ / (upper_bound_ - lower_bound_).norm() / 2.0;
  }
  problem_setup_.setStateValidityCheckingResolution(
      validity_checking_resolution);
}

bool VoxbloxOmplSst::getPathBetweenWaypoints(
    const mav_msgs::EigenTrajectoryPoint& start,
    const mav_msgs::EigenTrajectoryPoint& goal,
    mav_msgs::EigenTrajectoryPointVector* solution) {
  setupFromStartAndGoal(start, goal);

  // Solvin' time!
  if (problem_setup_.solve(num_seconds_to_plan_)) {
    if (problem_setup_.haveExactSolutionPath()) {
      // Simplify and print.
      // TODO(helenol): look more into this. Appears to actually prefer more
      // vertices with presumably shorter total path length, which is
      // detrimental to polynomial planning.
      // if (simplify_solution_) {
      //   problem_setup_.reduceVertices();
      // }
      if (verbose_) {
        problem_setup_.getSolutionPath().printAsMatrix(std::cout);
      }
    } else {
      ROS_WARN("OMPL planning failed.");
      return false;
    }
  }

  if (problem_setup_.haveSolutionPath()) {
    solutionPathToTrajectoryPoints(problem_setup_.getSolutionPath(), solution);
    return true;
  }
  return false;
}

void VoxbloxOmplSst::setupFromStartAndGoal(
    const mav_msgs::EigenTrajectoryPoint& start,
    const mav_msgs::EigenTrajectoryPoint& goal) {
  // if (planner_type_ == kPrm) {
  //   std::dynamic_pointer_cast<ompl::geometric::PRM>(problem_setup_.getPlanner())
  //       ->clearQuery();
  // } else {
    problem_setup_.clear();
  // }

  // ompl::base::ScopedState<ompl::mav::StateSpace> start_ompl(
  //     problem_setup_.getSpaceInformation());
  // ompl::base::ScopedState<ompl::mav::StateSpace> goal_ompl(
  //     problem_setup_.getSpaceInformation());
    ompl::base::ScopedState<ompl::mav::StateSpace> start_ompl(
      problem_setup_.getGeometricComponentStateSpace());
    ompl::base::ScopedState<ompl::mav::StateSpace> goal_ompl(
      problem_setup_.getGeometricComponentStateSpace());
    start_ompl->setX(start.position_W.x());
    start_ompl->setY(start.position_W.y());
    start_ompl->setZ(start.position_W.z());
    start_ompl->rotation().setIdentity();

    goal_ompl->setX(goal.position_W.x());
    goal_ompl->setY(goal.position_W.y());
    goal_ompl->setZ(goal.position_W.z());
    goal_ompl->rotation().setIdentity();
 

    problem_setup_.setStartAndGoalStates(
      problem_setup_.getFullStateFromGeometricComponent(start_ompl),
      problem_setup_.getFullStateFromGeometricComponent(goal_ompl),
      goalRadius_);
    problem_setup_.setup();

    if (verbose_) {
      problem_setup_.print();
      // ros::Duration(3).sleep();
    }

}

void VoxbloxOmplSst::solutionPathToTrajectoryPoints(
    ompl::control::PathControl& path,
    mav_msgs::EigenTrajectoryPointVector* trajectory_points) const {
  CHECK_NOTNULL(trajectory_points);
  trajectory_points->clear();
  trajectory_points->reserve(path.getStateCount());

  std::vector<ompl::base::State*>& state_vector = path.getStates();

  for (ompl::base::State* state_ptr : state_vector) {
    // printf("%f\t%f\t%f\n",
    //   state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>()->getX(),
    //   state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>()->getY(),
    //   state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>()->getZ());
    Eigen::Vector3d mav_position(
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getX(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getY(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getZ());
    mav_msgs::EigenTrajectoryPoint mav_trajectory_point;
    mav_trajectory_point.position_W = mav_position;
    trajectory_points->emplace_back(mav_trajectory_point);
  }
}


bool VoxbloxOmplSst::getBestPathTowardGoal(
    const mav_msgs::EigenTrajectoryPoint& start,
    const mav_msgs::EigenTrajectoryPoint& goal,
    mav_msgs::EigenTrajectoryPoint::Vector* solution) {
  CHECK_NOTNULL(solution);
  solution->clear();
  setupFromStartAndGoal(start, goal);

  // Solvin' time!
  bool solution_found = false;
  solution_found = problem_setup_.solve(num_seconds_to_plan_);
  if (solution_found) {
    if (problem_setup_.haveSolutionPath()) {
      // Simplify and print.
      // if (simplify_solution_) {
      //   problem_setup_.reduceVertices();
      // }
      if (verbose_) {
        problem_setup_.getSolutionPath().printAsMatrix(std::cout);
      }
      solutionPathToTrajectoryPoints(problem_setup_.getSolutionPath(),
                                     solution);
      return true;
    }
  }
  // The case where you actually have a solution path has returned by now.
  // Otherwise let's just see what the best we can do is.
  ompl::base::PlannerData planner_data(problem_setup_.getSpaceInformation());
  problem_setup_.getPlanner()->getPlannerData(planner_data);

  // Start traversing the graph and find the node that gets the closest to the
  // actual goal point.
  if (planner_data.numStartVertices() < 1) {
    ROS_ERROR("No start vertices in SST!");
    return false;
  }

  unsigned int min_index = 0;
  double min_distance = std::numeric_limits<double>::max();

  if (planner_data.numVertices() <= 0) {
    ROS_ERROR("No vertices in SST!");
    return false;
  }

  // Iterate over all vertices. Check which is the closest.
  for (unsigned int i = 0; i < planner_data.numVertices(); i++) {
    const ompl::base::PlannerDataVertex& vertex = planner_data.getVertex(i);
    double distance =
        getDistanceEigenToState(goal.position_W, vertex.getState());

    if (distance < min_distance) {
      min_distance = distance;
      min_index = i;
    }
  }

  unsigned int start_index = planner_data.getStartIndex(0);

  // Get the closest vertex back out, and then get its parents.
  mav_msgs::EigenTrajectoryPointVector trajectory_points;

  unsigned int current_index = min_index;
  while (current_index != start_index) {
    // Put this vertex in.
    const ompl::base::PlannerDataVertex& vertex =
        planner_data.getVertex(current_index);

    const ompl::base::State* state_ptr = vertex.getState();
    Eigen::Vector3d mav_position(
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getX(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getY(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getZ());
    mav_msgs::EigenTrajectoryPoint mav_trajectory_point;
    mav_trajectory_point.position_W = mav_position;
    trajectory_points.emplace_back(mav_trajectory_point);

    std::vector<unsigned int> edges;

    planner_data.getIncomingEdges(current_index, edges);

    if (edges.empty()) {
      break;
    }

    current_index = edges.front();
  }

  // Finally reverse the vector.
  std::reverse(std::begin(trajectory_points), std::end(trajectory_points));

  *solution = trajectory_points;
  return false;
}

double VoxbloxOmplSst::getDistanceEigenToState(
    const Eigen::Vector3d& eigen, const ompl::base::State* state_ptr) {
  Eigen::Vector3d state_pos(
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getX(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getY(),
      state_ptr->as<ompl::base::CompoundState>()->as<ompl::mav::StateSpace::StateType>(0)->getZ());

  return (eigen - state_pos).norm();
}

}  // namespace mav_planning
