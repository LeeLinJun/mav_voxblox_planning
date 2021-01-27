#include "systems/quadrotor_voxblox.hpp"
#include <iostream>
#include <string>
#include <utilities/debug.hpp>

#include <voxblox_ros/esdf_server.h>

using namespace std;


int main(int argc, char** argv) {
    ros::init(argc, argv, "voxblox_mpc_mpnet_planner");
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");
    FLAGS_alsologtostderr = true;


    std::string input_filepath = "/root/catkin_ws/src/mav_voxblox_planning/voxblox_sst_planner/maps/machine_hall/voxblox/rs_esdf_0.10.voxblox";
    voxblox::EsdfServer voxblox_server_(nh, nh_private);
    
    voxblox::EsdfMap::Ptr esdf_map_ = voxblox_server_.getEsdfMapPtr();
    CHECK(esdf_map_);
    voxblox::TsdfMap::Ptr tsdf_map_ = voxblox_server_.getTsdfMapPtr();
    CHECK(tsdf_map_);
    
    if (!input_filepath.empty()) {
        // Verify that the map has an ESDF layer, otherwise generate it.
        if (!voxblox_server_.loadMap(input_filepath)) {
        ROS_ERROR("Couldn't load ESDF map!");

        // Check if the TSDF layer is non-empty...
        if (tsdf_map_->getTsdfLayerPtr()->getNumberOfAllocatedBlocks() > 0) {
            ROS_INFO("Generating ESDF layer from TSDF.");
            // If so, generate the ESDF layer!

            const bool full_euclidean_distance = true;
            voxblox_server_.updateEsdfBatch(full_euclidean_distance);
        } else {
            ROS_ERROR("TSDF map also empty! Check voxel size!");
        }
        }
    }

    enhanced_system_t* model = new quadrotor_voxblox_t(lower_bound_.x(), upper_bound_.x(),
      lower_bound_.y(), upper_bound_.y(),
      lower_bound_.z(), upper_bound_.z(),
      min_vel_, max_vel_,
      min_omega_, max_omega_,
      min_z_control_, max_z_control_,
      min_rotate_control_, max_rotate_control_,
      2e-3,
      voxblox_server_.getEsdfMapPtr()->getEsdfLayerPtr(),
      0.25));
    
    // initialize cem
    double loss_weights[13] = {1, 1, 1, 
                               0.3, 0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3,
                               0.3, 0.3, 0.3};
    int ns = 1024,
        nt = 5,
        ne = 32,
        max_it = 20;
    double converge_r = 0.1,
           mu_u = 0,
           std_u = 4,
           mu_t = 0.1,
           std_t = 0.2,
           t_max = 0.5,
           dt = 2e-2,
           step_size = 0.5;
   
    const double in_start[13] = {0, 0, 0, 
                          0, 0, 0, 1,
                          0, 0, 0,
                          0, 0, 0,
                          };
    double in_goal[13] = {1, 1, 1, 
                          0, 0, 0, 1,
                          0, 0, 0,
                          0, 0, 0,
                          };;
    double in_radius = 3; 
    
    // Test system propagation
    double const control[4] = {-15, 0, 0, 0};
    double state[13] = {0, 0, 0, 
                        0, 0, 0, 1,
                        0, 0, 0,
                        0, 0, 0};

    check_state_validity(model, state);
    for(unsigned int step = 0; step < 10; step++){
        // check_state_validity(model, state);
        std::cout << model->propagate(state, model->get_state_dimension(), 
                         control, model->get_control_dimension(), 
                         10, state, dt) << "\t";
        print_state(model, state);

    }
    return 0;
}