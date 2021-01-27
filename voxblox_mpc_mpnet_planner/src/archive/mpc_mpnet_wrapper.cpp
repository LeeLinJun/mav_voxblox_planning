#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "systems/cart_pole_obs.hpp"
#include "systems/two_link_acrobot_obs.hpp"
#include "systems/quadrotor_obs.hpp"
#include "systems/car_obs.hpp"

#include "trajectory_optimizers/cem.hpp"
#include "trajectory_optimizers/cem_cuda_cartpole.hpp"
#include "trajectory_optimizers/cem_cuda_acrobot.hpp"
#include "trajectory_optimizers/cem_cuda_car.hpp"
#include "trajectory_optimizers/cem_cuda_quadrotor.hpp"


#include "networks/mpnet.hpp"
#include "networks/mpnet_cost.hpp"

// #include "motion_planners/deep_smp_mpc_sst.hpp"
#include "motion_planners/mpc_mpnet.hpp"

#include "cstdio"

namespace pybind11 {
    template <typename T>
    using safe_array = typename pybind11::array_t<T, pybind11::array::c_style>;
}

namespace py = pybind11;
using namespace pybind11::literals;

class __attribute__ ((visibility ("hidden"))) MPCMPNetWrapper{
    
public:

	/**
	 * @brief Python wrapper of DSSTMPC planner Constructor
	 * @details Python wrapper of DSSTMPC planner Constructor
	 *
	 */
    MPCMPNetWrapper(
            std::string system_string,
            std::string solver_type,
            const py::safe_array<double> &start_state_array,
            const py::safe_array<double> &goal_state_array,
            double goal_radius,
            unsigned int random_seed,
            double sst_delta_near,
            double sst_delta_drain,
            const py::safe_array<double> &obs_list_array,
            double width,
            bool verbose,
            std::string mpnet_weight_path, std::string cost_predictor_weight_path,
            std::string cost_to_go_predictor_weight_path,
            int num_sample,
            int shm_max_step,
            int np, int ns, int nt, int ne, int max_it,
            double converge_r, const py::safe_array<double> mu_u, const py::safe_array<double> std_u, double mu_t, double std_t, double t_max, double step_size, double integration_step,
            std::string device_id, float refine_lr, bool normalize,
            py::safe_array<double>& weights_array,
            py::safe_array<double>& obs_voxel_array
    ){
        system_type = system_string;
        auto start_state = start_state_array.unchecked<1>();
        auto goal_state = goal_state_array.unchecked<1>();
        // auto py_obs_list = obs_list_array.unchecked<2>();
        auto obs_voxel_data = obs_voxel_array.unchecked<1>();
        // std::vector<float> obs_vec;
        // if (obs_list_array.shape()[0] == 0) {
        //     throw std::runtime_error("Should contain at least one obstacles.");
        // }
        // if (width <= 0.) {
        //     throw std::runtime_error("obstacle width should be non-negative.");
        // }

        // if (system_type == "acrobot_obs" || system_type == "cartpole_obs" || system_type == "car_obs"){
        //     // Convert obs_pixels to tensor
            
        //     if (obs_list_array.shape()[1] != 2) {
        //         throw std::runtime_error("Shape of the obstacle input should be (N,2).");
        //     }
            
        //     // initialize the array
        //     obs_list = std::vector<std::vector<double>>(obs_list_array.shape()[0], std::vector<double>(2, 0.0));
        //     for (unsigned int i = 0; i < obs_list.size(); i++) {
        //         obs_list[i][0] = py_obs_list(i, 0);
        //         obs_list[i][1] = py_obs_list(i, 1);
        //     }

        //     for (unsigned i = 0; i < obs_voxel_data.shape(0); i++)
        //     {
        //         obs_vec.push_back(float(obs_voxel_data(i)));
        //     }
        //     obs_tensor = torch::from_blob(obs_vec.data(), {1, 1, 32, 32}).to(torch::Device(device_id));

            
        // } else if (system_type == "quadrotor_obs") {
        //     if (obs_list_array.shape()[1] != 3) {
        //         throw std::runtime_error("Shape of the obstacle input should be (N,3).");
        //     }
        //     // Convert obs_voxels to tensors
        //     // printf("%d, %d", obs_list_array.shape()[0], obs_list_array.shape()[1]);
        //     obs_list = std::vector<std::vector<double>>(obs_list_array.shape()[0], std::vector<double>(3, 0.0));
        //     for (unsigned int i = 0; i < obs_list.size(); i++) {
        //         obs_list[i][0] = py_obs_list(i, 0);
        //         obs_list[i][1] = py_obs_list(i, 1);
        //         obs_list[i][2] = py_obs_list(i, 2);
        //         // printf("%f, %f, %f\n", obs_list[i][0], obs_list[i][1], obs_list[i][2]);
        //     }
            
        //     // TODO: pass from para
        //     for (unsigned i = 0; i < obs_voxel_data.shape(0); i++)
        //     {
        //         obs_vec.push_back(float(obs_voxel_data(i)));
        //     }
        //     obs_tensor = torch::from_blob(obs_vec.data(), {1, 32, 32, 32}).to(torch::Device(device_id));

        //     // obs_tensor = torch::zeros({1,32,32,32}).to(torch::Device(device_id));
        // } else 
        if (system_type == "quadrotor_voxblox"){
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
        } else {

            throw std::runtime_error("undefined system");
        }      
       
        // if (system_type == "acrobot_obs"){
        //     system = new two_link_acrobot_obs_t(obs_list, width);
        //     distance_computer = two_link_acrobot_obs_t::distance;

        // } else if (system_type == "cartpole_obs"){
        //     system = new cart_pole_obs_t(obs_list, width);
        //     distance_computer = cart_pole_obs_t::distance;
        // } else if (system_type == "car_obs"){
        //     system = new car_obs_t(obs_list, width);
        //     distance_computer = car_obs_t::distance;

        // } else 
        // if (system_type == "quadrotor_obs") {
        //     system = new quadrotor_obs_t(obs_list, width);
        //     distance_computer = quadrotor_obs_t::distance;
        // } else 
        if (system_type == "quadrotor_voxblox"){
            system = new quadrotor_voxblox_t(voxblox_server_.getEsdfMapPtr()->getEsdfLayerPtr());
        }
         else {
            throw std::runtime_error("undefined system");
        }
        // std::cout <<system_type<<std::endl;
        dt = integration_step;

        auto py_weight_array = weights_array.unchecked<1>();
        loss_weights = new double[system->get_state_dimension()]();
        if(weights_array.shape()[0] > 0){
            for (unsigned int i = 0; i < weights_array.shape()[0]; i++) {
                loss_weights[i] = py_weight_array(i);
            }
        } else {
            for (unsigned int i = 0; i < system->get_state_dimension(); i++){
                loss_weights[i] = 1;
            }
        }
        mpnet.reset(
            new networks::mpnet_cost_t(mpnet_weight_path, 
            cost_predictor_weight_path, 
            cost_to_go_predictor_weight_path,
            num_sample, device_id, refine_lr, normalize)
            //  new networks::mpnet_t(mpnet_weight_path)
        );
        // std::cout <<"network"<<std::endl;

        double *mean_control = new double[system->get_control_dimension()]();
        double *std_control = new double[system->get_control_dimension()]();
        auto mean_control_ref = mu_u.unchecked<1>();
        auto std_control_ref = std_u.unchecked<1>();

        for(unsigned int ui = 0; ui < system->get_control_dimension(); ui++){
            mean_control[ui] = mean_control_ref[ui];
            std_control[ui] = std_control_ref[ui];
        }
        // if (solver_type == "cem_cuda")
        // {
        //     // see what system we are using
        //     if (system_type == "acrobot_obs")
        //     {
        //         cem.reset(
        //             new trajectory_optimizers_acrobot::CEM_CUDA_acrobot(
        //                 system, np, ns, nt,               
        //                 ne, converge_r, obs_list, 
        //                 mean_control, std_control, 
        //                 mu_t, std_t, t_max, 
        //                 dt, loss_weights, max_it, verbose, step_size)
        //         );
        //     }
        //     else if (system_type == "cartpole_obs")
        //     {
        //         cem.reset(
        //             new trajectory_optimizers_cartpole::CEM_CUDA_cartpole(
        //                 system, np, ns, nt,               
        //                 ne, converge_r, obs_list, 
        //                 mean_control, std_control, 
        //                 mu_t, std_t, t_max, 
        //                 dt, loss_weights, max_it, verbose, step_size)
        //         );
        //     }
        //     else if (system_type == "car_obs")
        //     {
        //         cem.reset(
        //             new trajectory_optimizers_car::CEM_CUDA_car(
        //                 system, np, ns, nt,               
        //                 ne, converge_r, obs_list, 
        //                 mean_control, std_control, 
        //                 mu_t, std_t, t_max, 
        //                 dt, loss_weights, max_it, verbose, step_size)
        //         );
        //     }
        //     else if (system_type == "quadrotor_obs")
        //     {
        //         cem.reset(
        //             new trajectory_optimizers_quadrotor::CEM_CUDA_quadrotor(
        //                 system, np, ns, nt,               
        //                 ne, converge_r, obs_list, 
        //                 mean_control, std_control, 
        //                 mu_t, std_t, t_max, 
        //                 dt, loss_weights, max_it, verbose, step_size)
        //         );
        //     }

        // }
        
        // else 
        if (solver_type == "cem")
        {
             cem.reset(
                 new trajectory_optimizers::CEM(
                     system, ns, nt,               
                     ne, converge_r, 
                     mean_control, std_control, 
                     mu_t, std_t, t_max, 
                     dt, loss_weights, max_it, verbose, step_size)
             );
         } else {
            throw std::runtime_error("unknown solver, support solvers: [\"cem_cuda\", \"cem\"]");
         }
        // std::cout <<"cem"<<std::endl;
       
        planner.reset(
            new mpc_mpnet_t(
                    &start_state(0), &goal_state(0), goal_radius,
                    system->get_state_bounds(),
                    system->get_control_bounds(),
                    distance_computer,
                    random_seed,
                    sst_delta_near, sst_delta_drain, 
                    cem.get(),
                    mpnet.get(),
                    np,
                    shm_max_step
                    )
        );

        delete mean_control;
        delete std_control;
    }
    ~MPCMPNetWrapper(){
        delete system;
        cem.reset();
        mpnet.reset();
        planner.reset();
    }

    py::object neural_step(bool refine, float refine_threshold, bool using_one_step_cost,
        bool cost_reselection, double goal_bias) {
        
        //std::cout << "vector to torch obs vector.." << std::endl;
        double* return_states = new double[system->get_state_dimension()*3]();
        planner -> neural_step(system, dt, obs_tensor, refine, refine_threshold, using_one_step_cost, cost_reselection, return_states, goal_bias);
        py::safe_array<double> terminal_array({system->get_state_dimension()*3}, return_states);
        return terminal_array;
    }
    
    py::object deep_smp_step(bool refine, float refine_threshold, bool using_one_step_cost,
        bool cost_reselection, double goal_bias, int NP) {
        
        //std::cout << "vector to torch obs vector.." << std::endl;
        double* return_states = new double[system->get_state_dimension()*3]();
        if(NP > 1){
            planner -> deep_smp_step(system, dt, obs_tensor, refine, refine_threshold, using_one_step_cost, cost_reselection, return_states, goal_bias, NP);

        } else {
            planner -> deep_smp_step(system, dt, obs_tensor, refine, refine_threshold, using_one_step_cost, cost_reselection, return_states, goal_bias);
        }
        py::safe_array<double> terminal_array({system->get_state_dimension()*3}, return_states);
        return terminal_array;
    }

    py::object neural_step_batch(bool refine, float refine_threshold, bool using_one_step_cost,
        bool cost_reselection, double goal_bias, const int NP) {
        
        //std::cout << "vector to torch obs vector.." << std::endl;
        double* return_states = new double[NP*system->get_state_dimension()*3]();
        planner -> neural_step_batch(system, dt, obs_tensor, refine, refine_threshold, using_one_step_cost, cost_reselection, return_states, goal_bias, NP);
        py::safe_array<double> terminal_array({NP, (int) system->get_state_dimension(), 3});

        //std::cout << "inside deep_smp_wrapper::neural_step_batch. before copying to terminal_array_ref" << std::endl;
        auto terminal_array_ref = terminal_array.mutable_unchecked<3>();
        for (unsigned pi = 0; pi < NP; pi++)
        {
            for (unsigned si = 0; si < system->get_state_dimension(); si++)
            {
                terminal_array_ref(pi, si, 0) = return_states[(pi*system->get_state_dimension()+si)*3];
                terminal_array_ref(pi, si, 1) = return_states[(pi*system->get_state_dimension()+si)*3+1];
                terminal_array_ref(pi, si, 2) = return_states[(pi*system->get_state_dimension()+si)*3+2];
            }
        }
        delete return_states;
        return terminal_array;
    }

    py::object deep_smp_step_batch(bool refine, float refine_threshold, bool using_one_step_cost,
        bool cost_reselection, double goal_bias, int NP) {
        
        //std::cout << "vector to torch obs vector.." << std::endl;
        double* return_states = new double[NP *system->get_state_dimension()*3]();
       
        planner -> deep_smp_step_batch(system, dt, obs_tensor, refine, refine_threshold, using_one_step_cost, cost_reselection, return_states, goal_bias, NP);
        
        py::safe_array<double> terminal_array({NP, (int) system->get_state_dimension(), 3});

        //std::cout << "inside deep_smp_wrapper::neural_step_batch. before copying to terminal_array_ref" << std::endl;
        auto terminal_array_ref = terminal_array.mutable_unchecked<3>();
        for (unsigned pi = 0; pi < NP; pi++)
        {
            for (unsigned si = 0; si < system->get_state_dimension(); si++)
            {
                terminal_array_ref(pi, si, 0) = return_states[(pi*system->get_state_dimension()+si)*3];
                terminal_array_ref(pi, si, 1) = return_states[(pi*system->get_state_dimension()+si)*3+1];
                terminal_array_ref(pi, si, 2) = return_states[(pi*system->get_state_dimension()+si)*3+2];
            }
        }
        
        delete return_states;
        return terminal_array;
    }

    // /**
	//  * @copydoc planner_t::step()
	//  */
    void step(int min_time_steps, int max_time_steps, double integration_step) {
        planner->step(system, min_time_steps, max_time_steps, integration_step);
    }

    void mpc_step(double integration_step) {
        planner->mpc_step(system, integration_step);
    }

   py::object steer(const py::safe_array<double> &start_array,
                     const py::safe_array<double> &sample_array){

        auto start_state_ref = start_array.unchecked<1>();
        auto sample_array_ref = sample_array.unchecked<1>();

        double* start = new double[system->get_state_dimension()]();
        double* sample = new double[system->get_state_dimension()]();

        for(int si = 0; si < system->get_state_dimension(); si++){
            start[si] = start_state_ref[si];
            sample[si] = sample_array_ref[si];
        }

        double *terminal_state = new double[system->get_state_dimension()]();

        planner->steer(system, start, sample, 
                       terminal_state, dt);
        py::safe_array<double> terminal_array({start_array.shape()[0]}, terminal_state);

        delete start;
        delete sample;
        return terminal_array;

    }

    py::object steer_sst(int min_time_steps, int max_time_steps, double integration_step){


        double* steer_start = new double[system->get_state_dimension()]();
        double* steer_end = new double[system->get_state_dimension()]();

        planner->step_with_output(system, min_time_steps, max_time_steps, integration_step, steer_start, steer_end);

        py::safe_array<double> py_steer_start({system->get_state_dimension()}, steer_start);
        py::safe_array<double> py_steer_end({system->get_state_dimension()}, steer_end);
        auto py_steer_start_ref = py_steer_start.mutable_unchecked<1>();
        auto py_steer_end_ref = py_steer_end.mutable_unchecked<1>();
        
        for (unsigned i=0; i<system->get_state_dimension(); i++)
        {
             py_steer_start_ref(i) = steer_start[i];
             py_steer_end_ref(i) = steer_end[i];
        }

        delete steer_start;
        delete steer_end;
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>>
            (py_steer_start, py_steer_end));
    }


    py::object steer_batch(const py::safe_array<double> &start_array,
                     const py::safe_array<double> &sample_array, const int NP){

        auto start_state_ref = start_array.unchecked<2>();
        auto sample_array_ref = sample_array.unchecked<2>();

        double* start = new double[NP*system->get_state_dimension()]();
        double* sample = new double[NP*system->get_state_dimension()]();

        for (unsigned pi = 0; pi < NP; pi++)
        {
            for(int si = 0; si < system->get_state_dimension(); si++){
                start[pi*system->get_state_dimension()+si] = start_state_ref(pi, si);
                sample[pi*system->get_state_dimension()+si] = sample_array_ref(pi, si);
            }

        }

        double *terminal_state = new double[NP*system->get_state_dimension()]();

        double* duration = new double[NP]();
        planner->steer_batch(system, start, sample, 
                       terminal_state, dt, NP, duration);
        py::safe_array<double> terminal_array({start_array.shape()[0], start_array.shape()[1]});
        auto terminal_array_ref = terminal_array.mutable_unchecked<2>();
        std::cout<<"running"<<std::endl;
        for (unsigned pi = 0; pi < NP; pi ++)
        {
            for (unsigned si = 0; si < system->get_state_dimension(); si++)
            {
                terminal_array_ref(pi, si) = terminal_state[pi*system->get_state_dimension() + si];
            }
        }
        delete terminal_state;
        delete duration;

        return terminal_array;

    }

    py::object neural_sample(const py::safe_array<double> &state_array, bool refine, float refine_threshold, 
        bool using_one_step_cost, bool cost_reselection){

            auto state_ref = state_array.unchecked<1>();
            double * state = new double[system->get_state_dimension()]();
            for(int si = 0; si < system->get_state_dimension(); si++){
                state[si] = state_ref[si];
            }

            double* neural_sample_state = new double[system->get_state_dimension()]();
            planner->neural_sample(system, 
                                state, 
                                neural_sample_state, 
                                obs_tensor, 
                                refine, 
                                refine_threshold, 
                                using_one_step_cost, 
                                cost_reselection);
            py::safe_array<double> neural_sample_state_array({system->get_state_dimension()}, neural_sample_state);
            delete state;
        
            return neural_sample_state_array;
        }

    py::object neural_sample_batch(const py::safe_array<double> &state_array, bool refine, float refine_threshold, 
        bool using_one_step_cost, bool cost_reselection, const int NP){

            auto state_ref = state_array.unchecked<1>();
            double * state = new double[system->get_state_dimension()]();
            for(int si = 0; si < system->get_state_dimension(); si++){
                state[si] = state_ref[si];
            }

            double* neural_sample_state = new double[NP*system->get_state_dimension()]();
            planner->neural_sample_batch(system, 
                                         state, 
                                         neural_sample_state, 
                                         obs_tensor, 
                                         refine, 
                                         refine_threshold, 
                                         using_one_step_cost, 
                                         cost_reselection,
                                         NP);
 
        int state_dim = system->get_state_dimension();                               
        py::safe_array<double> neural_sample_state_array({NP, state_dim});
        auto neural_sample_state_array_ref = neural_sample_state_array.mutable_unchecked<2>();
        for (unsigned pi = 0; pi < NP; pi ++)
        {
            for (unsigned si = 0; si < system->get_state_dimension(); si ++)
            {
                neural_sample_state_array_ref(pi, si) = neural_sample_state[pi*system->get_state_dimension() + si];
            }

        }

        return neural_sample_state_array;
        }

    /**
    * @copydoc sst_backend_t::nearest_vertex()
    */
    py::object nearest_vertex(const py::safe_array<double> &sample_state_array){
        auto sample_state = sample_state_array.unchecked<1>();
        nearest = this -> planner -> nearest_vertex(&sample_state(0));
        const double* nearest_point = nearest -> get_point();
        
        py::safe_array<double> nearest_array({sample_state_array.shape()[0]});
        auto state_ref = nearest_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < sample_state_array.shape()[0]; i++){
            state_ref(i) = nearest_point[i];
        }
        return nearest_array;
    }

    void add_to_tree(const py::safe_array<double> &sample_state_array,
        double duration
    ){
        auto sample_state = sample_state_array.unchecked<1>();
        const double* sample_control = new double[planner -> get_control_dimension()];
        planner -> add_to_tree(&sample_state(0), sample_control, nearest, duration);       

    }
    /**
	 * @copydoc planner_t::get_solution()
	 */
    py::object get_solution() {
        std::vector<std::vector<double>> solution_path;
        std::vector<std::vector<double>> controls;
        std::vector<double> costs;
        planner->get_solution(solution_path, controls, costs);
        

        if (controls.size() == 0) {
            return py::none();
        }

        py::safe_array<double> controls_array({controls.size(), controls[0].size()});
        py::safe_array<double> costs_array({costs.size()});
        auto controls_ref = controls_array.mutable_unchecked<2>();
        auto costs_ref = costs_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < controls.size(); ++i) {
            for (unsigned int j = 0; j < controls[0].size(); ++j) {
                controls_ref(i, j) = controls[i][j];
            }
            costs_ref(i) = costs[i];
        }

        py::safe_array<double> state_array({solution_path.size(), solution_path[0].size()});
        auto state_ref = state_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < solution_path.size(); ++i) {
            for (unsigned int j = 0; j < solution_path[0].size(); ++j) {
                state_ref(i, j) = solution_path[i][j];
            }
        }
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>, py::safe_array<double>>
            (state_array, controls_array, costs_array));
    }

    /**
	 * @copydoc planner_t::get_number_of_nodes()
	 */
    unsigned int get_number_of_nodes() {
        return this->planner->get_number_of_nodes();
    }

    std::string system_type;
protected:
    enhanced_system_t *system;
    std::function<double(const double*, const double*, unsigned int)> distance_computer;
    std::unique_ptr<trajectory_optimizers::CEM> cem;
    std::unique_ptr<networks::mpnet_cost_t> mpnet;
    // std::unique_ptr<networks::mpnet_cost_t> mpnet;

    std::unique_ptr<mpc_mpnet_t> planner;
    sst_node_t* nearest;
    std::vector<std::vector<double>> obs_list;
    double dt;
    double* loss_weights;
    torch::Tensor obs_tensor;
    torch::NoGradGuard no_grad;

private:

	/**
	 * @brief Captured distance computer python object to prevent its premature death
	 */
    py::object  _distance_computer_py;
};

PYBIND11_MODULE(_deep_smp_module, m) {
    m.doc() = "Python wrapper for deep smp planners";
    py::class_<MPCMPNetWrapper>(m, "MPCMPNetWrapper")
        .def(py::init<
            std::string,
            std::string,
            const py::safe_array<double>&,
            const py::safe_array<double>&,
            double,
            unsigned int,
            double,
            double,
            const py::safe_array<double>&,
            double,
            bool,
            std::string, std::string, 
            std::string, 
            int,
            int,
            int, int, int, int, int,
            double, const py::safe_array<double>, const py::safe_array<double>, double, double, double, double, double,
            std::string, float, bool,
            py::safe_array<double>&,
            py::safe_array<double>&>(),
            "system_type"_a,
            "solver_type"_a="cem",
            "start_state"_a,
            "goal_state"_a,
            "goal_radius"_a=10,
            "random_seed"_a=0,
            "sst_delta_near"_a=0.1,
            "sst_delta_drain"_a=0.1,
            "obs_list"_a=py::safe_array<double>(),
            "width"_a=0,
            "verbose"_a=false,
            "mpnet_weight_path"_a="", "cost_predictor_weight_path"_a="",
            "cost_to_go_predictor_weight_path"_a="",
            "num_sample"_a=1,
            "shm_max_step"_a=30,
            "np"_a=1, "ns"_a=32, "nt"_a=1, "ne"_a=4, "max_it"_a=10,
            "converge_r"_a=0.1, "mu_u"_a=py::safe_array<double>({0}), "std_u"_a=py::safe_array<double>({0}), "mu_t"_a=0, "std_t"_a=0, "t_max"_a=0, "step_size"_a=1, "integration_step"_a=2e-2,
            "device_id"_a="cuda:0", "refine_lr"_a=0.2, "normalize"_a=true,
            "weights_array"_a=py::safe_array<double>(),
            "obs_voxel_array"_a=py::safe_array<double>()
        )
        .def("steer", &MPCMPNetWrapper::steer,
            "start_array"_a,
             "sample_array"_a
        )
         .def("steer_sst", &MPCMPNetWrapper::steer_sst,
            "min_time_steps"_a,
             "max_time_steps"_a,
             "integration_step"_a
        )      
         .def("steer_batch", &MPCMPNetWrapper::steer_batch,
            "start_array"_a,
             "sample_array"_a,
             "num_of_problems"_a
        )       
        .def("neural_sample", &MPCMPNetWrapper::neural_sample,
            "state_array"_a,
            "refine"_a=false, 
            "refine_threshold"_a=0.2, 
            "using_one_step_cost"_a=false, 
            "cost_reselection"_a=false
        )
        .def("neural_sample_batch", &MPCMPNetWrapper::neural_sample_batch,
            "state_array"_a,
            "refine"_a=false, 
            "refine_threshold"_a=0.2, 
            "using_one_step_cost"_a=false, 
            "cost_reselection"_a=false,
            "num_of_problems"_a
        )
        .def("step", &MPCMPNetWrapper::step,
            "min_time_steps"_a,
            "max_time_steps"_a,
            "integration_step"_a
        )
        .def("mpc_step", &MPCMPNetWrapper::mpc_step,
            "integration_step"_a
        )
        .def("nearest_vertex", &MPCMPNetWrapper::nearest_vertex,
            "sample_state_array"_a
        )
        .def("add_to_tree", &MPCMPNetWrapper::add_to_tree,
            "sample_state_array"_a,
            "duration"_a
        )
        .def("neural_step", &MPCMPNetWrapper::neural_step,
            "refine"_a=false,
            "refine_threshold"_a=0,
            "using_one_step_cost"_a=false,
            "cost_reselection"_a=false,
            "goal_bias"_a=0
        )
        .def("deep_smp_step", &MPCMPNetWrapper::deep_smp_step,
            "refine"_a, 
            "refine_threshold"_a, 
            "using_one_step_cost"_a,
            "cost_reselection"_a,
            "goal_bias"_a=0,
            "NP"_a=1
        )
        .def("deep_smp_step_batch", &MPCMPNetWrapper::deep_smp_step_batch,
            "refine"_a, 
            "refine_threshold"_a, 
            "using_one_step_cost"_a,
            "cost_reselection"_a,
            "goal_bias"_a=0,
            "NP"_a=1
        )
        .def("neural_step_batch", &MPCMPNetWrapper::neural_step_batch,
            "refine"_a=false,
            "refine_threshold"_a=0,
            "using_one_step_cost"_a=false,
            "cost_reselection"_a=false,
            "goal_bias"_a=0,
            "num_of_problem"_a=1
        )
        .def("get_solution", &MPCMPNetWrapper::get_solution)
        .def("get_number_of_nodes", &MPCMPNetWrapper::get_number_of_nodes)
        ;
}
