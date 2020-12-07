#ifndef CEM_CUDA_CAR_HPP
#define CEM_CUDA_CAR_HPP
#include "systems/enhanced_system.hpp"

#include <vector>
// #include <random>
#include <utility>
#include "iostream"
// #include <algorithm>
// #include <chrono>   
#include <stdio.h>

#include "cuda.h"
// #include "cuda_dep.h"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

//#include <thrust/sort.h>
//#include <thrust/device_ptr.h>
//#include <thrust/sequence.h>
//#include <thrust/device_vector.h>
//#include <device_launch_parameters.h>   // this is required in Windows. But it should be deleted in Linux

#include "trajectory_optimizers/cem.hpp"

namespace trajectory_optimizers_car{
    class CEM_CUDA_car : public trajectory_optimizers::CEM{
        public:

            CEM_CUDA_car(enhanced_system_t* model, unsigned int number_of_samples, unsigned int number_of_t,
                unsigned int number_of_elite, double converge_r, 
                double* control_means, double* control_stds, 
                double time_means, double time_stds, double max_duration,
                double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size)
                : trajectory_optimizers::CEM(model, number_of_samples, number_of_t,
                    number_of_elite, converge_r, 
                    control_means, control_stds, 
                    time_means, time_stds, max_duration,
                    integration_step, loss_weights, max_iteration, verbose, step_size)
                {}

            CEM_CUDA_car(enhanced_system_t* model, unsigned int num_of_problems, unsigned int number_of_samples, unsigned int number_of_t,
                unsigned int number_of_elite,  double converge_r,
                std::vector<std::vector<double>>& _obs_list,
                double* control_means, double* control_stds, 
                double time_means, double time_stds, double max_duration,
                double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size);
                /*: CEM(model, number_of_samples, number_of_t,
                    number_of_elite, converge_r, 
                    control_means, control_stds, 
                    time_means, time_stds, max_duration,
                    integration_step, loss_weights, max_iteration, verbose, step_size) */
                
            virtual ~CEM_CUDA_car(){
                //std::cout << "inside CEM_CUDA_car deallication" << std::endl;
                cudaFree(d_temp_state);
                cudaFree(d_control);
                cudaFree(d_deriv);
                cudaFree(d_time);
                cudaFree(d_mean_time);
                cudaFree(d_mean_control);
                cudaFree(d_std_control);
                cudaFree(d_std_time);
                cudaFree(d_loss);
                cudaFree(d_top_k_loss);

                cudaFree(d_loss_ind);

                cudaFree(d_mu_u0);
                cudaFree(d_std_u0);
                if (best_u != NULL)
                {
                    cudaFree(d_best_u);  // newly added
                }
                if (best_t != NULL)
                {
                    cudaFree(d_best_t);  // newly added
                }

                if (d_sum_control != NULL)
                {
                    //std::cout << "CEM_CUDA_car: destroying d_sum_control" << std::endl;
                    cudaFree(d_sum_control);
                }
                if (d_ss_control != NULL)
                {
                    cudaFree(d_ss_control);
                }
                if (best_ut != NULL)
                {
                    cudaFree(d_best_ut);  // newly added
                }
                if (obs_list != NULL)
                {
                    cudaFree(d_obs_list);
                }
                if (obs_corner_list != NULL)
                {
                    cudaFree(d_obs_corner_list);
                }
                if (obs_axis_list != NULL)
                {
                    cudaFree(d_obs_axis_list);
                }
                if (obs_ori_list != NULL)
                {
                    cudaFree(d_obs_ori_list);
                }
                if (obs_min_max != NULL)
                {
                    cudaFree(d_obs_min_max);
                }

                cudaFree(d_active_mask);
                cudaFree(d_start_state);
                cudaFree(d_goal_state);
                cudaFree(devState);

                if (best_u != NULL)
                {
                    delete[] best_u;
                }
                if (best_t != NULL)
                {
                    delete[] best_t;
                }
                if (best_ut != NULL)
                {
                    delete[] best_ut;  // newly added
                }
                if (obs_list != NULL)
                {
                    delete[] obs_list;
                }
                if (obs_corner_list != NULL)
                {
                    delete[] obs_corner_list;
                }
                if (obs_axis_list != NULL)
                {
                    delete[] obs_axis_list;
                }
                if (obs_ori_list != NULL)
                {
                    delete[] obs_ori_list;
                }
                if (obs_min_max != NULL)
                {
                    delete[] obs_min_max;
                }

                delete[] mu_u0;
                delete[] std_u0;
            
                delete[] loss_ind;
                delete[] loss;
                //std::cout << "after CEM_CUDA_car deallication" << std::endl;

                // delete best_ut;

            };

            unsigned int get_control_dimension();
            
            unsigned int get_num_step();

            // (state, goal) -> u, update mu_u and std_u
            virtual void solve(const double* start, const double* goal, double *best_u, double *best_t);
            double* weight;
            int NP;
            int NS;
            int N_ELITE;
            int NT;
            int it_max;
            double* mu_u0;
            //double* mu_u;
            double* std_u0;
            //double* std_u;
            double mu_t0;
            double std_t0;
            double max_duration;
            int s_dim;
            int c_dim;
            double dt;

            bool verbose;
                // states for rolling
                
                // util variables for update statistics
            double converge_radius;
            // Cartpole();
            curandState* devState;


        protected:
            //enhanced_system_t *system;
            enhanced_system_t *system;
            double *d_temp_state, *d_control, *d_deriv, *d_time;
            double *d_mean_time, *d_mean_control, *d_std_control, *d_std_time;
            double *d_mu_u0, *d_std_u0;
            double *d_loss, *loss, *d_top_k_loss;
            int *d_loss_ind, *loss_ind;
            double /* *best_ut,*/ *d_best_u=NULL;
            double *best_ut=NULL, *d_best_ut=NULL;
            double *best_u=NULL, *best_t=NULL;
            double *d_best_t=NULL;
            // for obstacles
            // -- for other environments
            double* d_obs_list, *obs_list=NULL;
            // -- for car environment
            double* d_obs_corner_list, *obs_corner_list=NULL;
            double* d_obs_axis_list, *obs_axis_list=NULL;
            double* d_obs_ori_list, *obs_ori_list=NULL;
            // -- for quadrotor environment
            double* d_obs_min_max, *obs_min_max=NULL;


            // -- for update_statistics
            double* d_sum_control=NULL, *d_ss_control=NULL;

            bool* d_active_mask;
            std::vector<std::pair<double, int>> loss_pair;

            // for multi-start-goal
            double *d_start_state, *d_goal_state;
    };
}

#endif
