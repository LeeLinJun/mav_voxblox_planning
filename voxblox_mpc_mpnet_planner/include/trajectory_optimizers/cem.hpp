

#ifndef CEM_HPP
#define CEM_HPP
#include "systems/enhanced_system.hpp"
#include <vector>
#include <random>
#include <utility>
#include "iostream"
#include <algorithm>

namespace trajectory_optimizers{
    class CEM{
        public:
            CEM(enhanced_system_t* model, unsigned int number_of_samples, unsigned int number_of_t,
                unsigned int number_of_elite, double converge_r, 
                double* control_means, double* control_stds, 
                double time_means, double time_stds, double max_duration,
                double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size){
                    system = model;
                    this -> number_of_samples = number_of_samples;
                    this -> number_of_t = number_of_t;
                    this -> number_of_elite = number_of_elite;
                    mu_u0 = new double[system -> get_control_dimension()]();
                    std_u0 = new double[system -> get_control_dimension()]();
                    for(unsigned int ui = 0; ui < system -> get_control_dimension(); ui++){
                        mu_u0[ui] = control_means[ui];
                        std_u0[ui] = control_stds[ui];
                    }
                    mu_t0 = time_means;
                    std_t0 = time_stds;
                    this -> max_duration = max_duration;
                    s_dim = system -> get_state_dimension();
                    c_dim = system -> get_control_dimension();
                    dt = integration_step;
                    mu_u = new double[number_of_t * c_dim];
                    std_u = new double[number_of_t * c_dim];
                    mu_t = new double[number_of_t];
                    std_t = new double[number_of_t];
                    // sampled states, controls, times
                    states = new double[number_of_samples * s_dim];
                    controls = new double[number_of_samples * number_of_t * c_dim];
                    time = new double[number_of_samples * number_of_t];
                    // states for rolling
                    current_state = new double[s_dim];
                    this -> verbose = verbose;
                    converge_radius = converge_r;
                    
                    // util variables for update statistics
                    sum_of_controls = new double[number_of_t * c_dim];
                    sum_of_square_controls = new double[number_of_t * c_dim];
                    sum_of_time = new double[number_of_t];
                    sum_of_square_time = new double[number_of_t];
                    active_mask = new bool[number_of_samples];
                    step_size = step_size;
                    it_max = max_iteration;
                    weight = new double[s_dim];
                    for(unsigned int si = 0; si < s_dim; si++){
                        weight[si] = loss_weights[si];
                    }

            }
            ~CEM(){
                delete[] mu_u0;
                delete[] std_u0;
                delete[] mu_u;
                delete[] std_u;
                delete[] mu_t;
                delete[] std_t;
                delete[] states;
                delete[] controls;
                delete[] time;
                delete[] current_state;
                delete[] sum_of_controls;
                delete[] sum_of_square_controls;
                delete[] sum_of_time;
                delete[] sum_of_square_time;
                delete[] active_mask;
            };

            unsigned int get_control_dimension();
            
            unsigned int get_num_step();

            // (state, goal) -> u, update mu_u and std_u
            virtual void solve(const double* start, const double* goal, double *best_u, double *best_t);
            // (start, goal) -> state, update [path], slide mu_u and std_u
            virtual std::vector<std::vector<double>> rolling(double* start, double* goal);
            
            double converge_radius;
            double* weight;

        protected:
            enhanced_system_t *system;
            unsigned int number_of_samples, number_of_t, number_of_elite, it_max;
            double dt;
            unsigned s_dim, c_dim;
            double *states/* ns * dim_state */, 
                *controls/* ns * nt * dim_control */, *time/* ns * nt */,
                *current_state/* dim_state */;
            bool *active_mask/* ns */;
            double *mu_u/* nt * dim_control */, *std_u /* nt * dim_control */, *mu_t/* nt */, *std_t/* nt */;  
            double *mu_u0, *std_u0, mu_t0, std_t0, max_duration;
            std::vector<std::pair<double, int>> loss;
            std::default_random_engine generator;
            bool verbose;
            double *sum_of_controls;
            double *sum_of_square_controls;
            double *sum_of_time;
            double *sum_of_square_time;
            double step_size;
    };
}

#endif
