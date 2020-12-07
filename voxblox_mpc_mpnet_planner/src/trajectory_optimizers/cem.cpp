#include "trajectory_optimizers/cem.hpp"
#include <chrono>

// #define PROFILE

// #define DEBUG
#define OBS_PENALTY 1000

namespace trajectory_optimizers{
    void CEM::solve(const double* start, const double* goal, double* best_u, double* best_t){
        // initialize 
        // initialize control_dist: [nt * c_dim]
        std::normal_distribution<double>* control_dist = 
            new std::normal_distribution<double>[number_of_t * c_dim];
        for(unsigned int i = 0; i < number_of_t; i++){
            for(unsigned int ui = 0; ui < c_dim; ui++){
                mu_u[c_dim * i + ui] = mu_u0[ui];
                std_u[c_dim * i + ui] = std_u0[ui];
            }
           
        }
        // initialize time_dist: [nt]
        std::normal_distribution<double>* time_dist = 
            new std::normal_distribution<double>[number_of_t];
        for(unsigned int i = 0; i < number_of_t; i++){
            mu_t[i] = mu_t0;
            std_t[i] = std_t0;
        }
        // initialize loss: set to <0, index>
        loss = std::vector<std::pair<double, int>>();
        // initialize states: set to starts
        for(unsigned int si = 0; si < number_of_samples; si++){
            for(unsigned int j = 0; j < s_dim; j++){
                states[si * s_dim + j] = start[j];
            }
            loss.push_back(std::make_pair(0., si));
        }
        // set early stop parameters
        double min_loss = OBS_PENALTY - 1e-1; // to +inf
        unsigned int early_stop_count = 0;

        // double* best_u = new double[c_dim];
        // begin loop
        for(unsigned int it = 0; it < it_max*100; it++){
            // reset active_mask in every iteration
            for(unsigned int si = 0; si < number_of_samples; si++){
                active_mask[si] = true;
                for(unsigned int j = 0; j < s_dim; j++){
                    states[si * s_dim + j] = start[j];
                }
            }
            // sampling control and time
            for(unsigned int ti = 0; ti < number_of_t; ti++){
                // reset statistics parameters
                for(unsigned int ci = 0; ci < c_dim; ci++){
                    control_dist[ti * c_dim + ci] = 
                        std::normal_distribution<double>
                        (mu_u[ti * c_dim + ci], std_u[ti * c_dim + ci]);
                }
                time_dist[ti] = 
                        std::normal_distribution<double>
                        (mu_t[ti], std_t[ti]);
            }

            for(unsigned int si = 0; si < number_of_samples; si++){
                for (unsigned int ti = 0; ti < number_of_t; ti++){
                    // generate control samples
                    for(unsigned int ci = 0; ci < c_dim; ci++){
                        double _control = control_dist[ti * c_dim + ci](generator);
                        if( _control < system->get_control_bounds()[ci].first){
                            _control = system->get_control_bounds()[ci].first;
                        } else if ( _control > system->get_control_bounds()[ci].second){
                             _control = system->get_control_bounds()[ci].second;
                        }
                        // std::cout<<"sampled_c:"<<_control<<std::endl;
                        controls[si* number_of_t * c_dim + ti * c_dim + ci] = _control;
                    }
                    // generate duration samples and wrap to [0, max_duration]
                    time[si* number_of_t + ti] = 
                            time_dist[ti](generator);
                    // std::cout << "time"<<"["<<si<<","<<ti<<"]"<<time[si* number_of_t + ti]<<std::endl;
                    if (time[si* number_of_t + ti] > max_duration){
                        time[si* number_of_t + ti] = max_duration;
                    } else if(time[si* number_of_t + ti] < 0){
                        time[si* number_of_t + ti] = 0;
                    }
                }
            }
            // propagation loops
            for(unsigned int si = 0; si < number_of_samples; si++){ // reset loss
                loss.at(si).first = 0; // loss: vector<double loss_value, int index>
                loss.at(si).second = si;
            }
            for(unsigned int ti=0; ti < number_of_t; ti++){ // time loop
                for(unsigned int si = 0; si < number_of_samples; si++){
                    if (active_mask[si]){
                        if (system -> propagate(&states[si * s_dim], 
                            s_dim, 
                            &controls[si * number_of_t * c_dim + ti],
                            c_dim,
                            (int)(time[si * number_of_t + ti] / dt),
                            &states[si * s_dim],
                            dt)){// collision free
                                // loss.at(si).first += time[si * number_of_t + ti];
                                // std::cout <<"si="<<si<<",ti="<<ti <<"\t"<<controls[si * number_of_t * c_dim + ti]<<std::endl;    
                                // loss.at(si).first += system -> get_loss(
                                //     &states[si*s_dim], goal, weight
                                //     );
                                double current_sample_loss = system -> get_loss(
                                    &states[si*s_dim], goal, weight
                                    );
                                if (current_sample_loss < converge_radius){
                                    active_mask[si] = false;
                                    // std::cout <<"si="<<si<<",ti="<<ti<<" reached,"<<current_sample_loss<<std::endl;
                                }
                        }
                        else{ // collision
                            loss.at(si).first += OBS_PENALTY;
                            active_mask[si] = false;
                        }
                    }
                }
            }            
            for(unsigned int si = 0; si < number_of_samples; si++){ // terminal_loss
                loss.at(si).first += system -> get_loss(&states[si * s_dim], goal, weight);
            }
            #ifdef DEBUG
            for(unsigned int si = 0; si < number_of_samples; si++){
                std::cout<< "si=" <<loss.at(si).first<< "\tid=" <<loss.at(si).second<<"\t"<<
                    states[si * s_dim]<<","<<states[si * s_dim+1] <<","<<states[si * s_dim+2] <<","<<states[si * s_dim+3] <<","<<std::endl;
                for(unsigned int ti = 0; ti < number_of_t; ti++){
                    std::cout<<"\t"<<controls[loss.at(si).second * number_of_t+ ti ] << ","<< (int)(time[loss.at(si).second * number_of_t + ti] / dt)<<std::endl;
                }
            }
            #endif
            //update statistics

            #ifdef PROFILE
            auto profile_start = std::chrono::high_resolution_clock::now();
            #endif
            sort(loss.begin(), loss.end());

            #ifdef PROFILE
            auto profile_stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem:solve. sort calls takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem:solve. 1000 steps of sort calls takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif

            
            #ifdef DEBUG
            for(unsigned int si = 0; si < number_of_samples; si++){
                std::cout<< "si=" <<loss.at(si).first<< "\tid=" <<loss.at(si).second<<std::endl;
                for(unsigned int ti = 0; ti < number_of_t; ti++){
                    std::cout<<"\t"<<controls[loss.at(si).second * number_of_t+ ti ] << ","<< (int)(time[loss.at(si).second * number_of_t + ti] / dt)<<std::endl;
                }
            }
            #endif

            // early stop checking
            if(loss.at(0).first < min_loss /*&& loss.at(0).first < 100*/){
                min_loss = loss.at(0).first;
                int si = loss.at(0).second;
                for(unsigned int ti = 0; ti < number_of_t; ti++){
                    for(unsigned int ci = 0; ci < c_dim; ci++){
                        // si * number_of_t * c_dim + ti * c_dim
                        best_u[ti*c_dim + ci] = controls[si * number_of_t * c_dim + ti * c_dim + ci];
                    }
                    best_t[ti] = time[si* number_of_t + ti];
                    #ifdef DEBUG
                        std::cout << "storing: ti="<<ti<<"\tu=" << best_u[ti*c_dim + 0] <<",\t t="<< best_t[ti]<<std::endl;
                    #endif
                }

            } else{
                early_stop_count += 1;
                if(early_stop_count >= it_max || min_loss >= OBS_PENALTY){
                    break;
                }
            }
            // //debug
            // for(unsigned int si = 0; si < number_of_elite; si++){ // 
            //     std::cout<< loss.at(si).first <<"," <<loss.at(si).second<<"\t";
            //     int index = loss.at(si).second;
            //     for(unsigned int s_di = 0; s_di < s_dim; s_di ++){
            //         std::cout << states[index*s_dim + s_di]<<" ";
            //     }
                
            //     std::cout<<system -> get_loss(&states[index * s_dim], goal, weight)<<std::endl;
            // }
            // //debug
            for(unsigned int ti = 0; ti < number_of_t; ti++){
                for(unsigned int ci = 0; ci < c_dim; ci++){
                    sum_of_controls[ti * c_dim + ci] = 0;
                    sum_of_square_controls[ti * c_dim + ci] = 0;
                }
                sum_of_time[ti] = 0;
                sum_of_square_time[ti] = 0;
            }
            for(unsigned int index = 0; index < number_of_elite; index++){
                int si = loss.at(index).second;
                for(unsigned int ti = 0; ti < number_of_t; ti++){
                    for(unsigned int ci = 0; ci < c_dim; ci++){
                        double control_i = 
                            controls[si * number_of_t * c_dim + ti * c_dim + ci];
                        sum_of_controls[ti * c_dim + ci] += control_i;
                        sum_of_square_controls[ti * c_dim + ci] += control_i * control_i;
                    }
                    double time_i = time[si * number_of_t + ti];
                    sum_of_time[ti] += time_i;
                    sum_of_square_time[ti] += time_i * time_i;
                }
            }
            for(unsigned int ti = 0; ti < number_of_t; ti++){
                for(unsigned int ci = 0; ci < c_dim; ci++){
                    mu_u[ti * c_dim + ci] = step_size * sum_of_controls[ti * c_dim + ci] / number_of_elite + 
                        (1-step_size)*mu_u[ti * c_dim + ci];
                    std_u[ti * c_dim + ci] = step_size * sqrt(
                        std::max(sum_of_square_controls[ti * c_dim + ci] / 
                                 number_of_elite - mu_u[ti * c_dim + ci] * mu_u[ti * c_dim + ci], 1e-10)) + (1-step_size) * std_u[ti * c_dim + ci];
                    // std::cout<<"mu_u:"<<mu_u[ti * c_dim + ci]<<","<<std_u[ti * c_dim + ci]<<std::endl;

                }
                mu_t[ti] = step_size * sum_of_time[ti] / number_of_elite + (1-step_size)*mu_t[ti];
                std_t[ti] = step_size * sqrt(std::max(sum_of_square_time[ti]/ number_of_elite - mu_t[ti] * mu_t[ti], 1e-10)) + (1-step_size) * (std_t[ti]);
            }

            if(verbose){
                std::cout <<it<<"\tloss:"<< loss.at(0).first<<"\tminLoss:"<< min_loss;
                /*for(unsigned int si = 0; si < s_dim; si++){
                    std::cout <<<<"\tstates:\t" << states[loss.at(0).second * s_dim + si];
                }*/
                std::cout<< std::endl;
            }
            if(min_loss < converge_radius){
                    break;
            }
        }// end loop
        delete[] control_dist;
        delete[] time_dist;

    }

    std::vector<std::vector<double>> CEM::rolling(double* start, double* goal){
        // initialize
        for(unsigned int i = 0; i < s_dim; i++){
             current_state[i] = start[i];
        }

        for(unsigned int i = 0; i < number_of_t; i++){
            for(unsigned int ui = 0; ui < c_dim; ui++){
                mu_u[c_dim * i + ui] = mu_u0[ui];
                std_u[c_dim * i + ui] = std_u0[ui];
            }
           
        }
        unsigned int early_stop_count = 0;
        std::vector<std::vector<double>> path; 
        double current_loss = 1e2;
        double* u = new double[number_of_t * c_dim];
        double* t = new double[number_of_t];
        // rolling
        while(current_loss > converge_radius && early_stop_count < it_max){
            CEM::solve(start, goal, u, t);
            if(! system -> propagate(current_state, s_dim, &u[0], c_dim, 
                (int)(t[0]/dt), current_state, dt)){
                    break;
            }
            current_loss = system -> get_loss(current_state, goal, weight);
            std::cout<<"current_loss:"<< current_loss<< std::endl;
            std::vector<double> path_node;
            for(unsigned int i = 0; i < s_dim; i++){
                path_node.push_back(current_state[i]);
            }
            path.push_back(path_node);
        }
        return path;
     }

    unsigned int CEM::get_control_dimension(){
         return c_dim * number_of_t;
     }

    unsigned int CEM::get_num_step(){
         return number_of_t;
     }
    
}