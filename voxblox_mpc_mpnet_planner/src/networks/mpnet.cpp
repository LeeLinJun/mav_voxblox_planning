#ifndef MPNET_HPP
#include "networks/mpnet.hpp"
#endif
#include <string>

namespace networks{
    mpnet_t::mpnet_t(std::string network_weights_path, std::string device_id){
        load_weights(network_weights_path, this->network_torch_module_ptr, device_id);            

    }

    mpnet_t::~mpnet_t(){
        network_torch_module_ptr.reset();
    }

    at::Tensor mpnet_t::forward(std::vector<torch::jit::IValue> mpnet_input_container){
        return network_torch_module_ptr -> forward(mpnet_input_container).toTensor();
    }

    void mpnet_t::mpnet_sample(enhanced_system_t* system, torch::Tensor env_vox_tensor,
        const double* state, double* goal_state, double* neural_sample_state){
        double* normalized_state = new double[system->get_state_dimension()];
        double* normalized_goal = new double[system->get_state_dimension()];
        double* normalized_neural_sample_state = new double[system->get_state_dimension()];
        system -> normalize(state, normalized_state);
        system -> normalize(goal_state, normalized_goal);
        torch::Tensor state_goal_tensor = torch::ones({1, 8}).to(at::kCUDA); 
        std::vector<torch::jit::IValue> mpnet_input_container;

        // set value state_goal with dim 1 x 8
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si] = normalized_state[si]; 
        }
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            state_goal_tensor[0][si + system->get_state_dimension()] = normalized_goal[si]; 
        }

        #ifdef DEBUG_MPNET
        for(size_t i = 0; i < 8; i++){
            std::cout << (state_goal_tensor[0][i].item<float>())<< ",";
        }
        std::cout << std::endl;
        #endif

        mpnet_input_container.push_back(state_goal_tensor);
        mpnet_input_container.push_back((env_vox_tensor));
        at::Tensor output = this -> forward(mpnet_input_container);
       
        for(unsigned int si = 0; si < system->get_state_dimension(); si++){
            normalized_neural_sample_state[si] = output[0][si].item<double>();
        }
        #ifdef DEBUG_MPNET
        for(size_t i = 0; i < 4; i++){
            std::cout << (output[0][i].item<float>())<< ",";
        }
        std::cout << std::endl;
        #endif
        system -> denormalize(normalized_neural_sample_state, neural_sample_state);
        delete normalized_state;
        delete normalized_goal;
        delete normalized_neural_sample_state;
    }
    
}