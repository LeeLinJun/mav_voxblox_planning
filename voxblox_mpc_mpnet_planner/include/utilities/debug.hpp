#include "systems/enhanced_system.hpp"
#include <iostream>


void check_state_validity(enhanced_system_t* model, double* state){
    for(unsigned int si = 0; si < model -> get_state_dimension(); si++){
        model -> temp_state[si] = state[si];
    }
    std::cout<<model->valid_state()<<std::endl;
}

void print_state(enhanced_system_t* model, double* state){
    for(int i = 0; i < model->get_state_dimension(); i++){
        std::cout<<state[i]<<", ";
    }
    std::cout<<std::endl;
    // std::copy(state, state + model->get_state_dimension(), std::ostream_iterator<double>(std::cout, ", "));

}
