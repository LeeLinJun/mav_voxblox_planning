#ifndef MPNET_HPP
#define MPNET_HPP

#include "networks/network.hpp"
#include "systems/enhanced_system.hpp"

namespace networks{
    class mpnet_t : public network_t
    {
    public:
        mpnet_t() : network_t(){};
        mpnet_t(std::string network_weights_path, std::string device_id);
        at::Tensor forward(std::vector<torch::jit::IValue> mpnet_input_container);
        virtual void mpnet_sample(enhanced_system_t* system, torch::Tensor env_vox_tensor, 
            const double* state,  double* goal_state, double* neural_sample_state);

        ~mpnet_t();
    protected:
        std::shared_ptr<torch::jit::script::Module> network_torch_module_ptr;
    };
}
#endif
