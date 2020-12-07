#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/script.h>

#include <iostream>
#include <string>
namespace networks{
    class network_t
    {
    public:
        network_t(){}
        network_t(std::string network_weights_path, std::string device_id){
            // initialize network
            load_weights(network_weights_path, this->network_torch_module_ptr, device_id);            
        }
        void load_weights(std::string network_weights_path, std::shared_ptr<torch::jit::script::Module> &ptr, std::string device_id){
            if(network_weights_path == ""){
                // std::cout <<"Warning: Empty network_wieght_path, skipping model loading" << std::endl;
            } else{
                ptr.reset(new torch::jit::script::Module(
                    torch::jit::load(network_weights_path)));
                ptr -> to(torch::Device(device_id));
            }
        }

        virtual ~network_t(){
            network_torch_module_ptr.reset();
        }

        virtual at::Tensor forward(std::vector<torch::jit::IValue> mpnet_input_container) = 0;

    protected:
        /**
         * @brief loaded neural network
         */
        std::shared_ptr<torch::jit::script::Module> network_torch_module_ptr;

    };

}
#endif
