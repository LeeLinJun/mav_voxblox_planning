#ifndef MPNET_COST_HPP
#define MPNET_COST_HPP

#include "networks/mpnet.hpp"

namespace networks{
    class mpnet_cost_t : public mpnet_t
    {
    public:
        mpnet_cost_t() : mpnet_t() {}
        mpnet_cost_t(std::string network_weights_path,
                     std::string cost_predictor_weights_path,
                     std::string cost_to_go_predictor_weights_path,
                     int num_sample, 
                     std::string device_id,
                     float refine_lr,
                     bool normalize);
        at::Tensor forward(std::vector<torch::jit::IValue> input_container);
        at::Tensor forward_cost(std::vector<torch::jit::IValue> input_container);
        at::Tensor forward_cost_to_go(std::vector<torch::jit::IValue> input_container);

        // virtual void mpnet_sample(enhanced_system_t* system, torch::Tensor env_vox_tensor, 
        //     const double* state,  double* goal_state, double* neural_sample_state) override;
        void mpnet_sample(enhanced_system_t* system, torch::Tensor& env_vox_tensor, 
            const double* state,  double* goal_state, double* neural_sample_state, 
            bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection);

        void mpnet_sample_single_batch(enhanced_system_t* system, torch::Tensor& env_vox_tensor, 
            const double* state,  double* goal_state, double* neural_sample_state, 
            bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, const int NP);

        void mpnet_sample_batch(enhanced_system_t* system, torch::Tensor& env_vox_tensor, 
            const double* state,  double* goal_state, double* neural_sample_state, 
            bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, const int NP);
        int num_sample;
        std::string device_id;
        float refine_lr;
        bool normalize;
        ~mpnet_cost_t();
    protected:
        std::shared_ptr<torch::jit::script::Module> network_torch_module_ptr;
        std::shared_ptr<torch::jit::script::Module> cost_predictor_torch_module_ptr;
        std::shared_ptr<torch::jit::script::Module> cost_to_go_predictor_torch_module_ptr;

    };
}
#endif
