/**
 * @file sst.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Oleg Y. Sinyavskiy
 * 
 */

#include "motion_planners/sst.hpp"
#include "motion_planners/deep_smp_mpc_sst.hpp"
#include "nearest_neighbors/graph_nearest_neighbors.hpp"
#ifndef TORCH_H
#include <torch/script.h>
#endif
#include <cstdio>
#include <iostream>
#include <deque>


#include <chrono>


// #define PROFILE
// #define DEBUG
// #define DEBUG_MPNET
// #define DEBUG_CEM
// #define PRINT_GOAL

deep_smp_mpc_sst_t::deep_smp_mpc_sst_t(
    const double* in_start, const double* in_goal,
    double in_radius,
    const std::vector<std::pair<double, double> >& a_state_bounds,
    const std::vector<std::pair<double, double> >& a_control_bounds,
    std::function<double(const double*, const double*, unsigned int)> a_distance_function,
    unsigned int random_seed,
    double delta_near, double delta_drain,
    trajectory_optimizers::CEM* cem_ptr,
    networks::mpnet_cost_t *mpnet_ptr,
    int np, 
    int shm_max_step
    ) 
    : planner_t(in_start, in_goal, in_radius,
                a_state_bounds, a_control_bounds, a_distance_function, random_seed)
    , best_goal(nullptr)
    , sst_delta_near(delta_near)
    , sst_delta_drain(delta_drain)
    , cem_ptr(cem_ptr)
    , mpnet_ptr(mpnet_ptr)
    , NP(np)
    , shm_max_step(shm_max_step)
{
    //initialize the metrics
    unsigned int state_dimensions = this->get_state_dimension();
    std::function<double(const double*, const double*)> raw_distance =
        [state_dimensions, a_distance_function](const double* s0, const double* s1) {
            return a_distance_function(s0, s1, state_dimensions);
        };
    metric.set_distance(raw_distance);

    root = new sst_node_t(in_start, a_state_bounds.size(), nullptr, tree_edge_t(nullptr, 0, -1.), 0.);
    metric.add_node(root);
    number_of_nodes++;

    samples.set_distance(raw_distance);

    sample_node_t* first_witness_sample = new sample_node_t(static_cast<sst_node_t*>(root), start_state, this->state_dimension);
    samples.add_node(first_witness_sample);
    witness_nodes.push_back(first_witness_sample);

    shm_current_state = new double[np * state_dimensions]();
    shm_counter = new int[np]();
    // start_state = new double[state_dimensions]();
    for(int pi = 0; pi < np; pi++){
        for(int si = 0; si < state_dimensions; si ++){
            shm_current_state[pi * state_dimension + si] = in_start[si];
            // start_state[si] = in_start[si];
        }

        shm_counter[pi] = 0;
    }
    
}

deep_smp_mpc_sst_t::~deep_smp_mpc_sst_t() {
    delete root;
    for (auto w: this->witness_nodes) {
        delete w;
    }
    delete shm_current_state;
    delete shm_counter;
}


void deep_smp_mpc_sst_t::get_solution(std::vector<std::vector<double>>& solution_path, std::vector<std::vector<double>>& controls, std::vector<double>& costs)
{
	if(best_goal==NULL)
		return;
	sst_node_t* nearest_path_node = best_goal;
	
	//now nearest_path_node should be the closest node to the goal state
	std::deque<sst_node_t*> path;
	while(nearest_path_node->get_parent()!=NULL)
	{
		path.push_front(nearest_path_node);
        nearest_path_node = nearest_path_node->get_parent();
	}

    std::vector<double> root_state;
    for (unsigned c=0; c<this->state_dimension; c++) {
        root_state.push_back(root->get_point()[c]);
    }
    solution_path.push_back(root_state);

	for(unsigned i=0;i<path.size();i++)
	{
        std::vector<double> current_state;
        for (unsigned c=0; c<this->state_dimension; c++) {
            current_state.push_back(path[i]->get_point()[c]);
        }
        solution_path.push_back(current_state);

        std::vector<double> current_control;
        for (unsigned c=0; c<this->control_dimension; c++) {
            current_control.push_back(path[i]->get_parent_edge().get_control()[c]);
        }
        controls.push_back(current_control);
        costs.push_back(path[i]->get_parent_edge().get_duration());
	}
}

void deep_smp_mpc_sst_t::step(enhanced_system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    delete sample_state;
    delete sample_control;
}


void deep_smp_mpc_sst_t::step(system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    delete sample_state;
    delete sample_control;
}

void deep_smp_mpc_sst_t::step_with_output(enhanced_system_interface* system, int min_time_steps, int max_time_steps, double integration_step, double* steer_start, double* steer_goal)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    // copy to output
    for (unsigned i=0; i <this->state_dimension; i++)
    {
        steer_start[i] = nearest->get_point()[i];
        steer_goal[i] = sample_state[i];
    }

    delete sample_state;
    delete sample_control;
}




sst_node_t* deep_smp_mpc_sst_t::nearest_vertex(const double* sample_state)
{
	//performs the best near query
    std::vector<proximity_node_t*> close_nodes = metric.find_delta_close_and_closest(sample_state, this->sst_delta_near);

    double length = std::numeric_limits<double>::max();;
    sst_node_t* nearest = nullptr;
    for(unsigned i=0;i<close_nodes.size();i++)
    {
        tree_node_t* v = (tree_node_t*)(close_nodes[i]->get_state());
        double temp = v->get_cost() ;
        if( temp < length)
        {
            length = temp;
            nearest = (sst_node_t*)v;
        }
    }
    assert (nearest != nullptr);
    return nearest;
}

void deep_smp_mpc_sst_t::add_to_tree(const double* sample_state, const double* sample_control, sst_node_t* nearest, double duration)
{
	//check to see if a sample exists within the vicinity of the new node
    sample_node_t* witness_sample = find_witness(sample_state);

    sst_node_t* representative = witness_sample->get_representative();
	if(representative==NULL || representative->get_cost() > nearest->get_cost() + duration)
	{
		if(best_goal==NULL || nearest->get_cost() + duration <= best_goal->get_cost())
		{
			//create a new tree node
			//set parent's child
			sst_node_t* new_node = static_cast<sst_node_t*>(nearest->add_child(
			    new sst_node_t(
                    sample_state, this->state_dimension,
                    nearest,
                    tree_edge_t(sample_control, this->control_dimension, duration),
                    nearest->get_cost() + duration)
            ));
			number_of_nodes++;

            #ifdef PRINT_GOAL
            std::cout <<"goal_distance:" << distance(new_node->get_point(), goal_state, this->state_dimension) << std::endl;
	        #endif
            if(best_goal==NULL && this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }
	        else if(best_goal!=NULL && best_goal->get_cost() > new_node->get_cost() &&
	                this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }

            // Acquire representative again - it can be different
            representative = witness_sample->get_representative();
			if(representative!=NULL)
			{
				//optimization for sparsity
				if(representative->is_active())
				{
					metric.remove_node(representative);
					representative->make_inactive();
				}

	            sst_node_t* iter = representative;
	            while( is_leaf(iter) && !iter->is_active() && !is_best_goal(iter))
	            {
	                sst_node_t* next = (sst_node_t*)iter->get_parent();
	                remove_leaf(iter);
	                iter = next;
	            } 

			}
			witness_sample->set_representative(new_node);
			new_node->set_witness(witness_sample);
			metric.add_node(new_node);
		}
	}	

}

sample_node_t* deep_smp_mpc_sst_t::find_witness(const double* sample_state)
{
	double distance;
    sample_node_t* witness_sample = (sample_node_t*)samples.find_closest(sample_state, &distance)->get_state();
	if(distance > this->sst_delta_drain)
	{
		//create a new sample
		witness_sample = new sample_node_t(NULL, sample_state, this->state_dimension);
		samples.add_node(witness_sample);
		witness_nodes.push_back(witness_sample);
	}
    return witness_sample;
}

void deep_smp_mpc_sst_t::branch_and_bound(sst_node_t* node)
{
    // Copy children becuase apparently, they are going to be modified
    std::list<tree_node_t*> children = node->get_children();
    for (std::list<tree_node_t*>::const_iterator iter = children.begin(); iter != children.end(); ++iter)
    {
    	branch_and_bound((sst_node_t*)(*iter));
    }
    if(is_leaf(node) && node->get_cost() > best_goal->get_cost())
    {
    	if(node->is_active())
    	{
	    	node->get_witness()->set_representative(NULL);
	    	metric.remove_node(node);
	    }
    	remove_leaf(node);
    }
}

bool deep_smp_mpc_sst_t::is_leaf(tree_node_t* node)
{
	return node->is_leaf();
}

void deep_smp_mpc_sst_t::remove_leaf(sst_node_t* node)
{
	if(node->get_parent() != NULL)
	{
		node->get_parent_edge();
		node->get_parent()->remove_child(node);
		number_of_nodes--;
		delete node;
	}
}

bool deep_smp_mpc_sst_t::is_best_goal(tree_node_t* v)
{
	if(best_goal==NULL)
		return false;
    sst_node_t* new_v = best_goal;

    while(new_v->get_parent()!=NULL)
    {
        if(new_v == v)
            return true;

        new_v = new_v->get_parent();
    }
    return false;

}


void deep_smp_mpc_sst_t::mpc_step(enhanced_system_t* system, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* terminal_state = new double[this->state_dimension]();
    double* sample_state = new double[this->state_dimension];
	this->random_state(sample_state);
    sst_node_t* nearest = nearest_vertex(sample_state);
    double duration = steer(system, nearest->get_point(), sample_state, terminal_state, integration_step);
	if(duration > 0)
	{
		add_to_tree(terminal_state, 0, nearest, duration);
	}
    delete sample_state;
    delete terminal_state;
}


void deep_smp_mpc_sst_t::neural_sample(enhanced_system_t* system, const double* nearest,
    double* neural_sample_state, torch::Tensor& env_vox_tensor, bool refine, float refine_threshold, 
    bool using_one_step_cost, bool cost_reselection){
    mpnet_ptr->mpnet_sample(system, env_vox_tensor, nearest, goal_state, neural_sample_state, refine, refine_threshold, 
        using_one_step_cost, cost_reselection);
}

void deep_smp_mpc_sst_t::neural_sample_batch(enhanced_system_t* system, const double* nearest,
    double* neural_sample_state, torch::Tensor& env_vox_tensor, bool refine, float refine_threshold, 
    bool using_one_step_cost, bool cost_reselection, const int NP){

    mpnet_ptr->mpnet_sample_batch(system, env_vox_tensor, nearest, goal_state, neural_sample_state, refine, refine_threshold, 
        using_one_step_cost, cost_reselection, NP);

}


/**  Steer Function using CEM */
double deep_smp_mpc_sst_t::steer(enhanced_system_t* system, const double* start, const double* sample, 
   double* terminal_state, double integration_step){
    #ifdef DEBUG_CEM 
    std::cout<<"start state:\t";
    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << start[si]<<","; 
    }
    std::cout<<"\ngoal state:\t";
    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << sample[si]<<","; 
    }
    std::cout << std::endl;
    #endif
    double* solution_u = new double[cem_ptr -> get_control_dimension()];
    double* solution_t = new double[cem_ptr -> get_num_step()];
    double* costs = new double[cem_ptr -> get_num_step()];
    double* state = new double[this->state_dimension];
    cem_ptr -> solve(start, sample, solution_u, solution_t);
    // for(int ti = 0; ti < cem_ptr -> get_num_step(); ti++) {
    //     printf("solution_t:%f\n", solution_t[ti]);
    // }
    double duration = 0;
    for(unsigned int si = 0; si < this->state_dimension; si++){ //copy start state
        state[si] = start[si]; 
        terminal_state[si] = start[si]; 
    }
    // printf("%f, %f, %f, %f, %f, %f, %f\n", state[0], state[1], state[2], state[3],state[4],state[5],state[6]);
    // printf("%f, %f, %f, %f, %f, %f, %f\n", sample[0], sample[1], sample[2], sample[3],sample[4],sample[5],sample[6]);

    // printf("%f, %f, %f, %f, %f, %f, %f\n", terminal_state[0],terminal_state[1],terminal_state[2],terminal_state[3],terminal_state[4],terminal_state[5],terminal_state[6]);
    double min_loss = 1e3;// initialize logging variables
    unsigned int best_i = 0;

    for(unsigned int ti = 0; ti < cem_ptr -> get_num_step(); ti++){ // propagate
        if (system -> propagate(state, 
            this->state_dimension, 
            &solution_u[ti*system->get_control_dimension()], 
            this->control_dimension, 
            (int)(solution_t[ti]/integration_step), 
            state,
            integration_step)){
               double current_loss = system -> get_loss(state, sample, cem_ptr -> weight);
                #ifdef DEBUG_CEM
                    std::cout<<"current_loss:" << current_loss <<std::endl;
                #endif
                costs[ti] = integration_step * (int)(solution_t[ti]/integration_step); // logging costs
                if (current_loss < min_loss || this->distance(state, goal_state, this->state_dimension) < goal_radius){//update min_loss
                    min_loss = current_loss;
                    best_i = ti;
                    for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
                        terminal_state[si] = state[si];
                        // printf("%f, ", terminal_state[si]); 
                    }
                    // printf("/n");
                    if (min_loss < cem_ptr -> converge_radius){
                        break;
                    }
                    #ifdef DEBUG_CEM
                        std::cout<<"min_loss:" << min_loss <<std::endl;
                    #endif
                }        
        }
        else{
            // duration = -1000;
            break;
        } 
    }
    // if (min_loss < 1) {
    for(unsigned int ti = 0; ti <= best_i; ti++){    // compute duration until best duration
        duration += costs[ti];
    }
    // }else{
        // duration = -1;
    // }
   

    #ifdef DEBUG_CEM
        std::cout<<"terminal state:\t";
        for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            std::cout << terminal_state[si]; 
        }
        std::cout<<std::endl;    
        std::cout << duration<< std::endl;
        std::cout<<"steered"<<std::endl;
    #endif
    delete solution_u;
    delete solution_t;
    delete state;
    delete costs;
    return duration;
}

void deep_smp_mpc_sst_t::steer_batch(enhanced_system_t* system, const double* start, const double* sample, 
    double* terminal_state, double integration_step, const int NP, double* duration){
    /**
     * start: NP * N_STATE
     * sample: NP * N_STATE
     * 
    */

    double* solution_u = new double[NP * cem_ptr -> get_num_step() * system -> get_control_dimension()];  // NP x NT x NU
    double* solution_t = new double[NP * cem_ptr -> get_num_step()];  // NP x NT
    double* costs = new double[NP * cem_ptr -> get_num_step()]();
    double* state = new double[this->state_dimension];

    #ifdef PROFILE
    auto profile_start = std::chrono::high_resolution_clock::now();
    #endif
    cem_ptr -> solve(start, sample, solution_u, solution_t);
    #ifdef PROFILE
    auto profile_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> profile_duration = profile_stop - profile_start; 
    std::cout << "inside deep_smp_mpc_sst:steer_bath. solve takes " << profile_duration.count() << "s" << std::endl; 
    std::cout << "inside deep_smp_mpc_sst:steer_batch. 1000 steps of solve takes " << 1000*profile_duration.count() << "s" << std::endl; 
    #endif
    
    //double* duration = new double[NP];
    for(unsigned int pi = 0; pi < NP; pi++)
    {
        duration[pi] = 0;
        for(unsigned int si = 0; si < this->state_dimension; si++){ //copy start state
            state[si] = start[pi*this->state_dimension+si]; 
        }
        double min_loss = 1e3;// initialize logging variables
        unsigned int best_i = 0;
        
        // initialize terminal_state to starting point
        for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
            terminal_state[pi*this->state_dimension+si] = state[si]; 
        }
        double best_i_cost = 0.;  // record the cost when the best_i is recorded
        for(unsigned int ti = 0; ti < cem_ptr -> get_num_step(); ti++){ // propagate for the steps of CEM
            int num_dt = solution_t[pi*cem_ptr->get_num_step()+ti]/integration_step;
            costs[pi*cem_ptr->get_num_step()+ti] = 0.;
            int collision =false;
            int early_stop=false;
            for (unsigned int dti = 0; dti < num_dt; dti++)
            {
                if (system -> propagate(state, 
                    this->state_dimension, 
                    &solution_u[(pi * cem_ptr->get_num_step()+ti)*this->control_dimension], 
                    this->control_dimension, 
                    1, 
                    state,
                    integration_step)){

                    double current_loss = system -> get_loss(state, sample+pi*this->state_dimension, cem_ptr -> weight);
                    #ifdef DEBUG_CEM
                        std::cout<<"current_loss:" << current_loss <<std::endl;
                    #endif
                    costs[pi*cem_ptr->get_num_step()+ti] += integration_step; // logging costs

                    if (current_loss < min_loss || this->distance(state, goal_state, this->state_dimension) < goal_radius){//update min_loss
                        min_loss = current_loss;
                        best_i = ti;
                        best_i_cost = costs[pi*cem_ptr->get_num_step()+ti];
                        for(unsigned int si = 0; si < this->state_dimension; si++){// save best state
                            terminal_state[pi*this->state_dimension+si] = state[si]; 
                        }
                        if (min_loss < cem_ptr -> converge_radius || this->distance(state, goal_state, this->state_dimension) < goal_radius){
                                early_stop = true;
                                break;
                        }
                        #ifdef DEBUG_CEM
                            std::cout<<"min_loss:" << min_loss <<std::endl;
                        #endif
                    }        
                }

                else{
                    // duration = -1000;
                    collision = true;
                    break;
                } 
            }
            if (early_stop || collision){
                    break;
            }

        }
        for(unsigned int ti = 0; ti < best_i; ti++){    // compute duration until best duration
            duration[pi] += costs[pi*cem_ptr->get_num_step()+ti];
        }
        // for the last step (best_i), use the one up to the best dt
        duration[pi] += best_i_cost;
    }

    delete solution_u;
    delete solution_t;
    delete state;
    delete costs;
}

void deep_smp_mpc_sst_t::neural_step(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * apply neural sampling from this sample
     * connect the start node to the sample node with trajectory optimization
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */

    // previous working code
    
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[this->state_dimension]();
    double* terminal_state = new double[this->state_dimension]();
    double prob = this->random_generator.uniform_random(0, 1);
    if (prob < goal_bias){
        for (unsigned int i = 0; i < state_dimension; i++){
            sample_state[i] = goal_state[i];
        }
    }
    else{
	    this->random_state(sample_state);
    }
    sst_node_t* nearest = nearest_vertex(sample_state);
    //  add neural sampling 
    neural_sample(system, nearest->get_point(), neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection); 
    // steer func
    for(unsigned int i = 0; i < state_dimension; i++){
        system->temp_state[i] = neural_sample_state[i];
    }
    if (system -> valid_state()){
        double duration = steer(system, nearest->get_point(), neural_sample_state, terminal_state, integration_step);
        // std::cout<<"duration:" << duration << std::endl;    
        if(duration > 0)
        {
            add_to_tree(terminal_state, 0, nearest, duration);
        }
    }
    for(unsigned int i = 0; i < state_dimension; i++){
        states[i] = nearest->get_point()[i];
        states[i + state_dimension] = terminal_state[i];
        states[i + state_dimension*2] = neural_sample_state[i];
    }
    
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    
}


void deep_smp_mpc_sst_t::deep_smp_step(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias)
{
    /* Make prediction on current state
     * Find nearest neighbor of the prediction in the tree
     * Try to steer to that node
     * Add the terminal node to the tree.
     */
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[this->state_dimension]();
    double* terminal_state = new double[this->state_dimension]();
    //  add neural sampling 
    double prob = this->random_generator.uniform_random(0, 1);

    if (prob < goal_bias){
        for (unsigned int i = 0; i < state_dimension; i++){
            neural_sample_state[i] = goal_state[i];
        }
    } else {
        neural_sample(system, shm_current_state, neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection);
    }
    // printf("%f, %f, %f, %f, %f, %f, %f\n", neural_sample_state[0], neural_sample_state[1], neural_sample_state[2],
    //     neural_sample_state[3],neural_sample_state[4],neural_sample_state[5],neural_sample_state[6]);

    // sst_node_t* nearest = nearest_vertex(neural_sample_state);
    sst_node_t* nearest = nearest_vertex(shm_current_state);

 
    // steer func
    for(unsigned int si = 0; si < state_dimension; si++){
        system->temp_state[si] = neural_sample_state[si];
    }

    bool reset = true;
    // printf("%d", system -> valid_state());
    if (system -> valid_state()){
        shm_counter[0]++;
        double duration = steer(system, nearest->get_point(), neural_sample_state, terminal_state, integration_step);
        // std::cout<< duration << std::endl;
        // std::cout<<"duration:" << duration << std::endl;    
        if(duration > 0)
        {
            add_to_tree(terminal_state, 0, nearest, duration);
            for(unsigned int si = 0; si < state_dimension; si++){
                shm_current_state[si] = terminal_state[si];
                reset = false;
            }
            
            // if (this -> distance(shm_current_state, goal_state, this->state_dimension) > 1.5 * goal_radius || smp_counter > 40){    
            //     sst_node_t* goal_nearest = nearest_vertex(goal_state);
            //     double duration = steer(system, goal_nearest->get_point(), goal_state, terminal_state, integration_step);
            //     if(duration > 0)
            //     {
            //         add_to_tree(terminal_state, 0, goal_nearest, duration);
            //     }
            //     for(int si = 0; si < this -> state_dimension; si ++){
            //         shm_current_state[si] = root->get_point()[si];
            //     }
            // }
        } 
    } 
  

    for(unsigned int i = 0; i < state_dimension; i++){
        states[i] = nearest->get_point()[i];
        states[i + state_dimension] = terminal_state[i];
        states[i + state_dimension*2] = neural_sample_state[i];
    }

    if(reset || shm_counter[0] > shm_max_step) {
        // for(int si = 0; si < this -> state_dimension; si ++){
        //     shm_current_state[si] = root -> get_point()[si];
        //     // std::cout << shm_current_state[si] << " ";
        // }
        // // std::cout << std::endl;
        // smp_counter = 0;
        this->random_state(sample_state);
        nearest = nearest_vertex(sample_state);
        for(int si = 0; si < this -> state_dimension; si ++){
            // shm_current_state[si] = root -> get_point()[si];
            // std::cout << shm_current_state[si] << " ";
            shm_current_state[si] = nearest -> get_point()[si];
        }
        // std::cout << std::endl;
        shm_counter[0] = 0;
    }
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    
}


void deep_smp_mpc_sst_t::deep_smp_step(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias, int NP)
{
    /* Make prediction on previous node
     * Find nearest neighbor of the prediction in the tree
     * Try to steer to that node
     * Add the terminal node to the tree.
     */

    // int NP = 10;
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[NP*this->state_dimension]();
    double* terminal_state = new double[this->state_dimension]();
    //  add neural sampling 
    neural_sample_batch(system, shm_current_state, neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection, NP);
    bool reset = true;
    sst_node_t* nearest;
    // smp_counter++;
    nearest = nearest_vertex(shm_current_state);

    for(int pi = 0; pi < NP; pi++){
        //  nearest = nearest_vertex(&neural_sample_state[pi * state_dimension]);
         for(unsigned int si = 0; si < state_dimension; si++){
            system->temp_state[si] = neural_sample_state[pi * state_dimension + si];
            terminal_state[si] = shm_current_state[si];
        }

        if (system -> valid_state()){
            double duration = steer(system, nearest->get_point(), &neural_sample_state[pi * state_dimension], terminal_state, integration_step);
            // std::cout<<"duration:" << duration << std::endl;    
            if(duration > 0)
            {
                add_to_tree(terminal_state, 0, nearest, duration);
                for(unsigned int si = 0; si < state_dimension; si++){
                    shm_current_state[si] = terminal_state[si];
                }
                reset = false;
                break;
            } 
            // else {
            //     reset = true;
            // }
        } 
        // else {
        //     reset = true;
        // }
    }
    for(unsigned int i = 0; i < state_dimension; i++){
        states[i] = nearest->get_point()[i];
        states[i + state_dimension] = terminal_state[i];
        states[i + state_dimension*2] = neural_sample_state[i];
    }
    if(reset  /*|| smp_counter > 50*/) {
        this->random_state(sample_state);
        nearest = nearest_vertex(sample_state);
        for(int si = 0; si < this -> state_dimension; si ++){
            // shm_current_state[si] = root -> get_point()[si];
            // std::cout << shm_current_state[si] << " ";
            shm_current_state[si] = nearest -> get_point()[si];
        }
        // std::cout << std::endl;
        // smp_counter = 0;
    }
    
    
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    
}


void deep_smp_mpc_sst_t::deep_smp_step_batch(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias, int NP) {
    /* Make prediction on current state
     * Find nearest neighbor of the prediction in the tree
     * Try to steer to that node
     * Add the terminal node to the tree.
     */
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[NP * this->state_dimension]();
    double* terminal_state = new double[NP * this->state_dimension]();
    //  add neural sampling 
    double* prob = new double[NP]();
    std::vector<sst_node_t*> nearest_list(NP, NULL);
    bool* reset = new bool[NP]();
    double* duration = new double[NP]();
    double* steer_start_state =  new double[NP*this->state_dimension]();

    neural_sample_batch(system, shm_current_state, neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection, NP);
    for (int pi = 0; pi < NP; pi++){
        reset[pi] = true;
        prob[pi] = this->random_generator.uniform_random(0, 1);
        if (prob[pi] < goal_bias){
            for (unsigned int si = 0; si < state_dimension; si++){
                neural_sample_state[pi * this->state_dimension + si] = goal_state[si];
            }
        }
        // sst_node_t* nearest = nearest_vertex(neural_sample_state);
        nearest_list[pi] = nearest_vertex(shm_current_state + pi * this->state_dimension);
        // steer func
        for(int si = 0; si < this->state_dimension; si++){
            steer_start_state[pi * this->state_dimension + si] = nearest_list[pi]->get_point()[si];
        }
    }
    
    steer_batch(system, steer_start_state, neural_sample_state, terminal_state, integration_step, NP, duration);

    for (int pi = 0; pi < NP; pi++){
        for(unsigned int si = 0; si < state_dimension; si++){
            system->temp_state[si] = neural_sample_state[pi * this->state_dimension + si];
        }
        if (system -> valid_state()){
            shm_counter[pi]++;
            if(duration[pi] > 0)
            {
                add_to_tree(&terminal_state[pi * this->state_dimension], 0, nearest_list[pi], duration[pi]);
                for(unsigned int si = 0; si < state_dimension; si++){
                    shm_current_state[pi * this->state_dimension + si] = terminal_state[pi * this->state_dimension + si];
                    reset[pi] = false;
                }
            } 
        }
        for(unsigned int si = 0; si < state_dimension; si++){
            states[(pi * this->state_dimension + si) * 3] = nearest_list[pi]->get_point()[si];
            states[(pi * this->state_dimension + si) * 3 + 1] = terminal_state[pi * this->state_dimension + si];
            states[(pi * this->state_dimension + si) * 3 + 1] = neural_sample_state[pi * this->state_dimension + si];
        }
        if(reset[pi] || shm_counter[pi] > 30) {
            // for(int si = 0; si < this -> state_dimension; si ++){
            //     current_state[si] = root -> get_point()[si];
            //     // std::cout << current_state[si] << " ";
            // }
            // // std::cout << std::endl;
            // smp_counter = 0;
            this->random_state(sample_state);
            sst_node_t* nearest = nearest_vertex(sample_state);
            for(int si = 0; si < this -> state_dimension; si ++){
                // current_state[si] = root -> get_point()[si];
                // std::cout << current_state[si] << " ";
                shm_current_state[pi * this->state_dimension + si] = nearest -> get_point()[si];
            }
            // std::cout << std::endl;
            shm_counter[pi] = 0;
        }

    }
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    delete prob;
    delete reset;
    delete duration;
    delete steer_start_state;

}



void deep_smp_mpc_sst_t::neural_step_batch(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias, const int NP)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * apply neural sampling from this sample
     * connect the start node to the sample node with trajectory optimization
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */

    

    // previous working code
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[NP*this->state_dimension]();
    double* terminal_state = new double[NP*this->state_dimension]();
    double* steer_start_state = new double[NP*this->state_dimension]();
    double prob = this->random_generator.uniform_random(0, 1);

    double* neural_sample_start_state = new double[NP*this->state_dimension]();
    std::vector<sst_node_t*> nearest_list(NP,NULL);
    for (unsigned int pi=0; pi<NP; pi++)
    {
        if (prob < goal_bias){
            // for (unsigned int i = 0; i < this->state_dimension; i++){
            //     sample_state[i] = goal_state[i];
            // }
            this->random_state(sample_state);
        }
        else{
            this->random_state(sample_state);
        }

        //std::cout << "pi: " << pi << std::endl;
        //std::cout << "random state: " << sample_state[0] << ", " << sample_state[1] << ", " << sample_state[2] << ", " << sample_state[3] << "]" << std::endl;
        sst_node_t* nearest = nearest_vertex(sample_state);
        //std::cout << "nearest point: " << nearest->get_point()[0] << ", " << nearest->get_point()[1] << ", " << nearest->get_point()[2] << ", " << nearest->get_point()[3] << "]" << std::endl;

        nearest_list[pi] = nearest;
        for (unsigned int si=0; si < this->state_dimension; si++)
        {
            neural_sample_start_state[pi*this->state_dimension+si] = nearest->get_point()[si];
        }
    }

    //  add neural sampling 
    
    #ifdef PROFILE
    auto profile_start = std::chrono::high_resolution_clock::now();
    #endif

    neural_sample_batch(system, neural_sample_start_state, neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection, NP); 
    #ifdef PROFILE
    auto profile_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> profile_duration = profile_stop - profile_start; 
    std::cout << "inside deep_smp_mpc_sst. neural_sample_batch takes " << profile_duration.count() << "s" << std::endl; 
    std::cout << "inside deep_smp_mpc_sst. 1000 steps of neural_sample_batch takes " << 1000*profile_duration.count() << "s" << std::endl; 
    #endif

    // get multiple copies for the steer_start_State
    for (unsigned int pi = 0; pi < NP; pi++)
    {
        // generate a rrandom probability for each problem
        prob = this->random_generator.uniform_random(0, 1);  // for goal_bias

        for (unsigned int si = 0; si < this->state_dimension; si++)
        {
            steer_start_state[pi*this->state_dimension+si] = neural_sample_start_state[pi*this->state_dimension+si];
        }

        // ##### different from server
        // if sampling goal, then set nerual_sample_state to goal
        if (prob < goal_bias)
        {
            for (unsigned int si=0; si < this->state_dimension; si++)
            {
                neural_sample_state[pi*this->state_dimension+si] = goal_state[si];
            }
        }
        // std::cout << "pi: " << pi << std::endl;
        // std::cout << "steer_start_state: " << steer_start_state[pi*this->state_dimension+0] << ", " << steer_start_state[pi*this->state_dimension+1] << ", " << steer_start_state[pi*this->state_dimension+2] << ", " << steer_start_state[pi*this->state_dimension+3] << "]" << std::endl;
        // std::cout << "neural_sample_state: " << neural_sample_state[pi*this->state_dimension+0] << ", " << neural_sample_state[pi*this->state_dimension+1] << ", " << neural_sample_state[pi*this->state_dimension+2] << ", " << neural_sample_state[pi*this->state_dimension+3] << "]" << std::endl;

    }
    // steer func
    double* duration = new double[NP]();
    //std::cout << "before steer_batch" << std::endl;

    #ifdef PROFILE
    profile_start = std::chrono::high_resolution_clock::now();
    #endif
    steer_batch(system, steer_start_state, neural_sample_state, terminal_state, integration_step, NP, duration);
    //std::cout << "after steer_batch" << std::endl;

    #ifdef PROFILE
    profile_stop = std::chrono::high_resolution_clock::now();
    profile_duration = profile_stop - profile_start; 
    //std::cout << "inside deep_smp_mpc_sst. steer_batch takes " << profile_duration.count() << "s" << std::endl; 
    //std::cout << "inside deep_smp_mpc_sst. 1000 steps of steer_batch takes " << 1000*profile_duration.count() << "s" << std::endl; 
    #endif


    // std::cout<<"duration:" << duration << std::endl;

    for (unsigned int pi = 0; pi < NP; pi ++)
    {
        //std::cout << "duration[pi]: " << duration[pi] << std::endl;
        if(duration[pi] > 0)
        {
            sst_node_t* nearest = nearest_list[pi];
            //std::cout << "nearest: " << nearest << std::endl;
            //std::cout << "nearest: " << nearest->get_point()[0] << ", " << nearest->get_point()[1] << ", " << nearest->get_point()[2] << ", " << nearest->get_point()[3] << "]" << std::endl;
            add_to_tree(terminal_state+pi*this->state_dimension, 0, nearest, duration[pi]);
        }

        // states: NP x STATE_DIM x 3
        for(unsigned int i = 0; i < state_dimension; i++){
            states[(pi*this->state_dimension+i)*3] = steer_start_state[pi*this->state_dimension+i];
            states[(pi*this->state_dimension+i)*3+1] = terminal_state[pi*this->state_dimension+i];
            states[(pi*this->state_dimension+i)*3+2] = neural_sample_state[pi*this->state_dimension+i];
        }

    }

    delete duration;
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    delete steer_start_state;
    delete neural_sample_start_state;

}


void deep_smp_mpc_sst_t::neural_step_single_batch(enhanced_system_t* system, double integration_step, torch::Tensor& env_vox, 
    bool refine, float refine_threshold, bool using_one_step_cost, bool cost_reselection, double* states, double goal_bias, const int NP)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * apply neural sampling from this sample
     * connect the start node to the sample node with trajectory optimization
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */

    

    // previous working code
    double* sample_state = new double[this->state_dimension]();
    double* neural_sample_state = new double[NP*this->state_dimension]();
    double* terminal_state = new double[NP*this->state_dimension]();
    double* steer_start_state = new double[NP*this->state_dimension]();
    double prob = this->random_generator.uniform_random(0, 1);


    if (prob < goal_bias){
        // for (unsigned int i = 0; i < this->state_dimension; i++){
        //     sample_state[i] = goal_state[i];
        // }
        this->random_state(sample_state);
    }
    else{
        this->random_state(sample_state);
    }
    sst_node_t* nearest = nearest_vertex(sample_state);

    //  add neural sampling 
    
    #ifdef PROFILE
    auto profile_start = std::chrono::high_resolution_clock::now();
    #endif

    neural_sample_batch(system, nearest->get_point(), neural_sample_state, env_vox, refine, refine_threshold, using_one_step_cost, cost_reselection, NP); 
    #ifdef PROFILE
    auto profile_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> profile_duration = profile_stop - profile_start; 
    std::cout << "inside deep_smp_mpc_sst. neural_sample_batch takes " << profile_duration.count() << "s" << std::endl; 
    std::cout << "inside deep_smp_mpc_sst. 1000 steps of neural_sample_batch takes " << 1000*profile_duration.count() << "s" << std::endl; 
    #endif

    // get multiple copies for the steer_start_State
    for (unsigned int pi = 0; pi < NP; pi++)
    {
        // generate a rrandom probability for each problem
        prob = this->random_generator.uniform_random(0, 1);  // for goal_bias

        for (unsigned int si = 0; si < this->state_dimension; si++)
        {
            steer_start_state[pi*this->state_dimension+si] = nearest->get_point()[si];
        }

        // ##### different from server
        // if sampling goal, then set nerual_sample_state to goal
        if (prob < goal_bias)
        {
            for (unsigned int si=0; si < this->state_dimension; si++)
            {
                neural_sample_state[pi*this->state_dimension+si] = goal_state[si];
            }
        }
    }
    // steer func
    double* duration = new double[NP]();
    
    #ifdef PROFILE
    profile_start = std::chrono::high_resolution_clock::now();
    #endif
    steer_batch(system, steer_start_state, neural_sample_state, terminal_state, integration_step, NP, duration);
    #ifdef PROFILE
    profile_stop = std::chrono::high_resolution_clock::now();
    profile_duration = profile_stop - profile_start; 
    std::cout << "inside deep_smp_mpc_sst. steer_batch takes " << profile_duration.count() << "s" << std::endl; 
    std::cout << "inside deep_smp_mpc_sst. 1000 steps of steer_batch takes " << 1000*profile_duration.count() << "s" << std::endl; 
    #endif


    // std::cout<<"duration:" << duration << std::endl;

    for (unsigned int pi = 0; pi < NP; pi ++)
    {
        if(duration[pi] > 0)
        {
            add_to_tree(terminal_state+pi*this->state_dimension, 0, nearest, duration[pi]);
        }

        // states: NP x STATE_DIM x 3
        for(unsigned int i = 0; i < state_dimension; i++){
            states[(pi*this->state_dimension+i)*3] = nearest->get_point()[i];
            states[(pi*this->state_dimension+i)*3+1] = terminal_state[pi*this->state_dimension+i];
            states[(pi*this->state_dimension+i)*3+2] = neural_sample_state[pi*this->state_dimension+i];
        }

    }

    delete duration;
    delete sample_state;
    delete neural_sample_state;
    delete terminal_state;
    delete steer_start_state;

}
