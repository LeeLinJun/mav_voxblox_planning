/**
 * @file quadrotor_obs.hpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2020 Linjun Li
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Model definition is from OMPL App Quadrotor Planning:
 * https://ompl.kavrakilab.org/classompl_1_1app_1_1QuadrotorPlanning.html
 */

#ifndef SPARSE_QUADROTOR_OBS_HPP
#define SPARSE_QUADROTOR_OBS_HPP
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include "systems/enhanced_system.hpp"
#define frame_size 0.25
#include <cstdio>

class quadrotor_obs_t : public enhanced_system_t
{
public:
	quadrotor_obs_t(){
		state_dimension = 13;
		control_dimension = 4;
		temp_state = new double[state_dimension]();
		deriv = new double[state_dimension]();

		u = new double[control_dimension]();
		qomega = new double[4]();
		validity = true;
	}
	quadrotor_obs_t(std::vector<std::vector<double>> _obs_list, double width){
		state_dimension = 13;
		control_dimension = 4;
		temp_state = new double[state_dimension]();
		deriv = new double[state_dimension]();

		u = new double[control_dimension]();
		qomega = new double[4]();
		validity = true;
		frame = {{frame_size, 0, 0},
				 {0, frame_size, 0},
				 {-frame_size, 0, 0},
				 {0, -frame_size, 0}};
		// copy the items from _obs_list to obs_list
		for(unsigned int oi = 0; oi < _obs_list.size(); oi++){
			std::vector<double> min_max_i = {_obs_list.at(oi).at(0) - width / 2, _obs_list.at(oi).at(0) + width / 2,
											 _obs_list.at(oi).at(1) - width / 2, _obs_list.at(oi).at(1) + width / 2,
											 _obs_list.at(oi).at(2) - width / 2, _obs_list.at(oi).at(2) + width / 2};// size = 6
			this -> obs_min_max.push_back(min_max_i); // size = n_o (* 6)
		}
	}

	virtual ~quadrotor_obs_t(){
		delete[] temp_state;
		delete[] deriv;
		delete[] qomega;
		delete[] u;
		// obs_list.clear();
	}
	/**
	 * @copydoc enhanced_system_t::distance(double*, double*)
	 */
	static double distance(const double* point1, const double* point2, unsigned int);

	/**
	 * @copydoc enhanced_system_t::propagate(double*, double*, int, int, double*, double& )
	 */
	virtual bool propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step);
	
	/**
	 * @copydoc enhanced_system_t::enforce_bounds()
	 */
	virtual void enforce_bounds();
	
	/**
	 * @copydoc enhanced_system_t::valid_state()
	 */
	virtual bool valid_state();
	
	/**
	 * @copydoc enhanced_system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	/**
	 * enforce bounds for quaternions
	 * copied from ompl: https://ompl.kavrakilab.org/classompl_1_1base_1_1SO3StateSpace.html#a986034ceebbc859163bcba7a845b868a
	 * Details:
	 * https://ompl.kavrakilab.org/SO3StateSpace_8cpp_source.html
	 * SO3StateSpace.cpp:183
	 */
	void enforce_bounds_SO3(double* qstate);

	/**
	 * @copydoc enhanced_system_t::get_state_bounds()
	 */
	std::vector<std::pair<double, double>> get_state_bounds() const override;
    
	/**
	 * @copydoc enhanced_system_t::get_control_bounds()
	 */
	std::vector<std::pair<double, double>> get_control_bounds() const override;

	/**
	 * @copydoc enhanced_system_t::is_circular_topology()
	 */
    std::vector<bool> is_circular_topology() const override;

	/**
	 * normalize state to [-1,1]^13
	 */
	void normalize(const double* state, double* normalized);
	
	/**
	 * denormalize state back
	 */
	void denormalize(double* normalized,  double* state);
	
	/**
	 * get loss for cem-mpc solver
	 */
	double get_loss(double* point1, const double* point2, double* weight);
 	
	/**
	 * obstacle lists vector<vector<>>
	 */
	std::vector<std::vector<double>> obs_list;

protected:
	double* deriv;
	void update_derivative(const double* control);
	double *u;
    double *qomega;
	bool validity = true;
	std::vector<std::vector<double>> frame;
	std::vector<std::vector<double>> obs_min_max;


};
#endif