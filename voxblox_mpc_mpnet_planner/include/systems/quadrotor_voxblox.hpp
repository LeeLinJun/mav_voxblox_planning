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

#ifndef QUADROTOR_VOXBLOX_HPP
#define QUADROTOR_VOXBLOX_HPP
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include "systems/enhanced_system.hpp"
#include <cstdio>

#include <voxblox/core/esdf_map.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/integrator_utils.h>
#include <voxblox/utils/planning_utils.h>

#define EPS 2.107342e-08
#define MAX_QUATERNION_NORM_ERROR 1e-9

class quadrotor_voxblox_t : public enhanced_system_t
{
public:
	quadrotor_voxblox_t(
		double MIN_X, double MAX_X, 
		double MIN_Y, double MAX_Y, 
		double MIN_Z, double MAX_Z, 
		double MIN_V, double MAX_V, 
		double MIN_W, double MAX_W, 
		double MIN_C1, double MAX_C1, 
		double MIN_C, double MAX_C, 
		double integration_step,
		voxblox::Layer<voxblox::EsdfVoxel>* esdf_layer,
		double robot_radius)
		: MIN_X(MIN_X), MAX_X(MAX_X),
		  MIN_Y(MIN_Y), MAX_Y(MAX_Y),
		  MIN_Z(MIN_Z), MAX_Z(MAX_Z),
		  MIN_V(MIN_V), MAX_V(MAX_V),
		  MIN_W(MIN_W), MAX_W(MAX_W),
		  MIN_C1(MIN_C1), MAX_C1(MAX_C1),
		  MIN_C(MIN_C), MAX_C(MAX_C),
		  integration_step(integration_step),
		  interpolator_(esdf_layer),
		  robot_radius(robot_radius) {
		initialize_system();
		voxel_size_ = esdf_layer->voxel_size();
	}

	void initialize_system(){
		state_dimension = 13;
		control_dimension = 4;
		temp_state = new double[state_dimension]();
		deriv = new double[state_dimension]();

		u = new double[control_dimension]();
		qomega = new double[4]();
		validity = true;
		MIN_Q = -1;
		MAX_Q = 1;
		MASS_INV = 1; 
		BETA = 1;
	}


	virtual ~quadrotor_voxblox_t(){
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
	double robot_radius;
	double MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_Z, MAX_Z,
		   MIN_Q, MAX_Q,
		   MIN_V, MAX_V,
		   MIN_W, MAX_W, 
 
		   MASS_INV, 
		   BETA,
		   MIN_C1,
		   MAX_C1,
		   MIN_C,
		   MAX_C,
		   integration_step;
	constexpr static double g = 9.81;
	double voxel_size_;
	voxblox::Interpolator<voxblox::EsdfVoxel> interpolator_;

};
#endif