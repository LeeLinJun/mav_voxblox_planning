/**
 * @file quadrotor_obs.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2020 Linjun Li
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Model definition is from OMPL App Quadrotor Planning:
 * https://ompl.kavrakilab.org/classompl_1_1app_1_1QuadrotorPlanning.html
 */

#include "systems/quadrotor_voxblox.hpp"
#define _USE_MATH_DEFINES
#include <cmath>



void quadrotor_voxblox_t::enforce_bounds_SO3(double *qstate){
    //https://ompl.kavrakilab.org/SO3StateSpace_8cpp_source.html#l00183
    double nrmSqr = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3];
    double nrmsq = (std::fabs(nrmSqr - 1.0) > std::numeric_limits<double>::epsilon()) ? std::sqrt(nrmSqr) : 1.0;
    double error = std::abs(1.0 - nrmsq);
    if (error < EPS) {
        double scale = 2.0 / (1.0 + nrmsq);
        qstate[0] *= scale;
        qstate[1] *= scale;
        qstate[2] *= scale;
        qstate[3] *= scale;
    } else {
        if (nrmsq < 1e-6){
            for(int si = 0; si < 4; si++){
                qstate[si] = 0;
            }
            qstate[3] = 1;
        } else {
            double scale = 1.0 / std::sqrt(nrmsq);
            qstate[0] *= scale;
            qstate[1] *= scale;
            qstate[2] *= scale;
            qstate[3] *= scale;
        }
    }
}

bool quadrotor_voxblox_t::propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step){
    for(int si = 0; si < state_dimension; si++){
        temp_state[si] = start_state[si];
    }
    validity = true;
    for(int t = 0; t < num_steps; t++)
    {
        update_derivative(control);
        for(int si = 0; si < state_dimension; si++){
            temp_state[si] += deriv[si] * integration_step;
        }
        enforce_bounds();
        validity = validity && valid_state();
        if(validity){
            for(int si = 0; si < state_dimension; si++){
                result_state[si] = temp_state[si];
            }
        } else {
            break;
        }
    }
    
    return validity;
}

bool quadrotor_voxblox_t::valid_state(){
    Eigen::Vector3d robot_position(
        temp_state[0],
        temp_state[1],
        temp_state[2]
    );

    voxblox::FloatingPoint distance;
    bool success = interpolator_.getDistance(
        robot_position.cast<voxblox::FloatingPoint>(), &distance, false);
    if (!success) {
      return true;
    }
    // std::cout << distance <<"," << (bool)(robot_radius < distance) << std::endl;

    return robot_radius < distance;

}

void quadrotor_voxblox_t::enforce_bounds(){
    // for R^3
    // for quaternion
    enforce_bounds_SO3(&temp_state[3]);
    // for v and w
    for(int si = 7; si < 10; si++){
        if(temp_state[si] < MIN_V){
        temp_state[si] = MIN_V;
        }else if(temp_state[si] > MAX_V){
            temp_state[si] = MAX_V;
        }
    }
    for(int si = 10; si < state_dimension; si++){
        if(temp_state[si] < MIN_W){
        temp_state[si] = MIN_W;
        }else if(temp_state[si] > MAX_W){
            temp_state[si] = MAX_W;
        }
    }

};

void quadrotor_voxblox_t::update_derivative(const double* control){
    //https://ompl.kavrakilab.org/src_2omplapp_2apps_2QuadrotorPlanning_8cpp_source.html
    // enforce control
    if(control[0] > MAX_C1){
        u[0] = MAX_C1;
    } else if (control[0] < MIN_C1){
        u[0] = MIN_C1;
    } else {
        u[0] = control[0];
    }
    for(int i_u = 1; i_u < control_dimension; i_u++){
        if(control[i_u] > MAX_C){
            u[i_u] = MAX_C;
        } else if (control[i_u] < MIN_C){
            u[i_u] = MIN_C;
        } else {
            u[i_u] = control[i_u];
        }
    }
    // dx/dt = v
    deriv[0] = temp_state[7];
    deriv[1] = temp_state[8];
    deriv[2] = temp_state[9];
    qomega[0] = .5 * temp_state[10];
    qomega[1] = .5 * temp_state[11];
    qomega[2] = .5 * temp_state[12];
    qomega[3] = 0;
    enforce_bounds_SO3(qomega);
    double delta = temp_state[3] * qomega[0] + temp_state[4] * qomega[1] + temp_state[5] * qomega[2];
    // d theta / dt = omega
    deriv[3] = qomega[0] - delta * temp_state[3];
    deriv[4] = qomega[1] - delta * temp_state[4];
    deriv[5] = qomega[2] - delta * temp_state[5];
    deriv[6] = qomega[3] - delta * temp_state[6];
    // d v / dt = a 
    deriv[7] = MASS_INV * (-2*u[0]*(temp_state[6]*temp_state[4] + temp_state[3]*temp_state[5]) - BETA * temp_state[7]);
    deriv[8] = MASS_INV * (-2*u[0]*(temp_state[4]*temp_state[5] - temp_state[6]*temp_state[3]) - BETA * temp_state[8]);
    deriv[9] = MASS_INV * (-u[0]*(temp_state[6]*temp_state[6]-temp_state[3]*temp_state[3]-temp_state[4]*temp_state[4]+temp_state[5]*temp_state[5]) - BETA * temp_state[9]) - 9.81;
    // d omega / dt = alpha
    deriv[10] = u[1];
    deriv[11] = u[2];
    deriv[12] = u[3];

};

std::vector<std::pair<double, double> > quadrotor_voxblox_t::get_control_bounds() const{
    return {
            {MIN_C1, MAX_C1},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
    };
}

std::vector<std::pair<double, double> > quadrotor_voxblox_t::get_state_bounds() const {
    return {
            {MIN_X, MAX_X},
            {MIN_Y, MAX_Y},
            {MIN_Z, MAX_Z},

            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},

            {MIN_V, MAX_V},
            {MIN_V, MAX_V},
            {MIN_V, MAX_V},
 
            {MIN_W, MAX_W},
            {MIN_W, MAX_W},
            {MIN_W, MAX_W},
    };
}

std::vector<bool> quadrotor_voxblox_t::is_circular_topology() const{
    return {
            false,
            false,
            false,
            
            false,
            false,
            false,
            false,
            
            false,
            false,
            false,

            false,
            false,
            false
    };
}

void quadrotor_voxblox_t::normalize(const double* state, double* normalized){
    normalized[0] =  -1 + 2 * (state[0] - MIN_X)/ (MAX_X - MIN_X);
    normalized[1] =  -1 + 2 * (state[1] - MIN_Y)/ (MAX_Y - MIN_Y);
    normalized[2] =  -1 + 2 * (state[2] - MIN_Z)/ (MAX_Z - MIN_Z);

    for(int i = 3; i < state_dimension; i++){
        normalized[i] = state[i];
    }
    enforce_bounds_SO3(&normalized[3]);
}

void quadrotor_voxblox_t::denormalize(double* normalized,  double* state){
    state[0] = (normalized[0] + 1) / 2 * (MAX_X - MIN_X) + MIN_X;
    state[1] = (normalized[1] + 1) / 2 * (MAX_Y - MIN_Y) + MIN_Y;
    state[2] = (normalized[2] + 1) / 2 * (MAX_Z - MIN_Z) + MIN_Z;

    for(int i = 3; i < state_dimension; i++){
        state[i] = normalized[i]; 
    }
    enforce_bounds_SO3(&state[3]);

}


double quadrotor_voxblox_t::distance(const double* point1, const double* point2, unsigned int state_dimension){
    /**
     * In OMPL Model, StateSpace is [SE3StateSPace()*1, RealVectorStateSpace(6)*0.3]
     * Referenced OMPL Compound system SE3StateSpace: https://ompl.kavrakilab.org/SE3StateSpace_8cpp_source.html
     * In OMPL COmpoundStateSpace, distance is computed by https://ompl.kavrakilab.org/StateSpace_8cpp_source.html#l01068
     * distance = \sum{ weights_[i] * distance(subspace_i_state1, subspace_i_state2)}
     * where weights are 1.0 and 1.0: https://ompl.kavrakilab.org/SE3StateSpace_8h_source.html#l00113
    */

   /**
    * RealVectorStateSpace distance
    * https://ompl.kavrakilab.org/classompl_1_1base_1_1RealVectorStateSpace.html#a8226c880e4799cb219cadab1e601938b
    */ 
    double dist = 0.;
    for(int i = 0; i < 3; i++){
        dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    dist = sqrt(dist);
    /**
     * Distance between quaternion
     * https://ompl.kavrakilab.org/SO3StateSpace_8cpp_source.html#l00267
    */
    double dq  = 0.;
    for(int i = 3; i < 7; i++){
        dq  += point1[i] * point2[i] ;
    }
    dq = fabs(dq);
    if (dq > 1.0 - MAX_QUATERNION_NORM_ERROR){
        dq =  0.0;
    } else {
        dq = acos(dq);
    }

    double dist_v = 0.;
    for(int i = 7; i < 13; i++){
        dist_v += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    dist_v = sqrt(dist_v);
    /**
     * StateSpace has weights 1 * SE3 + 0.3 * Vel = 1 * R3 + 1 * SO3 + 0.3 * R6
     * https://ompl.kavrakilab.org/src_2omplapp_2apps_2QuadrotorPlanning_8cpp_source.html#l00099
    */
    return dist + dq + 0.3 * dist_v;
}

double quadrotor_voxblox_t::get_loss(double* point1, const double* point2, double* weight){
    return distance(point1, point2, state_dimension);
}
