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

#include "systems/quadrotor_obs.hpp"
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include <cstdio>

#define MIN_X -5
#define MAX_X 5
#define MIN_Q -1
#define MAX_Q 1

#define MIN_V -1.
#define MAX_V 1.

#define MASS_INV 1.
#define BETA 1.
#define EPS 2.107342e-08
#define MAX_QUATERNION_NORM_ERROR 1e-9

#define g 9.81

#define MIN_C1 -15
#define MAX_C1 -5.
#define MIN_C -1
#define MAX_C 1.

void quadrotor_obs_t::enforce_bounds_SO3(double *qstate){
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

bool quadrotor_obs_t::propagate(
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

bool quadrotor_obs_t::valid_state(){
     /** Quaternion to rotation matrix
     *  https://www.mathworks.com/help/fusion/ref/quaternion.rotmat.html    
     * also check 
     * https://ompl.kavrakilab.org/src_2omplapp_2apps_2QuadrotorPlanning_8cpp_source.html#l00069
     * Convert current configuration to 3D min max and then compute the bounding box
     * Key points on frames
     */
    for(int si = 0; si < 3; si++){
        if(temp_state[si] > MAX_X || temp_state[si] < MIN_X){
            return false;
        }
    }

    double a = temp_state[6],
           b = temp_state[3],
           c = temp_state[4],
           d = temp_state[5];
    std::vector<double> min_max = {MAX_X * 10, // min_x
                                   MIN_X * 10, // max_x
                                   MAX_X * 10, // min_y
                                   MIN_X * 10, // max_y
                                   MAX_X * 10, // min_z
                                   MIN_X * 10}; // max_z
    for(unsigned f_i = 0; f_i < frame.size(); f_i++){
        double x = (2 * a * a - 1 + 2 * b * b) * frame.at(f_i).at(0) +
                   (2 * b * c + 2 * a * d) * frame.at(f_i).at(1) +
                   (2 * b * d - 2 * a * c) * frame.at(f_i).at(2) +
                   /*world frame*/temp_state[0];
        // printf("%f, %f, %f\n", min_max.at(1), x, std::numeric_limits<double>::max());
        if(x < min_max.at(0)) {
            min_max.at(0) = x;
        } else if (x > min_max.at(1)) {
            min_max.at(1) = x;
        }

        double y = (2 * b * c - 2 * a * d) * frame.at(f_i).at(0) +
                   (2 * a * a - 1 + 2 * c * c) * frame.at(f_i).at(1) +
                   (2 * c * d + 2 * a * b) * frame.at(f_i).at(2) +
                   /*world frame*/temp_state[1];

        if(y < min_max.at(2)) {
            min_max.at(2) = y;
        } else if (y > min_max.at(3)) {
            min_max.at(3) = y;
        }

        double z = (2 * b * d  + 2 * a * c) * frame.at(f_i).at(0) +
                   (2 * c * d  - 2 * a * b) * frame.at(f_i).at(1) +
                   (2 * a * a - 1 + 2 * d * d) * frame.at(f_i).at(2) +
                   /*world frame*/temp_state[2];
        if(z < min_max.at(4)) {
            min_max.at(4) = z;
        } else if (z > min_max.at(5)) {
            min_max.at(5) = z;
        }
    }
    /** AABB
     * https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
     * AABB v.s. AABB
     * function intersect(a, b) {
     *  return (a.minX <= b.maxX && a.maxX >= b.minX) &&
     *          (a.minY <= b.maxY && a.maxY >= b.minY) &&
     *          (a.minZ <= b.maxZ && a.maxZ >= b.minZ);
     * }
     */
    bool current_validity = true;
    for(unsigned int oi = 0; oi < obs_min_max.size(); oi++){
        // printf("%f,%f,%f,%f;;;;",  min_max.at(0), min_max.at(1), obs_min_max.at(oi).at(0), obs_min_max.at(oi).at(1));
        // printf("%d, %d; %d, %d; %d, %d\n", 
        //     min_max.at(0) <= obs_min_max.at(oi).at(1), min_max.at(1) >= obs_min_max.at(oi).at(0),
        //     min_max.at(2) <= obs_min_max.at(oi).at(3), min_max.at(3) >= obs_min_max.at(oi).at(2),
        //     min_max.at(4) <= obs_min_max.at(oi).at(5), min_max.at(5) >= obs_min_max.at(oi).at(4));
        if(((min_max.at(0) <= obs_min_max.at(oi).at(1) && min_max.at(1) >= obs_min_max.at(oi).at(0)) &&
             (min_max.at(2) <= obs_min_max.at(oi).at(3) && min_max.at(3) >= obs_min_max.at(oi).at(2)) &&
             (min_max.at(4) <= obs_min_max.at(oi).at(5) && min_max.at(5) >= obs_min_max.at(oi).at(4)))){
                 current_validity = false;
                 break;
           }
    }
    return current_validity;
}

void quadrotor_obs_t::enforce_bounds(){
    // for R^3
    // for quaternion
    enforce_bounds_SO3(&temp_state[3]);
    // for v and w
    for(int si = 7; si < state_dimension; si++){
        if(temp_state[si] < MIN_V){
        temp_state[si] = MIN_V;
        }else if(temp_state[si] > MAX_V){
            temp_state[si] = MAX_V;
        }
    }

};

void quadrotor_obs_t::update_derivative(const double* control){
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

std::vector<std::pair<double, double> > quadrotor_obs_t::get_control_bounds() const{
    return {
            {MIN_C1, MAX_C1},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
            {MIN_C, MAX_C},
    };
}

std::vector<std::pair<double, double> > quadrotor_obs_t::get_state_bounds() const {
    return {
            {MIN_X, MAX_X},
            {MIN_X, MAX_X},
            {MIN_X, MAX_X},

            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},
            {MIN_Q, MAX_Q},

            {MIN_V,MAX_V},
            {MIN_V,MAX_V},
            {MIN_V,MAX_V},

            {MIN_V,MAX_V},
            {MIN_V,MAX_V},
            {MIN_V,MAX_V}
    };
}

std::vector<bool> quadrotor_obs_t::is_circular_topology() const{
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

void quadrotor_obs_t::normalize(const double* state, double* normalized){
    for(int i = 0; i < 3; i++){
        normalized[i] = state[i] / MAX_X;
    }
    for(int i = 3; i < state_dimension; i++){
        normalized[i] = state[i];
    }
    enforce_bounds_SO3(&normalized[3]);
}

void quadrotor_obs_t::denormalize(double* normalized,  double* state){
    for(int i = 0; i < 3; i++){
        state[i] = normalized[i] * MAX_X; 
    }
    for(int i = 3; i < state_dimension; i++){
        state[i] = normalized[i]; 
    }
    enforce_bounds_SO3(&state[3]);

}

std::tuple<double, double> quadrotor_obs_t::visualize_point(const double* state, unsigned int state_dimension) const{
    return std::make_tuple(0, 0);
}


double quadrotor_obs_t::distance(const double* point1, const double* point2, unsigned int state_dimension){
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

double quadrotor_obs_t::get_loss(double* point1, const double* point2, double* weight){
    return distance(point1, point2, state_dimension);
}
