#include "trajectory_optimizers/cem_cuda_quadrotor.hpp"
#include "systems/enhanced_system.hpp"

//#define PROFILE

#include <chrono>

// #define DEBUG
#define DT  2e-3
//#define MAX_T 1

#define PI  3.141592654f


#define DIM_STATE 13
#define DIM_CONTROL 4

#define NOBS 10

#define OBS_WIDTH 1.0

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

__constant__ double d_c_min[4] = {MIN_C1, MIN_C, MIN_C, MIN_C};
__constant__ double d_c_max[4] = {MAX_C1, MAX_C, MAX_C, MAX_C};

#define FRAME_SIZE 0.25
#define FRAME_LEN0 4
#define FRAME_LEN1 3


#define FRAME_LEN0 4
#define FRAME_LEN1 3
__constant__ double frame[FRAME_LEN0*FRAME_LEN1] = {FRAME_SIZE, 0, 0,
                            0, FRAME_SIZE, 0,
                            -FRAME_SIZE, 0, 0,
                            0, -FRAME_SIZE, 0};


#define OBS_PENALTY 1000.0

namespace trajectory_optimizers_quadrotor{
    __global__ void initCurand(curandState* state, unsigned long seed) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, idx, 0, &state[idx]);
    }


    __global__ 
    void set_statistics(double* d_mean_time, const double mean_time, double* d_mean_control, const double* mean_control, 
        double* d_std_control, const double* std_control, double* d_std_time, const double std_time, int NT){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        unsigned int id = np * NT + nt;
        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;// 0~NT * NP
        // printf("inside set_statistics. id: (%d, %d)\n", np, nt);
        // printf("inside set_statistics. mean_control: (%f, %f)\n", mean_control[0], mean_control[1]);

        d_mean_time[id] = mean_time;
        //d_mean_control[id] = mean_control;
        for (unsigned i=0; i < DIM_CONTROL; i++)
        {
            d_mean_control[id*DIM_CONTROL+i] = mean_control[i];
            d_std_control[id*DIM_CONTROL+i] = std_control[i];
        }

        d_std_time[id] = std_time;
        // printf("inside set_statistics. d_mean_time: %f\n", d_mean_time[id]);
        // printf("inside set_statistics. d_mean_control: %f, %f\n", d_mean_control[id*2], d_mean_control[id*2+1]);
        // printf("inside set_statistics. d_std_control: %f\n", d_std_control[id*2],d_std_control[id*2+1]);
        // printf("inside set_statistics. d_std_time: %f\n", d_std_time[id]);

        
    }

    __global__
    void set_start_state(double* temp_state, double* start, const int NS){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;
    
        for (unsigned i=0; i<DIM_STATE; i++)
        {
            temp_state[i + id*DIM_STATE] = start[i + np*DIM_STATE];
        }
        //printf("in set_start_state. id = %d,temp_state: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n", id, temp_state[id*DIM_STATE+0],temp_state[id*DIM_STATE+1],temp_state[id*DIM_STATE+2],temp_state[id*DIM_STATE+3],temp_state[id*DIM_STATE+4],temp_state[id*DIM_STATE+5],temp_state[id*DIM_STATE+6],temp_state[id*DIM_STATE+7],temp_state[id*DIM_STATE+8],temp_state[id*DIM_STATE+9],temp_state[id*DIM_STATE+10],temp_state[id*DIM_STATE+11],temp_state[id*DIM_STATE+12]);

        //printf("%d: %f, %f, %f, %f\n", id, temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W]);
    }

    __global__ 
    void sampling(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time, double MAX_T, const int NP, const int NS, const int NT, bool* active_mask,
        curandState* state){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        unsigned int id = np * NS * NT + ns * NT + nt;

        //printf("%d, %d, %d\n",np, ns, nt);
        //printf("inside sampling. id: (%d, %d, %d)\n", np, ns, nt);

        active_mask[np * NS + ns] = true;


        for (unsigned i=0; i<DIM_CONTROL; i++)
        {
            double c_i = std_control[(np * NT + nt)*DIM_CONTROL+i] * curand_normal(&state[id]) + mean_control[(np * NT + nt)*DIM_CONTROL+i];
            if (c_i > d_c_max[i])
            {
                c_i = d_c_max[i];
            }
            else if (c_i < d_c_min[i])
            {
                c_i = d_c_min[i];
            }
            control[(np * NS * NT + ns * NT + nt)*DIM_CONTROL+i] = c_i;
        }

        //printf("inside sampling. curand_normal control: %f, %f\n", c_v, c_w);
        //printf("inside sampling. std_control: %f, %f\n", std_control[(np * NT + nt)*2],std_control[(np * NT + nt)*2+1]);

        //printf("mean control:%f\n", mean_control[np * NT + nt]);


        //printf("inside sampling. curand_normal time: %f\n", curand_normal(&state[id]) + mean_time[np * NT + nt]);
        //printf("inside sampling. std_time: %f\n", std_time[np * NT + nt]);
        double t = std_time[np * NT + nt] * curand_normal(&state[id]) + mean_time[np * NT + nt];
        if(t < DT){
            t = 0;
        } else if (t > MAX_T) {
            t = MAX_T;
        }
        time[np * NS * NT + ns * NT + nt] = t;    

        //printf("c:%f, t:%f\n", c, t);
    }
    __device__
    void enforce_bounds_SO3(double *qstate){
        //https://ompl.kavrakilab.org/SO3StateSpace_8cpp_source.html#l00183
        double nrmSqr = qstate[0]*qstate[0] + qstate[1]*qstate[1] + qstate[2]*qstate[2] + qstate[3]*qstate[3];
        double nrmsq = (std::fabs(nrmSqr - 1.0) > 1e-10) ? std::sqrt(nrmSqr) : 1.0;
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

    __device__
    bool valid_state(const double* temp_state, const double* obs_min_max){
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
       double min_max_0 = MAX_X * 10;
       double min_max_1 = MIN_X * 10;
       double min_max_2 = MAX_X * 10;
       double min_max_3 = MIN_X * 10;
       double min_max_4 = MAX_X * 10;
       double min_max_5 = MIN_X * 10;
       for(unsigned f_i = 0; f_i < FRAME_LEN0; f_i++){
           double x = (2 * a * a - 1 + 2 * b * b) * frame[f_i*FRAME_LEN1+0] +
                      (2 * b * c + 2 * a * d) * frame[f_i*FRAME_LEN1+1] +
                      (2 * b * d - 2 * a * c) * frame[f_i*FRAME_LEN1+2] +
                      /*world frame*/temp_state[0];
           // printf("%f, %f, %f\n", min_max.at(1), x, std::numeric_limits<double>::max());
           if(x < min_max_0) {
               min_max_0 = x;
           } else if (x > min_max_1) {
               min_max_1 = x;
           }
   
           double y = (2 * b * c - 2 * a * d) * frame[f_i*FRAME_LEN1+0] +
                      (2 * a * a - 1 + 2 * c * c) * frame[f_i*FRAME_LEN1+1] +
                      (2 * c * d + 2 * a * b) * frame[f_i*FRAME_LEN1+2] +
                      /*world frame*/temp_state[1];
   
           if(y < min_max_2) {
               min_max_2 = y;
           } else if (y > min_max_3) {
               min_max_3 = y;
           }
   
           double z = (2 * b * d  + 2 * a * c) * frame[f_i*FRAME_LEN1+0] +
                      (2 * c * d  - 2 * a * b) * frame[f_i*FRAME_LEN1+1] +
                      (2 * a * a - 1 + 2 * d * d) * frame[f_i*FRAME_LEN1+2] +
                      /*world frame*/temp_state[2];
           if(z < min_max_4) {
               min_max_4 = z;
           } else if (z > min_max_5) {
               min_max_5 = z;
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
       for(unsigned int oi = 0; oi < NOBS; oi++){
           if(((min_max_0 <= obs_min_max[oi*6+1] && min_max_1 >= obs_min_max[oi*6+0]) &&
                (min_max_2 <= obs_min_max[oi*6+3] && min_max_3 >= obs_min_max[oi*6+2]) &&
                (min_max_4 <= obs_min_max[oi*6+5] && min_max_5 >= obs_min_max[oi*6+4]))){
                    current_validity = false;
                    break;
              }
       }
       return current_validity;
   }

    __device__
    void update_derivative(const double* temp_state, const double* control, double* deriv){
        //https://ompl.kavrakilab.org/src_2omplapp_2apps_2QuadrotorPlanning_8cpp_source.html
        // enforce control
        // dx/dt = v
        deriv[0] = temp_state[7];
        deriv[1] = temp_state[8];
        deriv[2] = temp_state[9];
        double qomega_0 = .5 * temp_state[10];
        double qomega_1 = .5 * temp_state[11];
        double qomega_2 = .5 * temp_state[12];
        double qomega_3 = 0;
        deriv[3] = qomega_0;
        deriv[4] = qomega_1;
        deriv[5] = qomega_2;
        deriv[6] = qomega_3;
        enforce_bounds_SO3(deriv+3);
        double delta = temp_state[3] * deriv[3] + temp_state[4] * deriv[4] + temp_state[5] * deriv[5];
        // d theta / dt = omega
        deriv[3] = deriv[3] - delta * temp_state[3];
        deriv[4] = deriv[4] - delta * temp_state[4];
        deriv[5] = deriv[5] - delta * temp_state[5];
        deriv[6] = deriv[6] - delta * temp_state[6];
        // d v / dt = a 
        deriv[7] = MASS_INV * (-2*control[0]*(temp_state[6]*temp_state[4] + temp_state[3]*temp_state[5]) - BETA * temp_state[7]);
        deriv[8] = MASS_INV * (-2*control[0]*(temp_state[4]*temp_state[5] - temp_state[6]*temp_state[3]) - BETA * temp_state[8]);
        deriv[9] = MASS_INV * (-control[0]*(temp_state[6]*temp_state[6]-temp_state[3]*temp_state[3]-temp_state[4]*temp_state[4]+temp_state[5]*temp_state[5]) - BETA * temp_state[9]) - 9.81;
        // d omega / dt = alpha
        deriv[10] = control[1];
        deriv[11] = control[2];
        deriv[12] = control[3];
    
    };
    __device__
    void enforce_bounds(double* temp_state){
        // for R^3
        // for quaternion
        enforce_bounds_SO3(&temp_state[3]);
        // for v and w
        for(int si = 7; si < DIM_STATE; si++){
            if(temp_state[si] < MIN_V){
            temp_state[si] = MIN_V;
            }else if(temp_state[si] > MAX_V){
                temp_state[si] = MAX_V;
            }
        }
    
    };
    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, 
        const int t_step, const int NS, const int NT, bool* active_mask,
        double* obs_min_max){
            unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int id = np * NS + ns;
            unsigned int control_id = np * NS * NT + ns * NT + t_step;  // NP x NS x NT x CONTROL_DIM
            //printf("Inside GPU propagate... %d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
                //printf("%d\n", id);

            double t = time[np * NS * NT + ns * NT + t_step];
            if (t < 0){
                t = 0;
            }
            int num_step = t / DT;                
            for(unsigned int i = 0; i < num_step; i++){
                if(!active_mask[id]){
                    break;
                }

                // update derivs
                update_derivative(temp_state+id*DIM_STATE, control+control_id*DIM_CONTROL, deriv+id*DIM_STATE);
                //printf("in propagate. id = %d, num_step=%d, deriv: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n", id, num_step, deriv[id*DIM_STATE+0],deriv[id*DIM_STATE+1],deriv[id*DIM_STATE+2],deriv[id*DIM_STATE+3],deriv[id*DIM_STATE+4],deriv[id*DIM_STATE+5],deriv[id*DIM_STATE+6],deriv[id*DIM_STATE+7],deriv[id*DIM_STATE+8],deriv[id*DIM_STATE+9],deriv[id*DIM_STATE+10],deriv[id*DIM_STATE+11],deriv[id*DIM_STATE+12]);

                //TODO: control index is wrong, need to FIX CAR!!!

                // update states
                for (unsigned si=0; si<DIM_STATE; si++)
                {
                    temp_state[id*DIM_STATE+si] += DT*deriv[id*DIM_STATE+si];
                }
                //printf("in propagate. id = %d, num_step=%d, temp_state: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n", id, num_step, temp_state[id*DIM_STATE+0],temp_state[id*DIM_STATE+1],temp_state[id*DIM_STATE+2],temp_state[id*DIM_STATE+3],temp_state[id*DIM_STATE+4],temp_state[id*DIM_STATE+5],temp_state[id*DIM_STATE+6],temp_state[id*DIM_STATE+7],temp_state[id*DIM_STATE+8],temp_state[id*DIM_STATE+9],temp_state[id*DIM_STATE+10],temp_state[id*DIM_STATE+11],temp_state[id*DIM_STATE+12]);

                // enforce bounds
                enforce_bounds(temp_state+id*DIM_STATE);
                // validate_states
                bool valid = valid_state(&temp_state[id*DIM_STATE], obs_min_max);
                active_mask[id] = active_mask[id] && valid;
            }
           // printf("%d, %d: %f, %f, %f, %f\n", ns, np, temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W]);

    }

    __device__
    double distance(double* point1, double* point2)
    {
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
    
    __global__
    void get_loss(double* temp_state, double* loss, const int NS, double* goal_state, bool* active_mask){
        //printf("%d\n", id);
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;
        //printf("in get_loss. id = %d,  goal_state: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n", id, goal_state[np*DIM_STATE+0],goal_state[np*DIM_STATE+1],goal_state[np*DIM_STATE+2],goal_state[np*DIM_STATE+3],goal_state[np*DIM_STATE+4],goal_state[np*DIM_STATE+5],goal_state[np*DIM_STATE+6],goal_state[np*DIM_STATE+7],goal_state[np*DIM_STATE+8],goal_state[np*DIM_STATE+9],goal_state[np*DIM_STATE+10],goal_state[np*DIM_STATE+11],goal_state[np*DIM_STATE+12]);
        //printf("in get_loss. id = %d, active_mask: %d, temp_state: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n", id, active_mask[id], temp_state[id*DIM_STATE+0],temp_state[id*DIM_STATE+1],temp_state[id*DIM_STATE+2],temp_state[id*DIM_STATE+3],temp_state[id*DIM_STATE+4],temp_state[id*DIM_STATE+5],temp_state[id*DIM_STATE+6],temp_state[id*DIM_STATE+7],temp_state[id*DIM_STATE+8],temp_state[id*DIM_STATE+9],temp_state[id*DIM_STATE+10],temp_state[id*DIM_STATE+11],temp_state[id*DIM_STATE+12]);

        loss[id] = distance(temp_state+id * DIM_STATE, goal_state+np * DIM_STATE);
        //printf("in get_loss. loss[%d]: %f\n", id, loss[id]);
        if (!active_mask[id]) {
            loss[id] += OBS_PENALTY;
        }
        /*printf("%d, %d: %f, %f, %f, %f, loss: %f\n", 
            ns, np, 
            temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W],
            loss[id]);*/
        

    }

    
    __global__
    void get_approx_topk_loss(double* loss, const int NS, double* top_k_loss, int* top_k_loss_ind, const int N_ELITE){
        //printf("%d\n", id);
        /**
        * #TODO
        * this uses the id to map to one of the k buckets, and then only find the min loss for that index.
        * this is approximate as the result may not be the top k.
        * for consistency against previous method, our inputs are of the following dimension:
        *       top_k_loss: NPxNS
        *       top_k_loss_ind: NPxNS
        * Since we have NP problems, our bucket is of size: NP x N_ELITE
        *       grid size: (1, 1, 1)
        *       block size: (NP, 1, NE)
        */
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ne = blockIdx.z * blockDim.z + threadIdx.z;

        // loop over NE to find best k
        double min_loss = 10*OBS_PENALTY;
        int min_loss_ind = -1;
        for (unsigned int ns_div_ne = 0; ns_div_ne < NS/N_ELITE; ns_div_ne++)
        {
            unsigned int ns = ns_div_ne * N_ELITE + ne;
            if (ns >= NS)
            {
                continue;
            }
            if (loss[np*NS + ns] < min_loss)
            {
                min_loss = loss[np*NS + ns];
                min_loss_ind = ns;
            }
        }
        // copy the min loss to the bucket
        top_k_loss[np*NS+ne] = min_loss;
        top_k_loss_ind[np*NS+ne] = min_loss_ind;
    }


    __global__
    void update_statistics(double* control, double* time, double* mean_control, double* mean_time, double* std_control, double* std_time,
        int* loss_ind, double* loss, int NP, int NS, int NT, int N_ELITE, double* best_u, double* best_t, double* sum_control, double* ss_control){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        unsigned int s_id = np * NT + nt;

        //printf("inside update_statistics. id: (%d, %d)\n", np, nt);

        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

        // initialize
        for (int ci = 0; ci < DIM_CONTROL; ci++)
        {
            sum_control[s_id*DIM_CONTROL+ci] = 0.;
            ss_control[s_id*DIM_CONTROL+ci] = 0.;
        }
        double sum_time = 0., ss_time = 0.;
        for(int i = 0; i < N_ELITE; i++){
            //printf("inside update_statistics. N_ELITE: %d\n", N_ELITE);
            //printf("inside update_statistics. elite_i: %d\n", i);
            unsigned int id = np * NS * NT + loss_ind[np * NS + i] * NT + nt;
            //printf("inside update_statistics. loss_ind: %d\n", loss_ind[np * NS + i]);
            //printf("inside update_statistics. id: %d\n", id);
            for (int ci = 0; ci < DIM_CONTROL; ci++)
            {
                sum_control[s_id*DIM_CONTROL+ci] += control[id*DIM_CONTROL+ci];
                ss_control[s_id*DIM_CONTROL+ci] += control[id*DIM_CONTROL+ci] * control[id*DIM_CONTROL+ci];
            }

            sum_time += time[id];
            ss_time += time[id] * time[id];
        }
        // printf("%f,%f\n",ss_control, ss_time);
        for (unsigned ci = 0; ci < DIM_CONTROL; ci++)
        {
            mean_control[s_id*DIM_CONTROL+ci] = sum_control[s_id*DIM_CONTROL+ci] / N_ELITE;
            //printf("mean_control[%d] = %f\n",s_id*DIM_CONTROL+ci, mean_control[s_id*DIM_CONTROL+ci]);

            double std_control_square = ss_control[s_id*DIM_CONTROL+ci] / N_ELITE - mean_control[s_id*DIM_CONTROL+ci] * mean_control[s_id*DIM_CONTROL+ci];

            if (std_control_square < 1e-5)
            {
                std_control_square = 1e-5;
            }
            std_control[s_id*DIM_CONTROL+ci] = sqrt(std_control_square);
            //printf("std_control[%d] = %f\n",s_id*DIM_CONTROL+ci, std_control[s_id*DIM_CONTROL+ci]);

        }
        mean_time[s_id] = sum_time / N_ELITE;

        double std_time_square = ss_time / N_ELITE - mean_time[s_id] * mean_time[s_id];
        if (std_time_square < 1e-5)
        {
            std_time_square = 1e-5;
        }
        std_time[s_id] = sqrt(std_time_square);
        
        // --- update best_u and best_t
        for (unsigned ci = 0; ci < DIM_CONTROL; ci++)
        {
            best_u[s_id*DIM_CONTROL+ci] = control[(np * NS * NT + loss_ind[np * NS] * NT + nt)*DIM_CONTROL+ci];
            //printf("best_u[%d]=%f\n", s_id*DIM_CONTROL+ci, best_u[s_id*DIM_CONTROL+ci]);
        }

        best_t[s_id] = time[np * NS * NT + loss_ind[np * NS] * NT + nt];

    }
    

    CEM_CUDA_quadrotor::CEM_CUDA_quadrotor(enhanced_system_t* model, unsigned int num_of_problems, unsigned int number_of_samples, unsigned int number_of_t,
        unsigned int number_of_elite,  double converge_r,
        std::vector<std::vector<double>>& _obs_list,
        double* control_means, double* control_stds, 
        double time_means, double time_stds, double max_duration,
        double integration_step, double* loss_weights, unsigned int max_iteration, bool verbose, double step_size)
        : trajectory_optimizers::CEM(model, number_of_samples, number_of_t,
            number_of_elite, converge_r, 
            control_means, control_stds, 
            time_means, time_stds, max_duration,
            integration_step, loss_weights, max_iteration, verbose, step_size)
    {
        /*
        * TODO:
        * for parent class, remove redundent members initialization and creation
        */

        system = model;
        this -> NP = num_of_problems;
        this -> NS = number_of_samples;
        this -> NT = number_of_t;
        this -> N_ELITE = number_of_elite;
        mu_u0 = new double[DIM_CONTROL];
        std_u0 = new double[DIM_CONTROL];
        for (unsigned i=0; i < DIM_CONTROL; i++)
        {
            mu_u0[i] = control_means[i];
            std_u0[i] = control_stds[i];
        }
        //mu_u0 = control_means;
        //std_u0 = control_stds;

        mu_t0 = time_means;
        std_t0 = time_stds;
        this -> max_duration = max_duration;
        s_dim = system -> get_state_dimension();
        c_dim = system -> get_control_dimension();
        dt = integration_step;

        // states for rolling
        this -> verbose = verbose;
        converge_radius = converge_r;
        
        // util variables for update statistics
        step_size = step_size;
        it_max = max_iteration;
        weight = new double[s_dim];
        for(unsigned int si = 0; si < s_dim; si++){
            weight[si] = loss_weights[si];
        }

        // for CUDA here
        //printf("setup...\n");
        best_u = new double[NP*NT*DIM_CONTROL];
        best_t = new double[NP*NT];
        //best_ut = (double*) malloc(2 * NP * NT /*time + control*/ * sizeof(double));   // 2 x NP x NT
        //cudaMalloc(&d_best_ut, NP * NT * 2 * sizeof(double)); 
        cudaMalloc(&d_mu_u0, DIM_CONTROL * sizeof(double)); 
        cudaMalloc(&d_std_u0, DIM_CONTROL * sizeof(double)); 
        cudaMemcpy(d_mu_u0, mu_u0, DIM_CONTROL * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_std_u0, std_u0, DIM_CONTROL * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_best_u, NP * NT * DIM_CONTROL * sizeof(double)); 
        cudaMalloc(&d_best_t, NP * NT * sizeof(double)); 

        // temp state, derivative, control, time samples
            // temp_state = (double*) malloc(NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_temp_state, NP * NS * DIM_STATE * sizeof(double)); 
        cudaMalloc(&d_deriv, NP * NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_control, NP * NS * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_time, NP * NS * NT * sizeof(double));
        // for sampling statistics
        cudaMalloc(&d_mean_time, NP * NT * sizeof(double));
        cudaMalloc(&d_mean_control, NP * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_std_control, NP * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_std_time, NP * NT * sizeof(double));
        // for update_statistics
        cudaMalloc(&d_sum_control, NP*NT*DIM_CONTROL*sizeof(double));
        cudaMalloc(&d_ss_control, NP*NT*DIM_CONTROL*sizeof(double));

        // for cem
        cudaMalloc(&d_loss, NP * NS * sizeof(double));

        cudaMalloc(&d_top_k_loss, NP * NS * sizeof(double)); 

        cudaMalloc(&d_loss_ind, NP * NS * sizeof(int));
        //loss_ind = (int*) malloc(NP * NS * sizeof(int));
        loss_ind = new int[NP*NS]();
        //memset(loss_ind, 0, NP * NS  * sizeof(int));
        
        loss = new double[NP*NS]();
        loss_pair.resize(NS, std::make_pair(0., 0));

        // obstacles
        cudaMalloc(&d_obs_min_max, NOBS*6*sizeof(double));


        cudaMalloc(&d_active_mask, NP * NS * sizeof(bool));
        
        obs_min_max = new double[NOBS*6]();        

		for(unsigned oi=0;oi<_obs_list.size();oi++)
        {
            obs_min_max[oi*6] = _obs_list.at(oi).at(0) - OBS_WIDTH / 2;  obs_min_max[oi*6+1] = _obs_list.at(oi).at(0) + OBS_WIDTH / 2;
            obs_min_max[oi*6+2] = _obs_list.at(oi).at(1) - OBS_WIDTH / 2; obs_min_max[oi*6+3] = _obs_list.at(oi).at(1) + OBS_WIDTH / 2;
            obs_min_max[oi*6+4] = _obs_list.at(oi).at(2) - OBS_WIDTH / 2; obs_min_max[oi*6+5] = _obs_list.at(oi).at(2) + OBS_WIDTH / 2;
        }
        //std::cout << "after initialization" << std::endl;    
        cudaMemcpy(d_obs_min_max, obs_min_max, sizeof(double) * NOBS * 6, cudaMemcpyHostToDevice);
        // for multiple start
        cudaMalloc(&d_start_state, NP * DIM_STATE * sizeof(double));
        cudaMalloc(&d_goal_state, NP * DIM_STATE * sizeof(double));

        // initiate curand
        cudaMalloc((void**)&devState,NP * NS * NT * sizeof(curandState));
        initCurand << <(NP * NS * NT + 31) / 32, 32 >> > (devState, 42);
        
        //printf("done, execution:\n");

    }
    void CEM_CUDA_quadrotor::solve(const double* start, const double* goal, double* best_u, double* best_t){
        // auto begin = std::chrono::system_clock::now();
        // start and goal should be NP * DIM_STATE

        #ifdef PROFILE
        auto profile_start = std::chrono::high_resolution_clock::now();
        #endif
        cudaMemcpy(d_start_state, start, NP * DIM_STATE * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_goal_state, goal, NP * DIM_STATE * sizeof(double), cudaMemcpyHostToDevice);
        //thrust::device_ptr<double> time_ptr(d_time);
        //thrust::device_ptr<double> control_ptr(d_control);
        #ifdef PROFILE

        auto profile_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> profile_duration = profile_stop - profile_start; 
        std::cout << "inside cem_cuda:solve. cudaMemcpy start & goal takes " << profile_duration.count() << "s" << std::endl; 
        std::cout << "inside cem_cuda:solve. 1000 steps of cudaMemcpy start & goal takes " << 1000*profile_duration.count() << "s" << std::endl; 
        #endif        
        dim3 grid(1, 1, 1);
        dim3 grid_s(1, NS, 1);

        dim3 block_pt(NP, 1, NT);
        dim3 block_p(NP, 1, 1);

        //thrust::device_ptr<double> loss_ptr(d_loss);
        //thrust::device_ptr<int> loss_ind_ptr(d_loss_ind);
        //init mean
        //printf("%f,%f,%f,%f\n", mu_t0, std_t0, mu_u0, std_u0);
        #ifdef PROFILE

        profile_start = std::chrono::high_resolution_clock::now();
        #endif

        set_statistics<<<grid, block_pt>>>(d_mean_time, mu_t0, d_mean_control, d_mu_u0, d_std_control, d_std_u0, d_std_time, std_t0, NT);

        #ifdef PROFILE
        profile_stop = std::chrono::high_resolution_clock::now();
        profile_duration = profile_stop - profile_start; 
        std::cout << "inside cem_cuda:solve. set_statistics takes " << profile_duration.count() << "s" << std::endl; 
        std::cout << "inside cem_cuda:solve. 1000 steps of set_statistics takes " << 1000*profile_duration.count() << "s" << std::endl; 
        #endif
        // double min_loss = 1e5;
        // double tmp_min_loss = 2e5;
        // auto init_end = std::chrono::system_clock::now();
        //std::cout<< "start" <<std::endl;
       

        for(unsigned int it = 0; it < it_max; it ++){
            //std::cout << "==============iteration:  " << it << "==================" << std::endl; 
            set_start_state<<<grid_s, block_p>>>(d_temp_state, d_start_state, NS);
            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            sampling << <grid_s, block_pt >> > (d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time, max_duration, NP, NS, NT, d_active_mask, devState);
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. sampling takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of sampling takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif
            //std::cout<< "start of propagation..." <<std::endl;


            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            for(unsigned int t_step = 0; t_step < NT; t_step++){
                propagate<<<grid_s, block_p >>>(d_temp_state, d_control, d_time, d_deriv, t_step, NS, NT, d_active_mask, d_obs_min_max);
            }
            #ifdef PROFILE

            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            //std::cout << "inside cem_cuda:solve. propagate takes " << profile_duration.count() << "s" << std::endl; 
            //std::cout << "inside cem_cuda:solve. 1000 steps of propagate takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif


            //std::cout<< "end of propagation." <<std::endl;
            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            get_loss<<< grid_s, block_p >>>(d_temp_state, d_loss, NS, d_goal_state, d_active_mask);
            
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. get_loss takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of get_loss takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif
            //std::cout<< "start of sorting..." <<std::endl;
            
            //** Below uses approximate top-k method to bypass the memcpy overhead
            #ifdef PROFILE
            profile_start = std::chrono::high_resolution_clock::now();
            #endif

            dim3 grid_topk(1, 1, 1);    
            dim3 block_topk(NP, 1, N_ELITE);
            get_approx_topk_loss <<< grid_topk, block_topk >>>(d_loss, NS, d_top_k_loss, d_loss_ind, N_ELITE);
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. get_approx_topk_loss takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of get_approx_topk_loss takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif

            //** End of approximate top-k



            //std::cout<< "end of sorting." <<std::endl;
            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            //std::cout << "updating statistics..." << std::endl;
            update_statistics<<<grid, block_pt >>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time,
                d_loss_ind,  d_loss, NP, NS, NT, N_ELITE, d_best_u, d_best_t, d_sum_control, d_ss_control);
            
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. update_statistics takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of update_statistics takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif
            //std::cout << "updated statistics." << std::endl;

            //update_statistics<<<grid, block_pt >>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time,
            //    thrust::raw_pointer_cast(loss_ind_ptr),  thrust::raw_pointer_cast(loss_ptr), NP, NS, NT, N_ELITE, d_best_ut);

            ////std::cout<< "update" <<std::endl;
            // for (unsigned int p = 0; p < NP; p++) {
            //     cudaMemcpy(&tmp_min_loss, thrust::raw_pointer_cast(loss_ptr + NS * p), sizeof(double), cudaMemcpyDeviceToHost);

            //     //printf("p=%d, %f,\t%f\n", p, tmp_min_loss, min_loss);

            // }
            //printf("\n");
        }
        // auto done = std::chrono::system_clock::now();
        //printf("done\n");



        // auto duration_init = std::chrono::duration_cast<std::chrono::microseconds>(init_end-begin);
        // auto duration_exec = std::chrono::duration_cast<std::chrono::microseconds>(done-init_end);
        //printf("init:%f\nexec:%f\n",double(duration_init.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den,
            // double(duration_exec.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den);
    

        //std::cout << "copying from d_best_ut to best_u and best_t...\n" << std::endl;
        #ifdef PROFILE
        profile_start = std::chrono::high_resolution_clock::now();
        #endif
        //std::cout << "copying best u and t..." << std::endl;

        cudaMemcpy(best_u, d_best_u, NP * NT * DIM_CONTROL * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(best_t, d_best_t, NP * NT * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef PROFILE
        profile_stop = std::chrono::high_resolution_clock::now();
        profile_duration = profile_stop - profile_start; 
        std::cout << "inside cem_cuda:solve. cudaMemcpy of best_ut takes " << profile_duration.count() << "s" << std::endl; 
        std::cout << "inside cem_cuda:solve. 1000 steps of cudaMemcpy of best_ut takes " << 1000*profile_duration.count() << "s" << std::endl; 
        #endif


        //std::cout << "copied from d_best_ut to best_u and best_t.\n" << std::endl;

        // printf("control = [");
        //  for (unsigned int pi = 0; pi < NP; pi ++)
        //  {
        //      for(unsigned int it = 0; it < NT; it ++){
        //          printf("%f,", best_u[pi*NT+it]);
        //      }
    
        //  }
        //  printf("]\ntime = [");
        //  for (unsigned int pi = 0; pi < NP; pi ++)
        //  {
        //      for(unsigned int it = 0; it < NT; it ++){
        //          printf("%f,", best_t[pi*NT+it]);
        //      }
        //      printf("]\n");
        //      // return d_control;
    
        //  }
    }

    
    unsigned int CEM_CUDA_quadrotor::get_control_dimension(){
        return c_dim * NT;
    }

    unsigned int CEM_CUDA_quadrotor::get_num_step(){
        return NT;
    }
}