#include "trajectory_optimizers/cem_cuda_car.hpp"
#include "systems/enhanced_system.hpp"

//#define PROFILE

#include <chrono>

// #define DEBUG
#define DT  2e-3
//#define MAX_T 1

#define PI  3.141592654f


#define DIM_STATE 3
#define DIM_CONTROL 2

#define NOBS 5


#define WIDTH 2.0
#define LENGTH 1.0
#define OBS_WIDTH 8.0

#define STATE_X 0
#define STATE_Y 1
#define STATE_THETA 2
#define MIN_X -25
#define MAX_X 25
#define MIN_Y -35
#define MAX_Y 35

#define MIN_V 0
#define MAX_V 2
#define MIN_W -0.5
#define MAX_W 0.5



#define OBS_PENALTY 1000.0
namespace trajectory_optimizers_car{

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
        d_mean_control[id*2] = mean_control[0];
        d_mean_control[id*2+1] = mean_control[1];

        //d_std_control[id] = std_control;
        d_std_control[id*2] = std_control[0];
        d_std_control[id*2+1] = std_control[1];

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
    
        temp_state[STATE_X + id*DIM_STATE] = start[STATE_X + np* DIM_STATE];
        temp_state[STATE_Y + id*DIM_STATE] = start[STATE_Y + np * DIM_STATE];
        temp_state[STATE_THETA + id*DIM_STATE] = start[STATE_THETA + np * DIM_STATE];
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

        double c_v = std_control[(np * NT + nt)*2] * curand_normal(&state[id]) + mean_control[(np * NT + nt)*2];
        double c_w = std_control[(np * NT + nt)*2+1] * curand_normal(&state[id]) + mean_control[(np * NT + nt)*2+1];

        //printf("inside sampling. curand_normal control: %f, %f\n", c_v, c_w);
        //printf("inside sampling. std_control: %f, %f\n", std_control[(np * NT + nt)*2],std_control[(np * NT + nt)*2+1]);

        //printf("mean control:%f\n", mean_control[np * NT + nt]);

        if (c_v > MAX_V) {
            c_v = MAX_V;
        }
        else if (c_v < MIN_V) {
            c_v = MIN_V;
        }
        if (c_w > MAX_W) {
            c_w = MAX_W;
        }
        else if (c_w < MIN_W) {
            c_w = MIN_W;
        }

        control[(np * NS * NT + ns * NT + nt)*2] = c_v;
        control[(np * NS * NT + ns * NT + nt)*2+1] = c_w;

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
    bool overlap(double b1corner_0_0,double b1corner_0_1,double b1corner_1_0,double b1corner_1_1,\
                 double b1corner_2_0,double b1corner_2_1,double b1corner_3_0,double b1corner_3_1,\
                 double b1axis_0_0,double b1axis_0_1,double b1axis_1_0,double b1axis_1_1,\
                 double b1orign_0,double b1orign_1,double b1ds_0,double b1ds_1,\
                 double b2corner_0_0,double b2corner_0_1,double b2corner_1_0,double b2corner_1_1,\
                 double b2corner_2_0,double b2corner_2_1,double b2corner_3_0,double b2corner_3_1,\
                 double b2axis_0_0,double b2axis_0_1,double b2axis_1_0,double b2axis_1_1,\
                 double b2orign_0,double b2orign_1,double b2ds_0,double b2ds_1)
    {
        //for (unsigned a = 0; a < 2; a++)
        double t = b1corner_0_0*b2axis_0_0 + b1corner_0_1*b2axis_0_1;
        double tMin = t;
        double tMax = t;
        //for (unsigned c = 1; c < 4; c++)
        t = b1corner_1_0*b2axis_0_0+b1corner_1_1+b2axis_0_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }        
        t = b1corner_2_0*b2axis_0_0+b1corner_2_1+b2axis_0_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }
        t = b1corner_3_0*b2axis_0_0+b1corner_3_1+b2axis_0_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }

        if ((tMin > (b2ds_0 + b2orign_0)) || (tMax < b2orign_0))
        {
            return false;
        }

        // a=1
        t = b1corner_0_0*b2axis_1_0 + b1corner_0_1*b2axis_1_1;
        tMin = t;
        tMax = t;
        //for (unsigned c = 1; c < 4; c++)
        t = b1corner_1_0*b2axis_1_0+b1corner_1_1+b2axis_1_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }        
        t = b1corner_2_0*b2axis_1_0+b1corner_2_1+b2axis_1_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }
        t = b1corner_3_0*b2axis_1_0+b1corner_3_1+b2axis_1_1;
        if (t < tMin)
        {
            tMin = t;
        }
        else if (t > tMax)
        {
            tMax = t;
        }

        if ((tMin > (b2ds_1 + b2orign_1)) || (tMax < b2orign_1))
        {
            return false;
        }
        return true;
    }

    __device__
    bool valid_state(double* temp_state, double* obs_corner_list, double* obs_axis_list, double* obs_ori_list)
    {
        if (temp_state[0] < MIN_X || temp_state[0] > MAX_X || temp_state[1] < MIN_Y || temp_state[1] > MAX_Y)
        {
            return false;
        }
        //std::cout << "inside  valid_state" << std::endl;
    
        double robot_corner_0_0; double robot_corner_0_1;
        double robot_corner_1_0; double robot_corner_1_1;
        double robot_corner_2_0; double robot_corner_2_1;
        double robot_corner_3_0; double robot_corner_3_1;

        double robot_axis_0_0; double robot_axis_0_1;
        double robot_axis_1_0; double robot_axis_1_1;
        double robot_ori_0; double robot_ori_1;
        double length_0; double length_1;
        double X1_0; double X1_1;
        double Y1_0; double Y1_1;
        
        X1_0=cos(temp_state[STATE_THETA])*(WIDTH/2.0);
        
        X1_1=-sin(temp_state[STATE_THETA])*(WIDTH/2.0);
        Y1_0=sin(temp_state[STATE_THETA])*(LENGTH/2.0);
        Y1_1=cos(temp_state[STATE_THETA])*(LENGTH/2.0);
    
        robot_corner_0_0 = temp_state[0]-X1_0-Y1_0;
        robot_corner_1_0 = temp_state[0]+X1_0-Y1_0;
        robot_corner_2_0 = temp_state[0]+X1_0+Y1_0;
        robot_corner_3_0 = temp_state[0]-X1_0+Y1_0;
        
        robot_axis_0_0 = robot_corner_1_0 - robot_corner_0_0;
        robot_axis_1_0 = robot_corner_3_0 - robot_corner_0_0;


        robot_corner_0_1 = temp_state[1]-X1_1-Y1_1;
        robot_corner_1_1 = temp_state[1]+X1_1-Y1_1;
        robot_corner_2_1 = temp_state[1]+X1_1+Y1_1;
        robot_corner_3_1 = temp_state[1]-X1_1+Y1_1;

        robot_axis_0_1 = robot_corner_1_1 - robot_corner_0_1;
        robot_axis_1_1 = robot_corner_3_1 - robot_corner_0_1;

        length_0 = sqrt(robot_axis_0_0 * robot_axis_0_0 + robot_axis_0_1 * robot_axis_0_1);
        length_1 = sqrt(robot_axis_1_0 * robot_axis_1_0 + robot_axis_1_1 * robot_axis_1_1);
    
        robot_axis_0_0 = robot_axis_0_0 / length_0;
        robot_axis_0_1 = robot_axis_0_1 / length_0;
        robot_axis_1_0 = robot_axis_1_0 / length_1;
        robot_axis_1_1 = robot_axis_1_1 / length_1;

        // obtain the projection of the left-bottom corner to the axis, to obtain the minimal projection length
        robot_ori_0 = robot_corner_0_0 * robot_axis_0_0 + robot_corner_0_1 * robot_axis_0_1;
        robot_ori_1 = robot_corner_0_0 * robot_axis_1_0 + robot_corner_0_1 * robot_axis_1_1;
    
    
        for (unsigned i=0; i<NOBS; i++)
        {
            bool collision = true;
            // do checking in both direction (b1 -> b2, b2 -> b1). It is only collision if both direcions are collision
            collision = overlap(robot_corner_0_0, robot_corner_0_1, robot_corner_1_0, robot_corner_1_1,\
                                robot_corner_2_0, robot_corner_2_1, robot_corner_3_0, robot_corner_3_1,\
                                robot_axis_0_0, robot_axis_0_1, robot_axis_1_0, robot_axis_1_1,\
                                robot_ori_0, robot_ori_1, WIDTH, LENGTH,\
                                obs_corner_list[i*8],obs_corner_list[i*8+1],obs_corner_list[i*8+2],obs_corner_list[i*8+3],\
                                obs_corner_list[i*8+4],obs_corner_list[i*8+5],obs_corner_list[i*8+6],obs_corner_list[i*8+7],\
                                obs_axis_list[i*4],obs_axis_list[i*4+1],obs_axis_list[i*4+2],obs_axis_list[i*4+3],\
                                obs_ori_list[i*2],obs_ori_list[i*2+1], OBS_WIDTH, OBS_WIDTH);
            collision = collision&overlap(obs_corner_list[i*8],obs_corner_list[i*8+1],obs_corner_list[i*8+2],obs_corner_list[i*8+3],\
                                            obs_corner_list[i*8+4],obs_corner_list[i*8+5],obs_corner_list[i*8+6],obs_corner_list[i*8+7],\
                                            obs_axis_list[i*4],obs_axis_list[i*4+1],obs_axis_list[i*4+2],obs_axis_list[i*4+3],\
                                            obs_ori_list[i*2],obs_ori_list[i*2+1], OBS_WIDTH, OBS_WIDTH,\
                                            robot_corner_0_0, robot_corner_0_1, robot_corner_1_0, robot_corner_1_1,\
                                            robot_corner_2_0, robot_corner_2_1, robot_corner_3_0, robot_corner_3_1,\
                                            robot_axis_0_0, robot_axis_0_1, robot_axis_1_0, robot_axis_1_1,\
                                            robot_ori_0, robot_ori_1, WIDTH, LENGTH);
            if (collision)
            {
                return false;  // invalid state
            }
        }
        //std::cout << "after valid" << std::endl;
    
        return true;
    }

    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, 
        const int t_step, const int NS, const int NT, bool* active_mask,
        double* obs_corner_list, double* obs_axis_list, double* obs_ori_list){
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
                deriv[id*DIM_STATE] = cos(temp_state[2 + id*DIM_STATE]) * control[0 + control_id*DIM_CONTROL];
                deriv[1 + id*DIM_STATE] = -sin(temp_state[2 + id*DIM_STATE]) * control[0 + control_id*DIM_CONTROL];
                deriv[2 + id*DIM_STATE] = control[1 + control_id*DIM_CONTROL];
                // update states
                temp_state[id*DIM_STATE] += DT * deriv[id*DIM_STATE];
                temp_state[1 + id*DIM_STATE] += DT * deriv[1 + id*DIM_STATE];
                temp_state[2 + id*DIM_STATE] += DT * deriv[2 + id*DIM_STATE];
                // enforce bounds
                if (temp_state[STATE_THETA + id*DIM_STATE] > PI){
                    temp_state[STATE_THETA + id*DIM_STATE] -= 2 * PI;
                }else if(temp_state[STATE_THETA + id*DIM_STATE] < -PI){
                    temp_state[STATE_THETA + id*DIM_STATE] += 2 * PI;
                }
                // validate_states
                bool valid = valid_state(&temp_state[id*DIM_STATE], obs_corner_list, obs_axis_list, obs_ori_list);
                active_mask[id] = active_mask[id] && valid;
            }        
           // printf("%d, %d: %f, %f, %f, %f\n", ns, np, temp_state[id * DIM_STATE + STATE_X], temp_state[id * DIM_STATE + STATE_V], temp_state[id * DIM_STATE + STATE_THETA], temp_state[id * DIM_STATE + STATE_W]);

    }

    __global__
    void get_loss(double* temp_state, double* loss, const int NS, double* goal_state, bool* active_mask){
        //printf("%d\n", id);
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;

        loss[id] = sqrt((temp_state[id * DIM_STATE + STATE_X] - goal_state[np * DIM_STATE + STATE_X]) * (temp_state[id * DIM_STATE + STATE_X] - goal_state[np * DIM_STATE + STATE_X])\
            + (temp_state[id * DIM_STATE + STATE_Y] - goal_state[np * DIM_STATE + STATE_Y]) * (temp_state[id * DIM_STATE + STATE_Y] - goal_state[np * DIM_STATE + STATE_Y])\
            + (temp_state[id * DIM_STATE + STATE_THETA] - goal_state[np * DIM_STATE + STATE_THETA]) * (temp_state[id * DIM_STATE + STATE_THETA] - goal_state[np * DIM_STATE + STATE_THETA]));

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
        int* loss_ind, double* loss, int NP, int NS, int NT, int N_ELITE, double* best_u, double* best_t){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        //printf("inside update_statistics. id: (%d, %d)\n", np, nt);

        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        double sum_control_v = 0., sum_control_w = 0., sum_time = 0., ss_control_v = 0., ss_control_w = 0., ss_time = 0.;
        for(int i = 0; i < N_ELITE; i++){
            //printf("inside update_statistics. N_ELITE: %d\n", N_ELITE);
            //printf("inside update_statistics. elite_i: %d\n", i);
            unsigned int id = np * NS * NT + loss_ind[np * NS + i] * NT + nt;
            //printf("inside update_statistics. loss_ind: %d\n", loss_ind[np * NS + i]);
            //printf("inside update_statistics. id: %d\n", id);

            sum_control_v += control[id*2];
            ss_control_v += control[id*2] * control[id*2];
            sum_control_w += control[id*2+1];
            ss_control_w += control[id*2+1] * control[id*2+1];

            sum_time += time[id];
            ss_time += time[id] * time[id];
        }
        // printf("%f,%f\n",ss_control, ss_time);
        unsigned int s_id = np * NT + nt;
        mean_control[s_id*2] = sum_control_v / N_ELITE;
        mean_control[s_id*2+1] = sum_control_w / N_ELITE;
        mean_time[s_id] = sum_time / N_ELITE;
        double std_control_square_v = ss_control_v / N_ELITE - mean_control[s_id*2] * mean_control[s_id*2];

        if (std_control_square_v < 1e-5)
        {
            std_control_square_v = 1e-5;
        }
        double std_control_square_w = ss_control_w / N_ELITE - mean_control[s_id*2+1] * mean_control[s_id*2+1];
        if (std_control_square_w < 1e-5)
        {
            std_control_square_w = 1e-5;
        }
        std_control[s_id*2] = sqrt(std_control_square_v);
        std_control[s_id*2+1] = sqrt(std_control_square_w);

        //printf("inside update_statistics. ss_time: %f\n", ss_time);
        //printf("inside update_statistics. ss_time/N_ELITE: %f\n", ss_time/N_ELITE);

        double std_time_square = ss_time / N_ELITE - mean_time[s_id] * mean_time[s_id];
        if (std_time_square < 1e-5)
        {
            std_time_square = 1e-5;
        }
        std_time[s_id] = sqrt(std_time_square);
        
        //printf("inside update_statistics. ss_time: %f\n", ss_time);
        //printf("inside update_statistics. ss_time/N_ELITE: %f\n", ss_time/N_ELITE);
        //printf("inside update_statistics. std_time_square: %f\n", ss_time / N_ELITE - mean_time[s_id] * mean_time[s_id]);
        //printf("inside update_statistics. mean_time: %f\n", mean_time[s_id]);
        //printf("inside update_statistics. mean_time_square: %f\n", mean_time[s_id] * mean_time[s_id]);

        //printf("inside update_statistics. std_time: %f\n", std_time[s_id]);

        best_u[s_id*2] = control[(np * NS * NT + loss_ind[np * NS] * NT + nt)*2];
        best_u[s_id*2+1] = control[(np * NS * NT + loss_ind[np * NS] * NT + nt)*2+1];

        best_t[s_id] = time[np * NS * NT + loss_ind[np * NS] * NT + nt];
        //printf("inside update_statistics. best_t[s_id]: %f\n",  best_t[s_id]);
        //printf("inside update_statistics. best_ut[s_id+NP*NT]: %f\n",  best_ut[s_id + NP * NT]);

    }
    
    CEM_CUDA_car::CEM_CUDA_car(enhanced_system_t* model, unsigned int num_of_problems, unsigned int number_of_samples, unsigned int number_of_t,
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

        double width = 8.;

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
        cudaMalloc(&d_obs_corner_list, NOBS * 8 * sizeof(double));
        cudaMalloc(&d_obs_axis_list, NOBS * 4 * sizeof(double));
        cudaMalloc(&d_obs_ori_list, NOBS * 2 * sizeof(double));

        cudaMalloc(&d_active_mask, NP * NS * sizeof(bool));

        
        
        obs_corner_list = new double[NOBS*8]();
        obs_axis_list = new double[NOBS*4]();
        obs_ori_list = new double[NOBS*2]();

		for(unsigned i=0;i<_obs_list.size();i++)
        {
            // each obstacle is represented by its middle point
            // calculate the four points representing the rectangle in the order
            // UL, UR, LR, LL
            // the obstacle points are concatenated for efficient calculation
            double x = _obs_list[i][0];
            double y = _obs_list[i][1];
			// order: (left-bottom, right-bottom, right-upper, left-upper)

            obs_corner_list[i*8 + 0] = x - width / 2;  obs_corner_list[i*8 + 1] = y + width / 2;
            obs_corner_list[i*8 + 2] = x + width / 2;  obs_corner_list[i*8 + 3] = y + width / 2;
            obs_corner_list[i*8 + 4] = x + width / 2;  obs_corner_list[i*8 + 5] = y - width / 2;
            obs_corner_list[i*8 + 6] = x - width / 2;  obs_corner_list[i*8 + 7] = y - width / 2;


			// horizontal axis and vertical
            obs_axis_list[i*4] = obs_corner_list[i*8+2] - obs_corner_list[i*8];
            obs_axis_list[i*4+2] = obs_corner_list[i*8+6] - obs_corner_list[i*8];
            obs_axis_list[i*4+1] = obs_corner_list[i*8+3] - obs_corner_list[i*8+1];
            obs_axis_list[i*4+3] = obs_corner_list[i*8+7] - obs_corner_list[i*8+1];
            std::vector<double> obs_length;
            obs_length.push_back(sqrt(obs_axis_list[i*4]*obs_axis_list[i*4]+obs_axis_list[i*4+1]*obs_axis_list[i*4+1]));
            obs_length.push_back(sqrt(obs_axis_list[i*4+2]*obs_axis_list[i*4+2]+obs_axis_list[i*4+3]*obs_axis_list[i*4+3]));
			// ormalize the axis
            for (unsigned i1=0; i1<2; i1++)
            {
                for (unsigned j1=0; j1<2; j1++)
                {
                    obs_axis_list[i*4+i1*2+j1] = obs_axis_list[i*4+i1*2+j1] / obs_length[i1];
                }
            }
            // obtain the inner product of the left-bottom corner with the axis to obtain the minimal of projection value
            obs_ori_list[i*2] = obs_corner_list[i*8]*obs_axis_list[i*4]+obs_corner_list[i*8+1]*obs_axis_list[i*4+1];
            obs_ori_list[i*2+1] = obs_corner_list[i*8]*obs_axis_list[i*4+2]+obs_corner_list[i*8+1]*obs_axis_list[i*4+3];
        }
		//std::cout << "after initialization" << std::endl;    
        cudaMemcpy(d_obs_corner_list, obs_corner_list, sizeof(double) * NOBS * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_obs_axis_list, obs_axis_list, sizeof(double) * NOBS * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_obs_ori_list, obs_ori_list, sizeof(double) * NOBS * 2, cudaMemcpyHostToDevice);
        // for multiple start
        cudaMalloc(&d_start_state, NP * DIM_STATE * sizeof(double));
        cudaMalloc(&d_goal_state, NP * DIM_STATE * sizeof(double));

        // initiate curand
        cudaMalloc((void**)&devState,NP * NS * NT * sizeof(curandState));
        initCurand << <(NP * NS * NT + 31) / 32, 32 >> > (devState, 42);
        
        //printf("done, execution:\n");

    }
    void CEM_CUDA_car::solve(const double* start, const double* goal, double* best_u, double* best_t){
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
            //std::cout << "iteration: " << it << std::endl; 
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


            //std::cout<< "end of sorting." <<std::endl;
            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            for(unsigned int t_step = 0; t_step < NT; t_step++){
                propagate<<<grid_s, block_p >>>(d_temp_state, d_control, d_time, d_deriv, t_step, NS, NT, d_active_mask, d_obs_corner_list, d_obs_axis_list, d_obs_ori_list);
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
            
            /**
            //**  below method converts GPU to CPU, sorts in CPU, and then converts back
            // copy gpu loss to cpu

            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            cudaMemcpy(loss, d_loss, NP * NS * sizeof(double), cudaMemcpyDeviceToHost);
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. cudaMemcpy loss takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of cudaMemcpy loss takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif
            for (unsigned int p = 0; p < NP; p++) {
                //std::cout<< "sorting... p=" << p <<std::endl;
                // copy loss to std::vector of std::pair. For sorting
                #ifdef PROFILE
                profile_start = std::chrono::high_resolution_clock::now();
                #endif
                for (unsigned int si = 0; si < NS; si++)
                {
                    loss_pair[si].first = loss[p*NS+si];
                    loss_pair[si].second = si;
                }

                sort(loss_pair.begin(), loss_pair.end());
                // copy sorted value from CPU to GPU
                for (unsigned int si = 0; si < NS; si++)
                {
                    loss[p*NS+si] = loss_pair[si].first;
                    loss_ind[p*NS+si] = loss_pair[si].second;
                }
                #ifdef PROFILE
                profile_stop = std::chrono::high_resolution_clock::now();
                profile_duration = profile_stop - profile_start; 
                std::cout << "inside cem_cuda:solve. sort takes " << profile_duration.count() << "s" << std::endl; 
                std::cout << "inside cem_cuda:solve. 1000 steps of sort takes " << 1000*profile_duration.count() << "s" << std::endl; 
                #endif


                // profile_start = std::chrono::high_resolution_clock::now();
                // thrust::sequence(loss_ind_ptr + NS * p, loss_ind_ptr + NS * p + NS);


                // thrust::sort_by_key(loss_ptr + NS * p, loss_ptr + NS * p + NS, loss_ind_ptr + NS * p);
                // profile_stop = std::chrono::high_resolution_clock::now();
                // profile_duration = profile_stop - profile_start; 
                // std::cout << "inside cem_cuda:solve. thrust calls takes " << profile_duration.count() << "s" << std::endl; 
                // std::cout << "inside cem_cuda:solve. 1000 steps of thrust calls takes " << 1000*profile_duration.count() << "s" << std::endl; 
        
            }
            #ifdef PROFILE
            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            // copy sorted value from CPU to GPU
            cudaMemcpy(d_loss_ind, loss_ind, NP*NS*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_loss, loss, NP*NS*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. cudaMemcpy loss to device takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of cudaMemcpy loss to device takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif

            */


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
                d_loss_ind,  d_loss, NP, NS, NT, N_ELITE, d_best_u, d_best_t);
            
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. update_statistics takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of update_statistics takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif

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

    
    unsigned int CEM_CUDA_car::get_control_dimension(){
        return c_dim * NT;
    }

    unsigned int CEM_CUDA_car::get_num_step(){
        return NT;
    }
}