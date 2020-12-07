#include "trajectory_optimizers/cem_cuda_acrobot.hpp"
#include "systems/enhanced_system.hpp"

//#define PROFILE

#include <chrono>

// #define DEBUG

// define system parameters
#include <cmath>
#define LENGTH 20.0
#define m 1.0

#define lc  .5
#define lc2  .25
#define l2  1
#define I1  0.2
#define I2  1.0
#define l  1.0
#define g  9.8



#define STATE_THETA_1 0
#define STATE_THETA_2 1
#define STATE_V_1 2
#define STATE_V_2 3
#define CONTROL_T 0

#define MIN_V_1 -6
#define MAX_V_1 6
#define MIN_V_2 -6
#define MAX_V_2 6
#define MIN_TORQUE -4
#define MAX_TORQUE 4


#define DT  2e-2

#define PI  3.141592654f

#define DIM_STATE 4
#define DIM_CONTROL 1
#define NOBS 4
#define OBS_PENALTY 1000.0





namespace trajectory_optimizers_acrobot{

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

        d_mean_time[id] = mean_time;
        d_mean_control[id] = mean_control[0];
        d_std_control[id] = std_control[0];
        d_std_time[id] = std_time;
        // printf("inside set_statistics. d_mean_time: %f\n", d_mean_time[id]);
        // printf("inside set_statistics. d_mean_control: %f\n", d_mean_control[id]);
        // printf("inside set_statistics. d_std_control: %f\n", d_std_control[id]);
        // printf("inside set_statistics. d_std_time: %f\n", d_std_time[id]);

        
    }

    __global__
    void set_start_state(double* temp_state, double* start, const int NS){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = np * NS + ns;
    
        temp_state[STATE_THETA_1 + id*DIM_STATE] = start[STATE_THETA_1 + np* DIM_STATE];
        temp_state[STATE_THETA_2 + id*DIM_STATE] = start[STATE_THETA_2 + np * DIM_STATE];
        temp_state[STATE_V_1 + id*DIM_STATE] = start[STATE_V_1 + np * DIM_STATE];
        temp_state[STATE_V_2 + id*DIM_STATE] = start[STATE_V_2 + np * DIM_STATE]; 
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
        //printf("inside sampling. curand_normal control: %f\n", curand_normal(&state[id]) + mean_control[np * NT + nt]);
        //printf("inside sampling. std_control: %f\n", std_control[np * NT + nt]);

        double c = std_control[np * NT + nt] * curand_normal(&state[id]) + mean_control[np * NT + nt];
        //printf("mean control:%f\n", mean_control[np * NT + nt]);

        if (c > MAX_TORQUE) {
            c = MAX_TORQUE;
        }
        else if (c < MIN_TORQUE) {
            c = MIN_TORQUE;
        }
        control[np * NS * NT + ns * NT + nt] = c;

        //printf("inside sampling. curand_normal time: %f\n", std_time[np * NT + nt] * curand_normal(&state[id]) + mean_time[np * NT + nt]);
        //printf("inside sampling. std_time: %f\n", std_time[np * NT + nt]);
        //printf("inside sampling. mean_time: %f\n", mean_time[np * NT + nt]);
        double t = std_time[np * NT + nt] * curand_normal(&state[id]) + mean_time[np * NT + nt];
        //if(t < DT){  // original working one (on cartpole)
        if (t < 0.){
            t = 0;
        } else if (t > MAX_T) {
            t = MAX_T;
        }
        time[np * NS * NT + ns * NT + nt] = t;      
        //printf("c:%f, t:%f\n", c, t);

    }

    __device__
    bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
    // compute whether two lines intersect with each other
    {
        // ref: http://www.jeffreythompson.org/collision-detection/line-rect.php
        // calculate the direction of the lines
        double uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
        double uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    
        // if uA and uB are between 0-1, lines are colliding
        if (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1)
        {
            // intersect
            return true;
        }
        // not intersect
        return false;
    }

    __device__
    bool valid_state(double* temp_state, double* obs_list)
    {
        // check the pole with the rectangle to see if in collision
        // calculate the pole state
        // check if the position is within bound
        double pole_x0 = 0.;
        double pole_y0 = 0.;
        double pole_x1 = (LENGTH) * cos(temp_state[STATE_THETA_1] - M_PI / 2);
        double pole_y1 = (LENGTH) * sin(temp_state[STATE_THETA_1] - M_PI / 2);
        double pole_x2 = pole_x1 + (LENGTH) * cos(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);
        double pole_y2 = pole_y1 + (LENGTH) * sin(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);

        //std::cout << "state:" << temp_state[0] << "\n";
        //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
        //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
        for(unsigned int i = 0; i < NOBS; i++)
        {
            // check if any obstacle has intersection with pole
            //std::cout << "obstacle " << i << "\n";
            //std::cout << "points: \n";
            for (unsigned int j = 0; j < 8; j+=2)
            {

                //std::cout << j << "-th point: " << "(" << obs_list[i][j] << ", " << obs_list[i][j+1] << ")\n";
            }
            for (unsigned int j = 0; j < 8; j+=2)
            {
                // check each line of the obstacle
                double x1 = obs_list[i * 8 + j];
                double y1 = obs_list[i * 8 + j + 1];
                double x2 = obs_list[i * 8 + (j+2) % 8];
                double y2 = obs_list[i * 8 +(j+3) % 8];
                if (lineLine(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2))
                {
                    // intersect
                    return false;
                }
                if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
                {
                    // intersect
                    return false;
                }
            }
        }
        return true;
    }

    __global__
    void propagate(double* temp_state, double* control, double* time, double* deriv, 
        const int t_step, const int NS, const int NT, bool* active_mask, double* obs_list){
            unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int ns = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int id = np * NS + ns;
            //printf("%d, %d, %d, %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
                //printf("%d\n", id);

            double t = time[np * NS * NT + ns * NT + t_step];
            if (t < 0){
                t = 0;
            }
            int num_step = t / DT;
            double _a = control[np * NS * NT + ns * NT + t_step];
                
            for(unsigned int i = 0; i < num_step; i++){
                if(!active_mask[id]){
                    break;
                }
                // update derivs
                double theta2 = temp_state[STATE_THETA_2 + id*DIM_STATE];
                double theta1 = temp_state[STATE_THETA_1 + id*DIM_STATE] - M_PI / 2;
                double theta1dot = temp_state[STATE_V_1 + id*DIM_STATE];
                double theta2dot = temp_state[STATE_V_2 + id*DIM_STATE];
                double _tau = _a;
            
                //extra term m*lc2
                double d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2;
            
                double d22 = m * lc2 + I2;
                double d12 = m * (lc2 + l * lc * cos(theta2)) + I2;
                double d21 = d12;
            
                //extra theta1dot
                double c1 = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2));
                double c2 = m * l * lc * theta1dot * theta1dot * sin(theta2);
                double g1 = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2));
                double g2 = m * lc * g * cos(theta1 + theta2);
            
                deriv[STATE_THETA_1 + id*DIM_STATE] = theta1dot;
                deriv[STATE_THETA_2 + id*DIM_STATE] = theta2dot;
            
                double u2 = _tau - 1 * .1 * theta2dot;
                double u1 = -1 * .1 * theta1dot;
                double theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21);
                double theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21);
            
                deriv[STATE_V_1 + id*DIM_STATE] = theta1dot_dot;
                deriv[STATE_V_2 + id*DIM_STATE] = theta2dot_dot;

                // update states
                temp_state[STATE_THETA_1 + id*DIM_STATE] += DT * deriv[STATE_THETA_1 + id*DIM_STATE];
                temp_state[STATE_THETA_2 + id*DIM_STATE] += DT * deriv[STATE_THETA_2 + id*DIM_STATE];
                temp_state[STATE_V_1 + id*DIM_STATE] += DT * deriv[STATE_V_1 + id*DIM_STATE];
                temp_state[STATE_V_2 + id*DIM_STATE] += DT * deriv[STATE_V_2 + id*DIM_STATE];
                // enforce bounds

                if(temp_state[0 + id*DIM_STATE]<-M_PI)
                        temp_state[0 + id*DIM_STATE]+=2*M_PI;
                else if(temp_state[0 + id*DIM_STATE]>M_PI)
                        temp_state[0 + id*DIM_STATE]-=2*M_PI;
                if(temp_state[1 + id*DIM_STATE]<-M_PI)
                        temp_state[1 + id*DIM_STATE]+=2*M_PI;
                else if(temp_state[1 + id*DIM_STATE]>M_PI)
                        temp_state[1 + id*DIM_STATE]-=2*M_PI;
                if(temp_state[2 + id*DIM_STATE]<MIN_V_1)
                        temp_state[2 + id*DIM_STATE]=MIN_V_1;
                else if(temp_state[2 + id*DIM_STATE]>MAX_V_1)
                        temp_state[2 + id*DIM_STATE]=MAX_V_1;
                if(temp_state[3 + id*DIM_STATE]<MIN_V_2)
                        temp_state[3 + id*DIM_STATE]=MIN_V_2;
                else if(temp_state[3 + id*DIM_STATE]>MAX_V_2)
                        temp_state[3 + id*DIM_STATE]=MAX_V_2;
                // validate_states
                bool valid = valid_state(&temp_state[id*DIM_STATE], obs_list);
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

        double loss1= temp_state[id * DIM_STATE + STATE_THETA_1] - goal_state[np * DIM_STATE + STATE_THETA_1];
        if (loss1 < 0)
        {
            loss1 = -loss1;
        }
        if (loss1 > M_PI)
        {
            loss1 = 2*M_PI - loss1;
        }
        double loss2 = temp_state[id * DIM_STATE + STATE_THETA_2] - goal_state[np * DIM_STATE + STATE_THETA_2];
        if (loss2 < 0)
        {
            loss2 = -loss2;
        }
        if (loss2 > M_PI)
        {
            loss2 = 2*M_PI - loss2;
        }       
        loss[id] = sqrt(loss1*loss1 + loss2*loss2 \
            + 0.5 * (temp_state[id * DIM_STATE + STATE_V_1] - goal_state[np * DIM_STATE + STATE_V_1]) * (temp_state[id * DIM_STATE + STATE_V_1] - goal_state[np * DIM_STATE + STATE_V_1])\
            + 0.5 * (temp_state[id * DIM_STATE + STATE_V_2] - goal_state[np * DIM_STATE + STATE_V_2]) * (temp_state[id * DIM_STATE + STATE_V_2] - goal_state[np * DIM_STATE + STATE_V_2]));
        if (!active_mask[id]) {
            loss[id] += OBS_PENALTY;
        }
        //printf("loss[%d]: %f", id, loss[id]);

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
        int* loss_ind, double* loss, int NP, int NS, int NT, int N_ELITE, double* best_ut){
        unsigned int np = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int nt = blockIdx.z * blockDim.z + threadIdx.z;
        //printf("inside update_statistics. id: (%d, %d)\n", np, nt);

        //unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
        double sum_control = 0., sum_time = 0., ss_control = 0., ss_time = 0.;
        for(int i = 0; i < N_ELITE; i++){
            //printf("inside update_statistics. N_ELITE: %d\n", N_ELITE);
            //printf("inside update_statistics. elite_i: %d\n", i);
            unsigned int id = np * NS * NT + loss_ind[np * NS + i] * NT + nt;
            //printf("inside update_statistics. loss: %f\n", loss[np * NS + i]);
            //printf("inside update_statistics. loss_ind: %d\n", loss_ind[np * NS + i]);
            //printf("inside update_statistics. id: %d\n", id);

            sum_control += control[id];
            ss_control += control[id] * control[id];
            sum_time += time[id];
            ss_time += time[id] * time[id];
        }
        // printf("%f,%f\n",ss_control, ss_time);
        unsigned int s_id = np * NT + nt;
        mean_control[s_id] = sum_control / N_ELITE;
        mean_time[s_id] = sum_time / N_ELITE;
        double std_control_square = ss_control / N_ELITE - mean_control[s_id] * mean_control[s_id];
        if (std_control_square < 1e-5)
        {
            std_control_square = 1e-5;
        }
        std_control[s_id] = sqrt(std_control_square);

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


        best_ut[s_id] = control[np * NS * NT + loss_ind[np * NS] * NT + nt];
        best_ut[s_id + NP * NT] = time[np * NS * NT + loss_ind[np * NS] * NT + nt];
        //printf("inside update_statistics. best_u[s_id]: %f\n",  best_ut[s_id]);
        //printf("inside update_statistics. best_t[s_id]: %f\n",  best_ut[s_id + NP * NT]);

    }
    

    CEM_CUDA_acrobot::CEM_CUDA_acrobot(enhanced_system_t* model, unsigned int num_of_problems, unsigned int number_of_samples, unsigned int number_of_t,
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
        mu_u0 = new double[DIM_CONTROL]();
        std_u0 = new double[DIM_CONTROL]();
        for (unsigned i=0; i < DIM_CONTROL; i++)
        {
            mu_u0[i] = control_means[i];
            std_u0[i] = control_stds[i];
        }
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

        double width = 6;

        // for CUDA here
        //printf("setup...\n");
        // best_ut = (double*) malloc(2 * NP * NT /*time + control*/ * sizeof(double));   // 2 x NP x NT
        cudaMalloc(&d_best_ut, NP * NT * 2 * sizeof(double)); 
        // temp state, derivative, control, time samples
            // temp_state = (double*) malloc(NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_temp_state, NP * NS * DIM_STATE * sizeof(double)); 
        cudaMalloc(&d_deriv, NP * NS * DIM_STATE * sizeof(double));
        cudaMalloc(&d_control, NP * NS * NT * DIM_CONTROL * sizeof(double));
        cudaMalloc(&d_time, NP * NS * NT * sizeof(double));
        // for sampling statistics
        cudaMalloc(&d_mean_time, NP * NT * sizeof(double));
        cudaMalloc(&d_mean_control, NP * NT* sizeof(double));
        cudaMalloc(&d_std_control, NP * NT * sizeof(double));
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
        cudaMalloc(&d_obs_list, NOBS * 8 * sizeof(double));
        cudaMalloc(&d_active_mask, NP * NS * sizeof(bool));

        
        
        obs_list = new double[NOBS*8]();
        for(unsigned i=0; i<_obs_list.size(); i++)
        {
            // each obstacle is represented by its middle point
            // calculate the four points representing the rectangle in the order
            // UL, UR, LR, LL
            // the obstacle points are concatenated for efficient calculation
            double x = _obs_list[i][0];
            double y = _obs_list[i][1];
            //std::cout << x <<","<< y << std::endl;
            obs_list[i*8 + 0] = x - width / 2;  obs_list[i*8 + 1] = y + width / 2;
            obs_list[i*8 + 2] = x + width / 2;  obs_list[i*8 + 3] = y + width / 2;
            obs_list[i*8 + 4] = x + width / 2;  obs_list[i*8 + 5] = y - width / 2;
            obs_list[i*8 + 6] = x - width / 2;  obs_list[i*8 + 7] = y - width / 2;

        }
        cudaMemcpy(d_obs_list, obs_list, sizeof(double) * NOBS * 8, cudaMemcpyHostToDevice);
        // for multiple start
        cudaMalloc(&d_start_state, NP * DIM_STATE * sizeof(double));
        cudaMalloc(&d_goal_state, NP * DIM_STATE * sizeof(double));

        // initiate curand
        cudaMalloc((void**)&devState,NP * NS * NT * sizeof(curandState));
        initCurand << <(NP * NS * NT + 31) / 32, 32 >> > (devState, 42);
        
        //printf("done, execution:\n");

    }
    void CEM_CUDA_acrobot::solve(const double* start, const double* goal, double* best_u, double* best_t){
        // auto begin = std::chrono::system_clock::now();
        // start and goal should be NP * DIM_STATE
        //std::cout << "inside CEM_CUDA::solve" << std::endl;
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

        set_statistics<<<grid, block_pt>>>(d_mean_time, mu_t0, d_mean_control, mu_u0, d_std_control, std_u0, d_std_time, std_t0, NT);

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
            #ifdef PROFILE

            profile_start = std::chrono::high_resolution_clock::now();
            #endif
            for(unsigned int t_step = 0; t_step < NT; t_step++){
                propagate<<<grid_s, block_p >>>(d_temp_state, d_control, d_time, d_deriv, t_step, NS, NT, d_active_mask, d_obs_list);
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

            /*
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
            update_statistics<<<grid, block_pt >>>(d_control, d_time, d_mean_control, d_mean_time, d_std_control, d_std_time,
                d_loss_ind,  d_loss, NP, NS, NT, N_ELITE, d_best_ut);
            
            #ifdef PROFILE
            profile_stop = std::chrono::high_resolution_clock::now();
            profile_duration = profile_stop - profile_start; 
            std::cout << "inside cem_cuda:solve. update_statistics takes " << profile_duration.count() << "s" << std::endl; 
            std::cout << "inside cem_cuda:solve. 1000 steps of update_statistics takes " << 1000*profile_duration.count() << "s" << std::endl; 
            #endif

        }

        //std::cout << "copying from d_best_ut to best_u and best_t...\n" << std::endl;
        #ifdef PROFILE
        profile_start = std::chrono::high_resolution_clock::now();
        #endif
        cudaMemcpy(best_u, d_best_ut, NP * NT * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(best_t, d_best_ut + NP * NT, NP * NT * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef PROFILE
        profile_stop = std::chrono::high_resolution_clock::now();
        profile_duration = profile_stop - profile_start; 
        std::cout << "inside cem_cuda:solve. cudaMemcpy of best_ut takes " << profile_duration.count() << "s" << std::endl; 
        std::cout << "inside cem_cuda:solve. 1000 steps of cudaMemcpy of best_ut takes " << 1000*profile_duration.count() << "s" << std::endl; 
        #endif

        //std::cout << "inside CEM_CUDA::solve end." << std::endl;

    }

    
    unsigned int CEM_CUDA_acrobot::get_control_dimension(){
        return c_dim * NT;
    }

    unsigned int CEM_CUDA_acrobot::get_num_step(){
        return NT;
    }
}