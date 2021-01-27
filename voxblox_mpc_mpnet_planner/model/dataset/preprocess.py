import numpy as np
import csv
import pathlib
from tqdm import tqdm

def path_to_tensor_forward(path_dict,
                           normalize,
                           goal_aug=False):
    """
    [env_id, state, goal]
    """ 

    path = path_dict['path']
    costs = path_dict['cost']
    costs2go = costs.copy()
    costs_sofar = costs.copy()
    for i in range(costs.shape[0]):
        costs2go[i] = costs[i:].sum()
        costs_sofar[i] = costs[:i].sum()
    # start_goal = path_dict['start_goal']
    n_nodes = path.shape[0]
    state_size = path.shape[1]
    data = []
    gt = []
    c2g = []
    csf = []
    c = []
    transit_pair_data = []

    for i in range(len(path)):
        if path[i, 6] < 0:
            path[i, 3:7] *= -1

    # start to first path node
#    data.append(np.concatenate(([env_id], start_goal[0], start_goal[-1])))
#    gt.append(path[0, :])
    for i_start in range(n_nodes-1):
        data.append(np.concatenate((path[i_start, :], path[-1, :])))
        #data.append(np.concatenate(([env_id], path[i_start, :], start_goal[-1])))
        transit_pair_data.append(np.concatenate((path[i_start, :], path[i_start+1, :])))
        gt.append(path[i_start+1, :])
        c2g.append(costs2go[i_start])
        csf.append(costs_sofar[i_start])
        c.append(costs[i_start])
        if goal_aug:
            ## goal aug
            for i_goal in range(i_start+1, n_nodes):#[n_nodes-1]:#
                data.append(np.concatenate((path[i_start, :], path[i_goal, :])))
                gt.append(path[i_start+1, :])
    ## last path node to goal
    # data.append(np.concatenate(([env_id], path[-1, :], start_goal[-1])))
    # gt.append(start_goal[-1])

    # data.append(np.concatenate(([env_id], path[-1, :], path[-1, :])))
    # gt.append(path[-1, :])
    #c2g.append(0)

    '''
    subsample_path 
    '''
    data = np.array(data)
    gt = np.array(gt)
    c2g = np.array(c2g)
    csf = np.array(csf)
    c = np.array(c)
    transit_pair_data = np.array(transit_pair_data)
            
    
    if normalize:
        gt_min = np.array([-20, -5, 0,
                            -1, -1, -1, -1,
                            -1, -1, -1,
                            -1, -1, -1]).astype(float)

        gt_max = np.array([5, 25, 2,
                            1, 1, 1, 1,
                            1, 1, 1,
                            1, 1, 1]).astype(float)
        data_min = np.concatenate((gt_min.copy(), gt_min.copy()), axis=0)
        data_max = np.concatenate((gt_max.copy(), gt_max.copy()), axis=0)
        data = -1 + 2 * (data - data_min) / (data_max - data_min)
        gt = -1 + 2 * (gt - gt_min) / (gt_max - gt_min) 
        
        # print((gt_max - gt_min) )
    return data, gt, c2g, csf, c, transit_pair_data

def path_to_dict(path_list, state_dim=13, control_dim=4, duration_dim=1):
    path_list = np.array(path_list)
    return {
        'path': path_list[:, :state_dim],
        'cost': path_list[:, (state_dim + control_dim):(state_dim + control_dim+duration_dim)],
        # 'start_goal' [],
    }

def preprocess(data_prefix="/home/arclabdl1/Linjun/catkin_ws/data", setup="default_norm", save_path=str(pathlib.Path(__file__).parent.absolute())+'/data'):
    hash_file = np.load(data_prefix+'/hash.npy', allow_pickle=True, encoding='bytes').tolist()
    
    path_data = []
    gt_data = []
    
    for start_goal_hash in tqdm(list(hash_file)[:int(0.9*len(hash_file))]):
        # print(start_goal_hash)
        traj = np.loadtxt(open("{}/traj_{}.txt".format(data_prefix, str(start_goal_hash)[1:-1].replace(", ", "_")), "rb"), delimiter=" ", skiprows=0)
        data, gt, c2g, csf, c, transit_pair_data = path_to_tensor_forward(path_to_dict(traj), normalize=True, goal_aug=False)
        path_data += data.tolist()
        gt_data += gt.tolist()

    path_data = np.array(path_data)
    gt_data = np.array(gt_data)
    print(path_data.shape, gt_data.shape)

    np.set_printoptions(suppress=True)
    print(path_data.min(axis=0), path_data.max(axis=0))
    print(gt_data.min(axis=0), gt_data.max(axis=0))

    np.save(save_path+'/{}/path_data.npy'.format(setup), path_data)
    np.save(save_path+'/{}/gt_data.npy'.format(setup), gt_data)
    

    # print(path_data.min(axis=0), path_data.max(axis=0))

def main():
    preprocess()

if __name__ == '__main__':
    main()