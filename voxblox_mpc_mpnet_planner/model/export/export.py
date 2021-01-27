import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

import torch
import numpy as np

import copy
from dataset.dataset import get_loader
from termcolor import colored

def export(net, setup, exported_path="exported/output/net.pt", batch_size=1):
    print(colored("Trying to export to {}".format(exported_path), "red"))
    env_vox = torch.from_numpy(np.load('dataset/data/voxel.npy')).float()
    _, test_loader = get_loader(batch_size=1, setup=setup, )
    
    env_input = env_vox.repeat(batch_size, 1, 1, 1).transpose(1, 3).cuda()
    state_input = test_loader.dataset[0:batch_size][0].cuda()

    # print(env_input.shape)
    with torch.no_grad():
        # net.eval()
        eval_net = copy.deepcopy(net)
        eval_net.eval()
        output = eval_net(state_input, env_input)
        print(colored("eval_output:", "blue"))
        print(output)
        eval_serilized_module = torch.jit.script(eval_net)
        print(colored("eval_serilized_output:", "blue"))
        print(eval_serilized_module(state_input, env_input))
        print(colored("Tracing and Saving...", "blue"))
        traced_script_module = torch.jit.trace(net, (state_input, env_input))
        serilized_module = torch.jit.script(net)
        serilized_output = serilized_module(state_input, env_input)
   
    traced_script_module.save(exported_path)
    print(colored("Exported to {}".format(exported_path), "red"))
