import torch
import numpy as np
import click

from export import export
from dataset.dataset import get_loader
from pathlib import Path
from networks.mpnet_quadrotor_voxblox import MPNet

@click.command()
@click.option('--setup', default='default_norm_aug')
@click.option('--ep', default=500)
@click.option('--outputfn', default="mpnet.pt")
@click.option('--network_name', default='mpnet')
@click.option('--batch_size', default=1)
def main(setup, ep, outputfn, network_name, batch_size):
    mpnet = MPNet(
        ae_input_size=100, 
        ae_output_size=64, 
        in_channels=20, 
        state_size=13).cuda()
    mpnet.load_state_dict(torch.load('output/{}/{}/ep{}.pth'.format(setup, network_name, ep)))
    mpnet.train()
    # mpnet.eval()
    Path("export/output").mkdir(exist_ok=True)

    export(mpnet, setup=setup, exported_path="export/output/{}".format(outputfn), batch_size=batch_size)

if __name__ == '__main__':
    main()
