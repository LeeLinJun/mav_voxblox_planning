from dataset.dataset import get_loader
import importlib

import torch

import numpy as np
import click

from training_utils.trainer import train_network

@click.command()
@click.option('--ae_output_size', default=64, help='ae_output_size')
@click.option('--state_size', default=13, help='')
@click.option('--lr', default=3e-4, help='learning_rate')
@click.option('--epochs', default=500, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--setup', default='default_norm')
@click.option('--loss_type', default='l1_loss')
@click.option('--lr_step_size', default=200)
@click.option('--aug', default=False)
@click.option('--network_name', default="mpnet")
def main(ae_output_size, state_size, lr, epochs, batch, 
    setup, loss_type, lr_step_size, aug, network_name):
    # mpnet_module = importlib.import_module('.mpnet_{}'.format(system), package=".networks")
    try:
        from networks.mpnet_quadrotor_voxblox import MPNet
        state_dim = 13
        in_channels = 20
    except:
        print("Unrecognized model name")
        raise
    
    print("Training with setup:\t{}".format(setup))
    mpnet = MPNet(ae_input_size=100, ae_output_size=ae_output_size, in_channels=in_channels, state_size=state_dim)

    data_loaders = get_loader(batch_size=batch, setup=setup)

    train_network(network=mpnet, data_loaders=data_loaders, network_name=network_name,
        lr=lr, epochs=epochs, batch=batch, 
        setup=setup,
        using_step_lr=True, step_size=lr_step_size, gamma=0.9,
        loss_type=loss_type, weight_save_epochs=25, aug=aug)

if __name__ == '__main__':
    main()
