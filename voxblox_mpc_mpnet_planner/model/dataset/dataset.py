import torch
import torch.utils.data as data_utils
import numpy as np


def np_to_loader(data, label, ind, batch, shuffle):
    th_data = torch.from_numpy(data[ind,:]).float()
    th_label = torch.from_numpy(label[ind, :]).float()
    dataset = data_utils.TensorDataset(th_data, th_label)
    return data_utils.DataLoader(dataset, batch_size=batch, shuffle=shuffle)

def get_loader(batch_size=128, setup='default', train_shuffle=True, test_shuffle=False, training_ratio=0.7):
    path_data = np.load('./dataset/data/{setup}/path_data.npy'.format(setup=setup))
    gt = np.load('./dataset/data/{setup}/gt_data.npy'.format(setup=setup))
    shuffle_ind = np.arange(path_data.shape[0])
    np.random.shuffle(shuffle_ind)
    np.random.seed(42)
    n_train = int(path_data.shape[0] * training_ratio)
    train_ind = shuffle_ind[:n_train]
    test_ind = shuffle_ind[n_train:]
    train_loader = np_to_loader(path_data, gt, train_ind, batch_size, shuffle=train_shuffle)
    test_loader = np_to_loader(path_data, gt, test_ind, batch_size, shuffle=test_shuffle)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_loader()
