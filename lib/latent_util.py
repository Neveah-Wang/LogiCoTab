import os
import torch
import numpy as np

from lib.make_dataset import make_dataset

def get_latent_train(raw_config):
    parent_dir = raw_config['parent_dir']
    train_z = torch.tensor(np.load(os.path.join(parent_dir, 'latent_data/train_z.npy'))).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    print("train_z.shape = {}".format(train_z.shape))

    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y = torch.from_numpy(dataset.y['train'])

    return train_z, y