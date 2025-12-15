import os
import argparse
import warnings
import json
import time
import torch
import numpy as np
import pandas as pd


import baselines.STaSy.datasets as datasets
from baselines.STaSy.utils import restore_checkpoint
import baselines.STaSy.losses as losses
import baselines.STaSy.sampling as sampling
from baselines.STaSy.models import ncsnpp_tabular
from baselines.STaSy.models import utils as mutils
from baselines.STaSy.models.ema import ExponentialMovingAverage
import baselines.STaSy.sde_lib as sde_lib
from baselines.STaSy.configs.config import get_config
import lib
from lib import save_synthesis_data
from lib.make_dataset import make_dataset_for_uncondition, FastTensorDataLoader, split_num_cat_target



warnings.filterwarnings("ignore")



def sample(raw_config):
    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])
    dataname = raw_config['dataname']
    config = get_config(dataname)

    dataset = make_dataset_for_uncondition(real_data_path, raw_config)
    if dataset.X_cat is not None:
        if dataset.X_num is not None:
            train_z = torch.from_numpy(np.concatenate([dataset.X_num['train'], dataset.X_cat['train']], axis=1)).float()
        else:
            train_z = torch.from_numpy(dataset.X_cat['train']).float()
    else:
        train_z = torch.from_numpy(dataset.X_num['train']).float()

    num_features = train_z.shape[1]
    config.data.image_size = num_features

    print('Input dimension: {}'.format(num_features))
    # Initialize model.
    score_model = mutils.create_model(config)
    print(score_model)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    # optimizer
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)


    # load_satate_dict

    state = restore_checkpoint(f'{save_dir}/model.pth', state, config.device)
    print('Loading SAVED model at from {}/model.pth'.format(save_dir))

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sampling_shape = (dataset.y['train'].shape[0], config.data.image_size)

    inverse_scaler = datasets.get_data_inverse_scaler(config)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    print('Start sampling...')
    start_time = time.time()
    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
    print(f"samples: ", samples)
    print(f"samples.shape: ", samples.shape)

    syn_num, syn_cat, syn_target = split_num_cat_target(samples, raw_config, dataset.num_transformer, dataset.cat_transformer, dataset.y_transformer)
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    x_df = pd.DataFrame(np.concatenate((syn_num, syn_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(syn_target, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)


    end_time = time.time()
    print(f'Sampling time = {end_time - start_time}')
    print('Saving sampled data to {}'.format(save_dir))

    # 保存生成的数据
    save_synthesis_data(raw_config, merged_df, syn_num, syn_cat, syn_target, w='null')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample of STaSy')
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)
    sample(raw_config)

"""
python baselines/STaSy/sample.py --config exp/adult/STaSy/config.toml
python baselines/STaSy/sample.py --config exp/shopper/STaSy/config.toml
python baselines/STaSy/sample.py --config exp/buddy/STaSy/config.toml
"""