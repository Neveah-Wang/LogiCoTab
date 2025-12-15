import os
import json
import argparse
import warnings
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import baselines.STaSy.datasets as datasets
from baselines.STaSy.utils import save_checkpoint, restore_checkpoint, apply_activate
import baselines.STaSy.losses as losses
from baselines.STaSy.models import ncsnpp_tabular
from baselines.STaSy.models import utils as mutils
from baselines.STaSy.models.ema import ExponentialMovingAverage
import baselines.STaSy.sde_lib as sde_lib
from baselines.STaSy.configs.config import get_config
import lib
from lib.make_dataset import make_dataset_for_uncondition, FastTensorDataLoader



warnings.filterwarnings("ignore")


def train(raw_config):
    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])
    task_type = raw_config['task_type']
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

    # train_z = torch.tensor(dataset.X_num['train'])
    config.data.image_size = train_z.shape[1]
    print(config.data.image_size)

    # Initialize model.
    config.device = device
    score_model = mutils.create_model(config)
    print(score_model)
    num_params = sum(p.numel() for p in score_model.parameters())
    print("the number of parameters", num_params)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    # optimizer
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

    initial_step = int(state['epoch'])
    batch_size = config.training.batch_size
    shuffle_buffer_size = 10000
    num_epochs = None

    train_data = train_z
    train_iter = DataLoader(train_data,
                            batch_size=config.training.batch_size,
                            shuffle=True,
                            num_workers=4)

    scaler = datasets.get_data_scaler(config) 
    inverse_scaler = datasets.get_data_inverse_scaler(config)

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
        logging.info(score_model)


    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting

    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, workdir=save_dir, spl=config.training.spl,
                                    alpha0=config.model.alpha0, beta0=config.model.beta0)

    best_loss = np.inf
    

    for epoch in range(initial_step, config.training.epoch+1):
        start_time = time.time()
        state['epoch'] += 1

        batch_loss = 0
        batch_num = 0
        for iteration, batch in enumerate(train_iter): 
            batch = batch.to(config.device).float()
            num_sample = batch.shape[0]
            batch_num += num_sample
            loss = train_step_fn(state, batch)
            batch_loss += loss.item() * num_sample
       
        batch_loss = batch_loss / batch_num
        print("epoch: %d, iter: %d, training_loss: %.5e" % (epoch, iteration, batch_loss))

        if batch_loss < best_loss:
            best_loss = batch_loss
            save_checkpoint(os.path.join(save_dir, 'model.pth'), state)

        # if epoch % 1000 == 0:
        #     save_checkpoint(os.path.join(save_dir, f'checkpoint_{epoch}.pth'), state)

        end_time = time.time()
        # print("training time: %.5f" % (end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoDi\config.toml")

    train(raw_config)

"""
python baselines/STaSy/train.py --config exp/adult/STaSy/config.toml
python baselines/STaSy/train.py --config exp/shopper/STaSy/config.toml
python baselines/STaSy/train.py --config exp/buddy/STaSy/config.toml
"""