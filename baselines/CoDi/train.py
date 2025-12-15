# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')


import numpy as np
import json
import argparse
import warnings
import time

import lib
from lib.make_dataset import make_dataset_for_uncondition, FastTensorDataLoader
from baselines.CoDi.diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
from baselines.CoDi.models.tabular_unet import tabularUnet
from baselines.CoDi.diffusion_discrete import MultinomialDiffusion
from baselines.CoDi.utils import *

warnings.filterwarnings("ignore")


def main(raw_config):
    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']
    device = torch.device(raw_config['device'])
    task_type = raw_config['task_type']

    dataset = make_dataset_for_uncondition(real_data_path, raw_config)
    train_con_data = dataset.X_num['train']
    train_dis_data = dataset.X_cat['train']
    # categories = dataset.get_category_sizes('train')
    # num_class = np.array(categories)
    if task_type == 'regression':
        categories = raw_config['num_categorical']
    else:
        categories = [raw_config['num_classes']] + raw_config['num_categorical']
    num_class = np.array(categories)
    print("num_class = ", num_class)

    class Args():
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, Args(value))
                else:
                    setattr(self, key, value)
    args = Args(raw_config['model_params'])


    # Condtinuous Diffusion Model Setup
    args.input_size = train_con_data.shape[1]
    args.cond_size = train_dis_data.shape[1]
    args.output_size = train_con_data.shape[1]
    args.encoder_dim = raw_config['model_params']['encoder_dim_con']
    args.nf = raw_config['model_params']['nf_con']
    model_con = tabularUnet(args)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=args.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, args.beta_1, args.beta_T, args.T, args.mean_type, args.var_type).to(device)


    args.input_size = train_dis_data.shape[1]
    args.cond_size = train_con_data.shape[1]
    args.output_size = train_dis_data.shape[1]
    args.encoder_dim = raw_config['model_params']['encoder_dim_dis']
    args.nf = raw_config['model_params']['nf_dis']
    model_dis = tabularUnet(args)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, args, timesteps=args.T, loss_type='vb_stochastic').to(device)

    print('Continuous model:')
    print(model_con)

    print('Discrete model:')
    print(model_dis)

    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    print('Continuous model params: %d' % (num_params_con))
    print('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10

    total_steps_both = args.total_epochs_both * int(train_con_data.shape[0] / args.training_batch_size + 1)
    sample_step = args.sample_step * int(train_con_data.shape[0] / args.training_batch_size + 1)
    print("Total steps: %d" % total_steps_both)
    print("Sample steps: %d" % sample_step)
    print("Continuous: %d, %d" % (train_con_data.shape[0], train_con_data.shape[1]))
    print("Discrete: %d, %d" % (train_dis_data.shape[0], train_dis_data.shape[1]))

    epoch = 0
    # train_iter_con = DataLoader(train_con_data, batch_size=args.training_batch_size)
    # train_iter_dis = DataLoader(train_dis_data, batch_size=args.training_batch_size)
    train_iter_con = FastTensorDataLoader(train_con_data, batch_size=args.training_batch_size)
    train_iter_dis = FastTensorDataLoader(train_dis_data, batch_size=args.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    best_loss = float('inf')
    start_time = time.time()
    for step in range(total_steps_both):

        # start_time = time.time()
        model_con.train()
        model_dis.train()

        x_0_con, = next(datalooper_train_con)
        x_0_dis, = next(datalooper_train_dis)
        x_0_con = torch.tensor(x_0_con).to(device).float()
        x_0_dis = torch.tensor(x_0_dis).to(device)

        ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
        con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con,
                                                                     ns_dis, categories, args)

        loss_con = con_loss + args.lambda_con * con_loss_ns
        loss_dis = dis_loss + args.lambda_dis * dis_loss_ns

        optim_con.zero_grad()
        loss_con.backward()
        torch.nn.utils.clip_grad_norm_(model_con.parameters(), args.grad_clip)
        optim_con.step()
        sched_con.step()

        optim_dis.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), args.grad_clip)  # , self.args.clip_value)
        torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), args.grad_clip)  # , self.args.clip_norm)
        optim_dis.step()
        sched_dis.step()

        total_loss = loss_con.item() + loss_dis.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model_con.state_dict(), f'{save_dir}/model_con.pt')
            torch.save(model_dis.state_dict(), f'{save_dir}/model_dis.pt')

        if (step + 1) % int(train_con_data.shape[0] / args.training_batch_size + 1) == 0:

            print(f"Epoch:{epoch}, step = {step}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}, "
                  f"CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}, "
                  f"Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}, "
                  f"lr_con: {optim_con.param_groups[0]['lr']}, lr_dis: {optim_dis.param_groups[0]['lr']}")
            epoch += 1

            # if epoch % 1000 == 0:
            #     torch.save(model_con.state_dict(), f'{save_dir}/model_con_{epoch}.pt')
            #     torch.save(model_dis.state_dict(), f'{save_dir}/model_dis_{epoch}.pt')

        # end_time = time.time()
        # print(f"Time taken: {end_time - start_time:.3f}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoDi\config.toml")

    main(raw_config)

"""
python baselines/CoDi/train.py --config exp/adult/CoDi/config.toml
python baselines/CoDi/train.py --config exp/shopper/CoDi/config.toml
python baselines/CoDi/train.py --config exp/buddy/CoDi/config.toml
"""