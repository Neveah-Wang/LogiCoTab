# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')


import os
import json
import warnings
import argparse
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
    if task_type == 'regression':
        categories = raw_config['num_categorical']
    else:
        categories = [raw_config['num_classes']] + raw_config['num_categorical']
    num_class = np.array(categories)
    print("categories = ", categories)

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
    args.nf = args.nf_con
    model_con = tabularUnet(args)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=args.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, args.beta_1, args.beta_T, args.T, args.mean_type, args.var_type).to(device)

    args.input_size = train_dis_data.shape[1] 
    args.cond_size = train_con_data.shape[1]
    args.output_size = train_dis_data.shape[1]
    args.encoder_dim = raw_config['model_params']['encoder_dim_dis']
    args.nf = args.nf_dis
    model_dis = tabularUnet(args)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, args, timesteps=args.T, loss_type='vb_stochastic').to(device)


    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    print('Continuous model params: %d' % (num_params_con))
    print('Discrete model params: %d' % (num_params_dis))


    total_steps_both = args.total_epochs_both * int(train_con_data.shape[0]/args.training_batch_size+1)
    sample_step = args.sample_step * int(train_con_data.shape[0]/args.training_batch_size+1)
    print("Total steps: %d" %total_steps_both)
    print("Sample steps: %d" %sample_step)
    print("Continuous: %d, %d" %(train_con_data.shape[0], train_con_data.shape[1]))
    print("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))


    model_con.load_state_dict(torch.load(f'{save_dir}/model_con.pt'))
    model_dis.load_state_dict(torch.load(f'{save_dir}/model_dis.pt'))

    model_con.eval()
    model_dis.eval()
    
    print("Start sampling")
    start_time = time.time()
    with torch.no_grad():
        x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
        log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
        x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, categories, args)

    x_dis = apply_activate(x_dis, num_class)
    syn_num = x_con
    syn_cat = x_dis

    if raw_config['Transform']['y_policy'] == 'one-hot':
        n_y_feat = raw_config['num_classes']
    else:
        n_y_feat = len(raw_config['y_column'])
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']

    if task_type == 'regression':
        syn_target = syn_num[:, :n_y_feat]
        syn_num = syn_num[:, n_y_feat:]
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :n_y_feat]
        syn_cat = syn_cat[:, n_y_feat:]

    syn_num = dataset.num_transformer.inverse_transform(syn_num.detach().cpu().numpy())
    syn_cat = dataset.cat_transformer.inverse_transform(syn_cat.detach().cpu().numpy()) if dataset.cat_transformer is not None else syn_cat.detach().cpu().numpy()
    syn_target = dataset.y_transformer.inverse_transform(syn_target.detach().cpu().numpy())
    # sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])

    end_time = time.time()
    print('Samping time:', end_time - start_time)

    x_df = pd.DataFrame(np.concatenate((syn_num, syn_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(syn_target, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)

    # 保存生成的数据
    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")
    merged_df.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)
    if raw_config['num_numerical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_num_synthesis", syn_num)
    if raw_config['num_categorical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_cat_synthesis", syn_cat)
    y_real = np.load(os.path.join(real_data_path, f'Y_train.npy'), allow_pickle=True)
    print("type(y_real) = ", y_real.dtype)
    if y_real.dtype == bool:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_target)
    else:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_target.astype(y_real.dtype))
    print(f'The generated data has been saved at {save_dir}/synthesis_null')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()
    raw_config = lib.util.load_config(args.config)

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成/v11\exp/adult\CoDi\config.toml")

    main(raw_config)

"""
python baselines/CoDi/sample.py --config exp/adult/CoDi/config.toml
python baselines/CoDi/sample.py --config exp/shopper/CoDi/config.toml
python baselines/CoDi/sample.py --config exp/buddy/CoDi/config.toml
"""