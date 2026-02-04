# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v9')

import os
import torch
import math
import pandas as pd
import numpy as np
import argparse
import warnings
import time
import lib
from baselines.TabSyn.models.model import MLPDiffusion, Model
from baselines.TabSyn.models.latent_util import get_input_generate, split_num_cat_target
from baselines.TabSyn.models.diffusion_utils import sample, sample_batched

warnings.filterwarnings('ignore')


def main(raw_config):
    device = torch.device(raw_config['sample']['device'])
    save_dir = raw_config['parent_dir']
    real_data_path = raw_config['real_data_path']

    # 本步骤将 vae 的 decoder_model 进行 load，并将 decoder_model 保存在了 info 中
    train_z, info, num_inverse, cat_inverse, y_inverse = get_input_generate(raw_config)
    in_dim = train_z.shape[1]
    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f'{save_dir}/model.pth'))

    ''' Generating samples '''
    start_time = time.time()

    # num_samples = train_z.shape[0]   # 生成和训练集相同数量的样本
    num_samples = int(train_z.shape[0] * (raw_config['ir'] - 1))
    sample_dim = in_dim

    batch_size = 1024
    # 初始化列表用于收集各批次结果
    all_syn_num = []
    all_syn_cat = []
    all_syn_target = []
    # 分批生成
    num_batches = math.ceil(num_samples / batch_size)
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        if current_batch_size <= 0:
            break
        x_next = sample(model.denoise_fn_D, current_batch_size, sample_dim)
        x_next = x_next * 2 + mean.to(device)
        syn_data = x_next.float()

        # 该步骤将Diffusion生成的数据(latent space) 解码到 真实空间，并划分 num和cat
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, raw_config, info, num_inverse, cat_inverse, y_inverse)

        # 累积结果（确保是 NumPy）
        all_syn_num.append(syn_num)
        all_syn_cat.append(syn_cat)
        all_syn_target.append(syn_target)

    # 拼接所有批次（沿第0维）
    syn_num = np.concatenate(all_syn_num, axis=0) if all_syn_num else np.empty((0, 0))
    syn_cat = np.concatenate(all_syn_cat, axis=0) if all_syn_cat else np.empty((0, 0))
    syn_target = np.concatenate(all_syn_target, axis=0) if all_syn_target else np.empty((0,))

    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    x_df = pd.DataFrame(np.concatenate((syn_num, syn_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(syn_target, columns=y_column)
    syn_df = pd.concat([x_df, y_df], axis=1)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    # ------------ 开始保存数据 --------------
    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")

    # 仅保存少数类数据
    TARGET = y_column[0]
    value_counts = syn_df[TARGET].value_counts()
    majority_class = value_counts.idxmax()
    minority_class = value_counts.idxmin()
    df_minority = syn_df[syn_df[TARGET] == minority_class]
    df_minority.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)

    if raw_config['num_numerical_features'] != 0:
        syn_num = df_minority[raw_config['X_num_columns']].values
        np.save(f"{save_dir}/synthesis_null/X_num_synthesis", syn_num)

    if raw_config['num_categorical_features'] != 0:
        syn_cat = df_minority[raw_config['X_cat_columns']].values
        np.save(f"{save_dir}/synthesis_null/X_cat_synthesis", syn_cat)

    y_real = np.load(os.path.join(real_data_path, f'Y_train.npy'), allow_pickle=True)
    print("type(y_real) = ", y_real.dtype)
    syn_y = df_minority[raw_config['y_column']].values
    if y_real.dtype == bool:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_y)
    else:
        np.save(f"{save_dir}/synthesis_null/Y_synthesis", syn_y.astype(y_real.dtype))

    # 保存全部生成的数据
    '''
    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")
    syn_df.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)
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
    '''
    print(f'The generated data has been saved at {save_dir}/synthesis_null')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample of TabSyn')
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)
    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)
    main(raw_config)

"""
python baselines/TabSyn/sample.py --config exp/adult/TabSyn/config.toml
python baselines/TabSyn/sample.py --config exp/shopper/TabSyn/config.toml
python baselines/TabSyn/sample.py --config exp/buddy/TabSyn/config.toml
"""