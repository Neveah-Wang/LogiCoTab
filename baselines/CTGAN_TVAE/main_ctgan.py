# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\LogiCoTab-vae')

import shutil

import lib
import os
import numpy as np
import argparse
from ctgan import CTGAN as CTGANSynthesizer
from baselines.CTGAN_TVAE.eval import eval
from pathlib import Path
import torch
import pickle
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

def train_ctgan(
        parent_dir,
        real_data_path,
        raw_config,
        train_params={"batch_size": 512},
        device="cpu"
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    device = torch.device(device)

    X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, raw_config, 'train')
    X = concat_to_pd(X_num_train, X_cat_train, y_train)

    X.columns = [str(_) for _ in X.columns]
    cat_features = list(map(str, range(X_num_train.shape[1], X_num_train.shape[1] + X_cat_train.shape[1]))) if X_cat_train is not None else []

    if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
        cat_features += ["y"]

    # train_params["batch_size"] = min(y_train.shape[0], train_params["batch_size"])

    print(train_params)
    synthesizer = CTGANSynthesizer(**train_params)
    synthesizer.fit(X, cat_features)


    with open(parent_dir / "ctgan.obj", "wb") as f:
        pickle.dump(synthesizer, f)

    return synthesizer


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sample_ctgan(
        raw_config,
        synthesizer,
        parent_dir,
        real_data_path,
        num_samples,
        train_params,
        device="cpu",
        seed=0
):
    real_data_path = Path(real_data_path)
    parent_dir = Path(parent_dir)
    device = torch.device(device)


    X_num_train, X_cat_train, y_train = lib.read_pure_data(real_data_path, raw_config, 'train')
    X = concat_to_pd(X_num_train, X_cat_train, y_train)


    X.columns = [str(_) for _ in X.columns]
    cat_features = list(map(str, range(X_num_train.shape[1], X_num_train.shape[1] + X_cat_train.shape[1]))) if X_cat_train is not None else []

    if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
        cat_features += ["y"]

    with open(parent_dir / "ctgan.obj", 'rb') as f:
        synthesizer = pickle.load(f)

    gen_data = synthesizer.sample(num_samples)

    y = gen_data['y'].values
    if len(np.unique(y)) == 1:
        y[0] = 0
        y[1] = 1

    # y = y.astype(float)
    # if lib.load_json(real_data_path / "info.json")["task_type"] != "regression":
    #     y = y.astype(int)

    X_cat = gen_data[cat_features].drop('y', axis=1, errors="ignore").values if len(cat_features) else None
    X_num = gen_data.values[:, :X_num_train.shape[1]] if X_num_train is not None else None



    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    x_df = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(y, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)


    # ------------ 开始保存数据 --------------
    save_dir = raw_config['parent_dir']
    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")

    # 仅保存少数类数据
    TARGET = y_column[0]
    value_counts = merged_df[TARGET].value_counts()
    majority_class = value_counts.idxmax()
    minority_class = value_counts.idxmin()
    df_minority = merged_df[merged_df[TARGET] == minority_class]
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

    # 保存全部的数据
    # lib.save_synthesis_data(raw_config, merged_df, X_num, X_cat, y, w='null')


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass



def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)

    args = parser.parse_args()

    # class Args():
    #     def __init__(self):
    #         self.config = "D:\Study\自学\表格数据生成/v11\exp/adult\CTGAN\config.toml"
    #         self.train = True
    #         self.sample = True
    # args = Args()

    raw_config = lib.load_config(args.config)
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)
    ctgan = None
    print(f"train = {args.train}")
    if args.train:
        ctgan = train_ctgan(
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            train_params=raw_config['train_params'],
            device=raw_config['device'],
            raw_config=raw_config
        )
    print(f"sample = {args.sample}")
    if args.sample:
        sample_ctgan(
            raw_config,
            synthesizer=ctgan,
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            num_samples=int(raw_config['sample']['num_samples'] * (raw_config['ir'] - 1)),
            train_params=raw_config['train_params'],
            seed=raw_config['sample']['seed'],
            device=raw_config['device']
        )
    print(f"eval = {args.eval}")
    if args.eval:
        eval(raw_config)

"""
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/adult/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/magic/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/churn/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/shopper/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/obesity/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/bean/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/page/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/buddy/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/winequality/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/yeast_me2/CTGAN/config.toml --train --sample --eval
python baselines/CTGAN_TVAE/main_ctgan.py --config exp/mammography/CTGAN/config.toml --train --sample --eval
"""