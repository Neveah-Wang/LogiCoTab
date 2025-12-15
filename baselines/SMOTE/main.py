# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\v11')

import os
import lib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Any
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist


class MySMOTE(SMOTE):
    def __init__(
            self,
            lam1=0.0,
            lam2=1.0,
            *,
            sampling_strategy="auto",
            random_state=None,
            k_neighbors=5,
            n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

        self.lam1 = lam1
        self.lam2 = lam2

    def _make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


class MySMOTENC(SMOTENC):
    def __init__(
            self,
            lam1=0.0,
            lam2=1.0,
            *,
            categorical_features,
            sampling_strategy="auto",
            random_state=None,
            k_neighbors=3,
            n_jobs=None
    ):
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

        self.lam1 = 0.0
        self.lam2 = 1.0

    def _make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, lam1=0.0, lam2=1.0):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


def save_data(raw_config, X, y, path, n_cat_features=0):
    if n_cat_features > 0:
        X_num = X[:, :-n_cat_features]
        X_cat = X[:, -n_cat_features:]
    else:
        X_num = X
        X_cat = None

    if not os.path.exists(f"{path}/synthesis_null"):
        os.makedirs(f"{path}/synthesis_null")
    np.save(f"{path}/synthesis_null/X_num_synthesis", X_num.astype(float), allow_pickle=True)
    np.save(f"{path}/synthesis_null/Y_synthesis", y, allow_pickle=True)
    if X_cat is not None:
        np.save(f"{path}/synthesis_null/X_cat_synthesis", X_cat, allow_pickle=True)

    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
    x_df = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(y, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)
    merged_df.to_csv(f"{path}/synthesis_null/synthesis_null.csv", index=False)

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth


def sample_smote(raw_config):

    real_data_path = Path(raw_config['real_data_path'])
    parent_dir = raw_config['parent_dir']
    frac_lam_del = raw_config['smote_params']['frac_lam_del']
    info = lib.load_json(real_data_path / 'info.json')
    is_regression = info['task_type'] == 'regression'

    X_num = {}
    X_cat = {}
    y = {}


    X_num['train'], X_cat['train'], y['train'] = lib.read_pure_data(real_data_path, raw_config, 'train')
    X_num['val'], X_cat['val'], y['val'] = lib.read_pure_data(real_data_path, raw_config, 'val')
    X_num['test'], X_cat['test'], y['test'] = lib.read_pure_data(real_data_path, raw_config, 'test')
    X = {k: X_num[k] for k in X_num.keys()}

    if is_regression:
        X['train'] = np.concatenate([X["train"], y["train"].reshape(-1, 1)], axis=1, dtype=object)
        y['train'] = np.where(y["train"] > np.median(y["train"]), 1, 0)

    n_num_features = X['train'].shape[1]
    n_cat_features = X_cat['train'].shape[1] if X_cat['train'] is not None else 0
    cat_features = list(range(n_num_features, n_num_features + n_cat_features))
    print(cat_features)

    scaler = MinMaxScaler().fit(X["train"])
    X["train"] = scaler.transform(X["train"]).astype(object)

    if X_cat['train'] is not None:
        for k in X_num.keys():
            X[k] = np.concatenate([X[k], X_cat[k]], axis=1, dtype=object)


    print("Before:", X['train'].shape)

    lam1 = 0.0 + frac_lam_del / 2
    lam2 = 1.0 - frac_lam_del / 2

    strat = {k: int((1 + raw_config['smote_params']['frac_samples']) * np.sum(y['train'] == k)) for k in np.unique(y['train'])}
    print(strat)
    if n_cat_features > 0:
        sm = MySMOTENC(
            lam1=lam1,
            lam2=lam2,
            random_state=0,
            k_neighbors=raw_config['smote_params']['k_neighbours'],
            categorical_features=cat_features,
            sampling_strategy=strat
        )
    else:
        sm = MySMOTE(
            lam1=lam1,
            lam2=lam2,
            random_state=0,
            k_neighbors=raw_config['smote_params']['k_neighbours'],
            sampling_strategy=strat
        )

    X_res, y_res = sm.fit_resample(X['train'], y['train'])
    if is_regression:
        X_res[:, :X_num["train"].shape[1] + 1] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1] + 1])
        y_res = X_res[:, X_num["train"].shape[1]]
        X_res = np.delete(X_res, [X_num["train"].shape[1]], axis=1)
    else:
        X_res[:, :X_num["train"].shape[1]] = scaler.inverse_transform(X_res[:, :X_num["train"].shape[1]])
        # y_res = y_res.astype(int)


    X_res = X_res[X['train'].shape[0]:]
    y_res = y_res[X['train'].shape[0]:]

    disc_cols = []
    for col in range(X_num["train"].shape[1]):
        uniq_vals = np.unique(X_num["train"][:, col])
        if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
            disc_cols.append(col)
    if len(disc_cols):
        X_res[:, :X_num["train"].shape[1]] = round_columns(X_num["train"], X_res[:, :X_num["train"].shape[1]], disc_cols)


    save_data(raw_config, X_res, y_res, parent_dir, n_cat_features)

    X['train'] = X_res
    y['train'] = y_res

    return X, y




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)

    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)

    if args.sample:
        sample_smote(raw_config)



"""
python baselines/SMOTE/main.py --config exp/adult/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/shopper/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/covertype/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/page/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/obesity/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/buddy/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/bike/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/productivity/SMOTE/config.toml --sample
"""