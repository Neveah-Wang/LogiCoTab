# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\LogiCoTab-vae')

import os
import lib
import argparse
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from lib.make_dataset import make_dataset, concat_features, make_dataset_for_evaluation
from lib.metrics import evaluate_to_file
from collections import Counter
from catboost import CatBoostClassifier

def save_data(raw_config, X_num, X_cat, y):
    save_dir = raw_config['parent_dir']
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']

    if not os.path.exists(f"{save_dir}/synthesis_null"):
        os.makedirs(f"{save_dir}/synthesis_null")

    if raw_config['num_numerical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_num_synthesis", X_num)
    if raw_config['num_categorical_features'] != 0:
        np.save(f"{save_dir}/synthesis_null/X_cat_synthesis", X_cat)
    np.save(f"{save_dir}/synthesis_null/Y_synthesis", y)

    X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
    x_df = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), columns=X_num_columns + X_cat_columns)
    y_df = pd.DataFrame(y, columns=y_column)
    merged_df = pd.concat([x_df, y_df], axis=1)
    merged_df.to_csv(f"{save_dir}/synthesis_null/synthesis_null.csv", index=False)

    print(f'\nThe generated data has been saved at {save_dir}/synthesis_null')


def main(raw_config):
    T_dict = raw_config['eval']['Transform']
    T_dict['normalization'] = "None"
    dataname = raw_config['dataname']
    dataset, X = make_dataset_for_evaluation(
        raw_config,
        synthetic_data_path=None,
        real_data_path=raw_config['real_data_path'],
        eval_type='real',
        T_dict=T_dict,
        change_val=False,
        sampling_method=None
    )

    X_train = X['train']
    y_train = pd.Series(dataset.y['train'].ravel())
    X_val = X['val']
    y_val = pd.Series(dataset.y['val'].ravel())

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    clf = CatBoostClassifier()
    clf.fit(X_resampled, y_resampled)
    evaluate_to_file(clf, X_val, y_val, help_str=f"SMOTE + CatBoost, Dataset:{dataname}", log_file='eval.log')


if __name__ == '__main__':
    raw_config_list = []
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp\churn\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/shopper\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/Magic\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/winequality\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml"))


    for raw_config in raw_config_list:
        main(raw_config)
