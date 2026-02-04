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
from lib.metrics import evaluate_to_file, evaluate_multiple_seeds, write_avg_results_to_file
from collections import Counter
from catboost import CatBoostClassifier


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


    """单次评估"""
    # clf = CatBoostClassifier(random_seed=4)
    # clf.fit(X_train, y_train)
    # evaluate_to_file(clf, X_val, y_val, help_str=f"SMOTE + CatBoost, Dataset:{dataname}", log_file='eval.log')

    """多种随机种子，求平均"""
    avg_results = evaluate_multiple_seeds(
        X_train, y_train,
        X_val, y_val,
        seeds=list(range(10)),
        catboost_params={}
    )
    write_avg_results_to_file(
        avg_results,
        help_str=f"Real + CatBoost, Dataset:{dataname}",
        log_file="eval_average.log"
    )


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
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/mammography\CoTable\config.toml"))


    for raw_config in raw_config_list:
        main(raw_config)
