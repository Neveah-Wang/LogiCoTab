# -*- coding: gbk -*-
# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT')

import lib
from lib.make_dataset import make_dataset_for_evaluation
from lib.metrics import evaluate_metrics, write_avg_results_to_file

import argparse
import pandas as pd
import numpy as np
from imbens.ensemble import AsymBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict


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

    all_results = defaultdict(list)

    for seed in list(range(10)):
        clf = AsymBoostClassifier(
            random_state=seed,
            n_estimators=50,
            estimator=DecisionTreeClassifier(max_depth=100),
            # estimator=CatBoostClassifier(random_seed=seed, verbose=False,),
        )
        clf.fit(X_train, y_train)

        # evaluate(clf, X_val, y_val, help=f"Self-Paced Ensemble, Dataset:{dataname}")
        # evaluate_to_file(clf, X_val, y_val, help_str=f"Self-Paced Ensemble, Dataset:{dataname}", log_file='eval.log')
        metrics = evaluate_metrics(clf, X_val, y_val)
        for k, v in metrics.items():
            all_results[k].append(v)

    # 求均值
    avg_results = {k: np.mean(v) for k, v in all_results.items()}

    write_avg_results_to_file(
        avg_results,
        help_str=f"AsymBoost + DecisionTree, Dataset:{dataname}",
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