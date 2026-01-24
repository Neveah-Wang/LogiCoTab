# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\LogiCoTab-vae')

import lib
from lib.make_dataset import make_dataset_for_evaluation

import argparse
import pandas as pd
import numpy as np
from imbens.ensemble import SelfPacedEnsembleClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from catboost import CatBoostClassifier
from contextlib import redirect_stdout

def evaluate(classifier, X, y, help:str):
    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)[:, 1]

    f1 = f1_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_score)
    mcc = matthews_corrcoef(y, y_pred)
    report = classification_report(y, y_pred, digits=4)

    # 计算G-mean
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    G_mean = np.sqrt(sensitivity * specificity)

    print("")
    print('-' * 20, help, '-' * 20)
    print(f"F1 (macro): {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"mcc: {mcc:.4f}")
    print(f"G_mean: {G_mean:.4f}")
    print("Report:")
    print(report)

    return auc, f1

def evaluate_to_file(classifier, X, y, help_str, log_file=None):
    if log_file:
        # 以追加模式打开文件
        with open(log_file, 'a', encoding='utf-8') as f:
            with redirect_stdout(f):
                return evaluate(classifier, X, y, help_str)
    else:
        return evaluate(classifier, X, y, help_str)

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

    clf = SelfPacedEnsembleClassifier(
        random_state=0,
        n_estimators=5,
        estimator=CatBoostClassifier(),
    )
    clf.fit(X_train, y_train, train_verbose=True, )
    # evaluate(clf, X_val, y_val, help=f"Self-Paced Ensemble, Dataset:{dataname}")
    evaluate_to_file(clf, X_val, y_val, help_str=f"Self-Paced Ensemble, Dataset:{dataname}", log_file='eval.log')


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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', metavar='FILE')
    # parser.add_argument('--sample', action='store_true', default=False)
    # parser.add_argument('--n_sample', type=int, default=0, help='暂时没有用到！！')
    #
    # args = parser.parse_args()
    #
    # raw_config = lib.util.load_config(args.config)

    for raw_config in raw_config_list:
        main(raw_config)