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

def sample(raw_config):
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    plus1 = raw_config['Transform']['y_plus1']

    dataset = make_dataset(raw_config['real_data_path'], raw_config)
    X = concat_features(dataset, X_num_columns, X_cat_columns)

    cat_features = list(range(len(X_num_columns), len(X_num_columns) + len(X_cat_columns)))
    print(cat_features)

    smote_nc = SMOTENC(
        categorical_features=cat_features,
        random_state=0
    )
    X_resampled, y_resampled = smote_nc.fit_resample(X['train'], dataset.y['train'])

    print("\n------ Before ------")
    print("y['train'].shape", dataset.y['train'].shape)
    print("X['train'].shape", X['train'].shape)
    print(sorted(Counter(dataset.y['train'].flatten()).items()))
    print("\n------ After OverSample with SMOTENC------")
    print("X_resampled.shape", X_resampled.shape)
    print("y_resampled.shape", y_resampled.shape)
    print(sorted(Counter(y_resampled).items()))


    X_num = X_resampled[X_num_columns].to_numpy().astype(np.float32)
    X_cat = X_resampled[X_cat_columns].to_numpy()
    y = y_resampled[:, None]

    X_num = dataset.num_transformer.inverse_transform(X_num) if dataset.num_transformer is not None else X_num
    X_cat = dataset.cat_transformer.inverse_transform(X_cat) if dataset.cat_transformer is not None else X_cat
    if plus1:
        y = dataset.y_transformer.inverse_transform(y - 1) if dataset.y_transformer is not None else y - 1
    else:
        y = dataset.y_transformer.inverse_transform(y) if dataset.y_transformer is not None else y
    # breakpoint()

    X_num_synthesis = X_num[dataset.info['train_size']:]
    X_cat_synthesis = X_cat[dataset.info['train_size']:]
    y_synthesis = y[dataset.info['train_size']:]
    save_data(raw_config, X_num_synthesis, X_cat_synthesis, y_synthesis)

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
    clf.fit(X_train, y_train)
    evaluate_to_file(clf, X_val, y_val, help_str=f"SMOTE + CatBoost, Dataset:{dataname}", log_file='eval.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--n_sample', type=int, default=0, help='暂时没有用到！！')

    args = parser.parse_args()

    raw_config = lib.util.load_config(args.config)

    if args.sample:
        sample(raw_config)



"""
python baselines/SMOTE/main.py --config exp/adult/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/churn/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/shopper/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/covertype/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/page/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/obesity/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/buddy/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/bike/SMOTE/config.toml --sample
python baselines/SMOTE/main.py --config exp/productivity/SMOTE/config.toml --sample
"""