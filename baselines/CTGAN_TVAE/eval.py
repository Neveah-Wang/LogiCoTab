import pandas as pd
from catboost import CatBoostClassifier

from lib.metrics import evaluate_to_file
from lib.make_dataset import read_pure_data, make_dataset_for_evaluation



def eval(raw_config):
    T_dict = raw_config['eval']['Transform']
    T_dict['normalization'] = "None"
    T_dict['cat_encode_policy'] = "Ordinal"
    T_dict['y_policy'] = "Ordinal"

    dataname = raw_config['dataname']
    dataset, X = make_dataset_for_evaluation(
        raw_config,
        synthetic_data_path=f"{raw_config['parent_dir']}/synthesis_null",
        real_data_path=raw_config['real_data_path'],
        eval_type='merged',
        T_dict=T_dict,
        change_val=False,
        sampling_method=None
    )

    X_train = X['train']
    y_train = pd.Series(dataset.y['train'].ravel())
    X_val = X['val']
    y_val = pd.Series(dataset.y['val'].ravel())

    for i in range(3):
        clf = CatBoostClassifier(random_seed=i,)
        clf.fit(X_train, y_train)
        evaluate_to_file(clf, X_val, y_val, help_str=f"CTGAN + CatBoost, Dataset:{dataname}, seed={i}", log_file='baselines/CTGAN_TVAE/eval.log')