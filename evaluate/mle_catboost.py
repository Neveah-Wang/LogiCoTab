import zero
from pathlib import Path
import subprocess
import tempfile
import os
from pathlib import Path
from copy import deepcopy
from pprint import pprint
import shutil
from catboost import CatBoostClassifier, CatBoostRegressor

import lib
from lib import util, metrics
from lib.make_dataset import make_dataset_for_evaluation

pipeline = {
    'CoTable': 'main_sample.py',
    'TabDDPM': 'baselines/TabDDPM/main.py',
    'TabSyn': 'baselines/TabSyn/main.py',
    'CoDi': 'baselines/CoDi/main.py',
    'STaSy': 'baselines/STaSy/main.py',
    'GReaT': 'baselines/GReaT/main.py',
    'SMOTE': 'baselines/SMOTE/main.py',
    'TVAE': 'baselines/CTGAN_TVAE/main_tvae.py',
    'CTGAN': 'baselines/CTGAN_TVAE/main_ctgan.py',
}

def get_catboost_config(ds_name, is_cv=False):
    # ds_name = Path(real_data_path).name
    C = util.load_json(f'evaluate/turned_catboost/{ds_name}_cv.json')
    return C


def train_catboost(
        raw_config,
        parent_dir,
        real_data_path,
        eval_type,
        T_dict,
        guide_w,
        seed=0,
        params=None,
        change_val=False,
        sampling_method='CoTable',
        device=None  # dummy
):
    # zero.improve_reproducibility(seed)
    synthetic_data_path = os.path.join(parent_dir, f'synthesis_{guide_w}')
    dataset, X = make_dataset_for_evaluation(raw_config, synthetic_data_path, real_data_path, eval_type, T_dict, change_val, sampling_method)
    # print(X)
    print("dataset.is_multiclass(): ", dataset.is_multiclass())

    if params is None:
        catboost_config = get_catboost_config(raw_config['dataname'], is_cv=True)
    else:
        catboost_config = params

    if raw_config.get('X_num_columns_real', False):
        X_num_columns = raw_config['X_num_columns_real']
        X_cat_columns = raw_config['X_cat_columns_real']
        y_column = raw_config['y_column_real']
    else:
        X_num_columns = raw_config['X_num_columns']
        X_cat_columns = raw_config['X_cat_columns']
        y_column = raw_config['y_column']

    for split in X.keys():
        X[split][X_cat_columns] = X[split][X_cat_columns].astype(str)
        X[split][X_num_columns] = X[split][X_num_columns].astype(float)

    # print("X[split]: ", X[split])

    print(T_dict)
    pprint(catboost_config, width=100)
    print('-' * 100)


    # 1. 初始化模型
    if dataset.is_regression():
        model = CatBoostRegressor(**catboost_config, eval_metric='RMSE', random_seed=seed)
        predict = model.predict

    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if dataset.is_multiclass() else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(dataset.n_classes)] if dataset.is_multiclass() else ["0", "1"]
        )
        predict = (model.predict_proba if dataset.is_multiclass() else lambda x: model.predict_proba(x)[:, 1])

    # 2. 训练模型
    model.fit(X['train'], dataset.y['train'].ravel(), eval_set=(X['val'], dataset.y['val'].ravel()), verbose=100)

    # 3. 预测
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    # 4. 评估
    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = dataset.calculate_metrics(predictions, None if dataset.is_regression() else 'probs')

    # 打印指标
    metrics_report = metrics.MetricsReport(report['metrics'], dataset.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        util.dump_json(report, os.path.join(parent_dir, "results_catboost.json"))

    return metrics_report



def eval_seeds_catboost(
        raw_config,
        n_seeds,
        eval_type,
        guide_w,
        sampling_method="ddpm",
        model_type="catboost",
        n_datasets=1,
        dump=True,
        change_val=False
):
    metrics_seeds_report = metrics.SeedsMetricsReport()
    parent_dir = Path(raw_config["parent_dir"])

    if eval_type == 'real':
        n_datasets = 1

    T_dict = deepcopy(raw_config['eval']['Transform'])
    T_dict["normalization"] = "None"  # 对于catboost，不需要对数据进行预处理
    T_dict["cat_encode_policy"] = "None"

    temp_config = deepcopy(raw_config)

    # %%  通过with语句创建临时文件，with会自动关闭临时文件
    with tempfile.TemporaryDirectory(dir='D:\Study\自学\表格数据生成/v11') as dir_:
        dir_ = Path(dir_)
        temp_config["parent_dir"] = str(dir_)  # 把"parent_dir"改成了一个临时目录，使用完会自动清除。
        # temp_config['ddpm']['guide_w_list'] = [guide_w]

        # 将训练好的生成模型 copy 到临时目录中
        if eval_type != 'real':
            if sampling_method == "CoTable":
                temp_config['ddpm']['guide_w_list'] = [guide_w]
                if raw_config['ddpm']['use_guide']:
                    shutil.copy2(parent_dir / f"model_{temp_config['ddpm_train']['epoch']}.pth", temp_config["parent_dir"])
                    shutil.copy2(parent_dir / "vae_decoder_model.pth", temp_config["parent_dir"])
                    shutil.copytree(parent_dir / "latent_data", Path(temp_config["parent_dir"]) / "latent_data")
                else:
                    shutil.copy2(parent_dir / f"model_{temp_config['ddpm_train']['epoch']}_null.pth", temp_config["parent_dir"])
                    shutil.copy2(parent_dir / "vae_decoder_model.pth", temp_config["parent_dir"])
                    shutil.copytree(parent_dir / "latent_data", Path(temp_config["parent_dir"]) / "latent_data")
            elif sampling_method == "TabDDPM":
                shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])
            elif sampling_method == "TabSyn":
                shutil.copy2(parent_dir / f"model.pth", temp_config["parent_dir"])
                shutil.copy2(parent_dir / "vae_decoder_model.pth", temp_config["parent_dir"])
                shutil.copytree(parent_dir / "latent_data", Path(temp_config["parent_dir"]) / "latent_data")
            elif sampling_method == "CoDi":
                shutil.copy2(parent_dir / "model_con.pt", temp_config["parent_dir"])
                shutil.copy2(parent_dir / "model_dis.pt", temp_config["parent_dir"])
            elif sampling_method == "STaSy":
                shutil.copy2(parent_dir / "model.pth", temp_config["parent_dir"])
            elif sampling_method == "GReaT":
                shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])
            elif sampling_method == "TVAE":
                shutil.copy2(parent_dir / "tvae.obj", temp_config["parent_dir"])
            elif sampling_method == "CTGAN":
                shutil.copy2(parent_dir / "ctgan.obj", temp_config["parent_dir"])
            elif sampling_method in ["ctabgan", "ctabgan-plus"]:
                shutil.copy2(parent_dir / "ctabgan.obj", temp_config["parent_dir"])


        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            util.dump_config(temp_config, dir_ / "config.toml")
            if eval_type != 'real':
                subprocess.run(['python', f'{pipeline[sampling_method]}', '--config', f'{str(dir_ / "config.toml")}', '--sample'],  check=True)

            for seed in range(n_seeds):
                print(f'\n\n**Eval Iter: {sample_seed * n_seeds + (seed + 1)}/{n_seeds * n_datasets}**')

                metric_report = train_catboost(
                    raw_config,
                    parent_dir=temp_config['parent_dir'],
                    real_data_path=temp_config['real_data_path'],
                    eval_type=eval_type,
                    T_dict=T_dict,
                    guide_w=guide_w,
                    seed=seed,
                    change_val=change_val,
                    sampling_method=sampling_method
                )
                """
                metric_report 是一个类
                metric_report._res是一个字典，存放了一次的实验结果。其中 f1 存的是正样本和负样本的 macro avg （正样本和负样本都有自己的 f1） :
                {
                    'train': {'acc': 0.916000463445719, 'f1': 0.8243831530078591, 'roc_auc': 0.9325563852307093}, 
                    'val': {'acc': 0.9027027027027027, 'f1': 0.7973150636905457, 'roc_auc': 0.9211257756997463}, 
                    'test': {'acc': 0.9069767441860465, 'f1': 0.8030971698019753, 'roc_auc': 0.9272266958384776}
                }
                """

                metrics_seeds_report.add_report(metric_report)

    # %%
    metrics_seeds_report.get_mean_std()
    print("=" * 100)
    res = metrics_seeds_report.print_result('catboost')

    if eval_type != 'real':
        file_path = f"synthesis_{guide_w}/eval_{model_type}.json"
    else:
        file_path = f"eval_{model_type}_real.json"

    if os.path.exists(parent_dir / file_path):
        eval_dict = util.load_json(parent_dir / file_path)
        eval_dict = eval_dict | {eval_type: res}   # 字典合并操作
    else:
        eval_dict = {eval_type: res}
        if not os.path.exists(f"{parent_dir}/synthesis_{guide_w}"):
            os.makedirs(f"{parent_dir}/synthesis_{guide_w}")

    if dump:
        util.dump_json(eval_dict, parent_dir / file_path)

    raw_config['sample']['seed'] = 0
    util.dump_config(raw_config, parent_dir / 'config.toml')
    return res