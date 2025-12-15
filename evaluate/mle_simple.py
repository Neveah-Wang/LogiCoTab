# import zero
from pathlib import Path
import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path
from copy import deepcopy
import shutil
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

import lib
from lib import util, metrics
from lib.metrics import SeedsMetricsReport
from lib.make_dataset import make_dataset_for_evaluation

pipeline = {
    'ddpm': 'main_sample.py',
    'smote': 'main_smote.py',
    'ctabgan': 'main_ctabgan.py',
    'ctabgan-plus': 'main_ctabganp.py',
    'tvae': 'main_tvae.py'
}


def train_simple(raw_config, parent_dir, real_data_path, eval_type, T_dict, guide_w,
                 model_name="tree", seed=0, change_val=False,
                 params=None,  # dummy
                 device=None  # dummy
                 ):
    synthetic_data_path = os.path.join(parent_dir, f'synthesis_{guide_w}')
    dataset, X = make_dataset_for_evaluation(raw_config, synthetic_data_path, real_data_path, eval_type, T_dict, change_val)

    # 1. 初始化模型
    # zero.improve_reproducibility(seed)
    if dataset.is_regression():
        models = {
            "tree": DecisionTreeRegressor(max_depth=28, random_state=seed),
            "rf": RandomForestRegressor(max_depth=28, random_state=seed),
            "lr": Ridge(max_iter=500, random_state=seed),
            "mlp": MLPRegressor(max_iter=100, random_state=seed)
        }
    else:
        models = {
            "tree": DecisionTreeClassifier(max_depth=250, random_state=seed),
            "rf": RandomForestClassifier(max_depth=250, random_state=seed),
            "lr": LogisticRegression(max_iter=500, n_jobs=2, random_state=seed),
            "mlp": MLPClassifier(max_iter=500, random_state=seed)
        }

    model = models[model_name]

    # 2. 训练模型
    model.fit(X['train'], dataset.y['train'].ravel())

    # 3. 预测
    predict = (
        model.predict
        if dataset.is_regression()
        else model.predict_proba
        if dataset.is_multiclass()
        else lambda x: model.predict_proba(x)[:, 1]
    )
    predictions = {k: predict(v) for k, v in X.items()}

    # 4. 评估
    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = dataset.calculate_metrics(predictions, None if dataset.is_regression() else 'probs')  # 计算各项指标
    # print("predictions: ", predictions)

    # 打印指标
    print('-' * (len(str({model.__class__.__name__})) + 12))
    print(f'|       {model.__class__.__name__}       |')
    print('-' * (len(str({model.__class__.__name__})) + 12))
    metrics_report = metrics.MetricsReport(report['metrics'], dataset.task_type)
    metrics_report.print_metrics()

    # if parent_dir is not None:
    # lib.dump_json(report, os.path.join(parent_dir, "results_catboost.json"))

    return metrics_report


def eval_seeds_simple(
        raw_config,
        n_seeds,
        eval_type,
        guide_w,
        sampling_method="ddpm",
        model_type="simple",
        n_datasets=1,
        dump=True,
        change_val=False
):
    parent_dir = Path(raw_config["parent_dir"])
    models = ["tree", "lr", "rf", "mlp"]
    metrics_seeds_report = {k: SeedsMetricsReport() for k in models}

    if eval_type == 'real':
        n_datasets = 1

    T_dict = deepcopy(raw_config['eval']['Transform'])
    T_dict["normalization"] = "minmax"  # 不同的模型，需要不同的标准化方法，因此后期设置起来会比较方便
    T_dict["cat_encode_policy"] = "Ordinal"

    temp_config = deepcopy(raw_config)

    # %%  通过with语句创建临时文件，with会自动关闭临时文件
    with tempfile.TemporaryDirectory(dir='D:\Study\自学\表格数据生成/v11') as dir_:
        dir_ = Path(dir_)
        temp_config["parent_dir"] = str(dir_)
        temp_config['ddpm']['guide_w_list'] = [guide_w]

        # 将训练好的生成模型 copy 到临时目录中
        if eval_type != 'real':
            if sampling_method == "ddpm":
                if raw_config['ddpm']['use_guide']:
                    shutil.copy2(parent_dir / f"model_{temp_config['ddpm_train']['epoch']}.pth", temp_config["parent_dir"])
                    shutil.copy2(parent_dir / "vae_decoder_model.pth", temp_config["parent_dir"])
                    shutil.copytree(parent_dir / "latent_data", Path(temp_config["parent_dir"])/"latent_data")
                else:
                    shutil.copy2(parent_dir / f"model_{temp_config['ddpm_train']['epoch']}_null.pth", temp_config["parent_dir"])
                    shutil.copy2(parent_dir / "vae_decoder_model.pth", temp_config["parent_dir"])
                    shutil.copytree(parent_dir / "latent_data", Path(temp_config["parent_dir"])/"latent_data")
            elif sampling_method in ["ctabgan", "ctabgan-plus"]:
                shutil.copy2(parent_dir / "ctabgan.obj", temp_config["parent_dir"])
            elif sampling_method == "tvae":
                shutil.copy2(parent_dir / "tvae.obj", temp_config["parent_dir"])

        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            lib.util.dump_config(temp_config, dir_ / "config.toml")  # 将 config 文件保存到临时文件夹
            if eval_type != 'real':
                subprocess.run(['python', f'{pipeline[sampling_method]}', '--config', f'{str(dir_ / "config.toml")}'], check=True)   # 采样

            for seed in range(n_seeds):
                print('\n', '*' * 30, f'* Eval Iter: {sample_seed * n_seeds + (seed + 1)}/{n_seeds * n_datasets} *', '*' * 30, end='\n\n')

                for model in models:
                    metric_report = train_simple(
                        raw_config,
                        parent_dir=temp_config['parent_dir'],
                        real_data_path=temp_config['real_data_path'],
                        model_name=model,
                        eval_type=eval_type,
                        T_dict=T_dict,
                        guide_w=guide_w,
                        seed=seed,
                        change_val=change_val,
                    )

                    metrics_seeds_report[model].add_report(metric_report)

    # %%

    # 对上面多次重复运行的结果，求mean和sta，并打印
    for k in models:
        metrics_seeds_report[k].get_mean_std()
    print("=" * 100)
    res = {k: metrics_seeds_report[k].print_result(k) for k in models}

    # 将上面的结果保存在 eval_model.josn中
    # m1, m2 = ("r2-mean", "rmse-mean") if "r2-mean" in res["tree"]["val"] else ("f1-mean", "acc-mean")
    m1, m2, m3 = ("f1-mean", "acc-mean", "roc_auc-mean")
    res["avg"] = {
        "val": {
            m1: np.around(np.mean([res[k]["val"][m1] for k in models]), 4),
            m2: np.around(np.mean([res[k]["val"][m2] for k in models]), 4),
            m3: np.around(np.mean([res[k]["val"][m3] for k in models]), 4)
        },
        "test": {
            m1: np.around(np.mean([res[k]["test"][m1] for k in models]), 4),
            m2: np.around(np.mean([res[k]["test"][m2] for k in models]), 4),
            m3: np.around(np.mean([res[k]["test"][m3] for k in models]), 4)
        },
    }

    if eval_type != 'real':
        file_path = f"synthesis_{guide_w}/eval_{model_type}.json"
    else:
        file_path = f"eval_{model_type}_real.json"

    if os.path.exists(parent_dir / file_path):
        eval_dict = util.load_json(parent_dir / file_path)
        # eval_dict = eval_dict | {eval_type: res}  这是字典合并操作，仅在python3.9及以上版本支持该操作
        eval_dict = {**eval_dict, **{eval_type: res}}
    else:
        eval_dict = {eval_type: res}
        if not os.path.exists(f"{parent_dir}/synthesis_{guide_w}"):
            os.makedirs(f"{parent_dir}/synthesis_{guide_w}")

    if dump:
        util.dump_json(eval_dict, parent_dir / file_path)

    raw_config['sample']['seed'] = 0
    util.dump_config(raw_config, parent_dir / 'config.toml')

    return res