import enum
from typing import Any, Optional, Tuple, Dict, Union, cast, Literal
from functools import partial

import numpy as np
import scipy.special
import sklearn.metrics as skm
from catboost import CatBoostClassifier
from collections import defaultdict

from . import util
from lib.util import TaskType

class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'


class MetricsReport:
    def __init__(self, report: dict, task_type: TaskType):
        self._res = {k: {} for k in report.keys()}

        if task_type in ('binclass', 'multiclass'):
            self._metrics_names = ["acc", "f1"]
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                if task_type == 'binclass':
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]
                    self._metrics_names.append("roc_auc")

        elif task_type == 'regression':
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list:
        return self._res.keys()

    def get_metrics_names(self) -> list:
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        return self._res[split][metric]

    def get_val_score(self) -> float:
        return self._res["val"]["r2"] if "r2" in self._res["val"] else self._res["val"]["f1"]
    
    def get_test_score(self) -> float:
        return self._res["test"]["r2"] if "r2" in self._res["test"] else self._res["test"]["f1"]
    
    def print_metrics(self) -> None:
        res = {
            "train": {k: np.around(self._res["train"][k], 4) for k in self._res["train"]},
            "train_real": {k: np.around(self._res["train_real"][k], 4) for k in self._res["train_real"]},
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }

        print("train: ", end=' ')
        print(res["train"])
        print("train_real: ", end=' ')
        print(res["train_real"])
        print("val: ", end=' ')
        print(res["val"])
        print("test: ", end=' ')
        print(res["test"], end='\n\n')

        return res



class SeedsMetricsReport:
    def __init__(self):
        self._reports = []

    def add_report(self, report: MetricsReport) -> None:
        self._reports.append(report)
    
    def get_mean_std(self) -> dict:
        res = {k: {} for k in ["train", "val", "test", "train_real"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test", "train_real"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def get_mean_std_min_max(self) -> dict:
        res = {k: {} for k in ["train", "val", "test", "train_real"]}
        splits = self._reports[0].get_splits_names()
        metrics = self._reports[0].get_metrics_names()

        for split in splits:
            for metric in metrics:
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test", "train_real"]}
        # 扩展统计函数：新增 min 和 max
        stats_funcs = [
            ("count", len),
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max)
        ]

        for split in splits:
            for metric in metrics:
                for suffix, func in stats_funcs:
                    agg_res[split][f"{metric}-{suffix}"] = float(func(res[split][metric]))

        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self, model_name) -> dict:
        res = {split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]} for split in ["train", "val", "test", "train_real"]}
        print(f"\nEVAL RESULTS of {model_name}:")
        print("[train] ", end=' ')
        print(res["train"])
        print("[train_real] ", end=' ')
        print(res["train_real"])
        print("[val] ", end=' ')
        print(res["val"])
        print("[test] ", end=' ')
        print(res["test"])
        return res


def aggregate_metrics(metrics_list):
    if not metrics_list:
        raise ValueError("metrics_list is empty")

    # 定义所有需要聚合的路径
    # 每个路径是 (split, sub_key, metric_name) 三元组
    paths = []
    splits = list(metrics_list[0].keys())  # ['train', 'val', 'test', 'train_real']
    labels = ['0', '1', 'macro avg', 'weighted avg']
    label_metrics = ['precision', 'recall', 'f1-score', 'support']
    global_metrics = ['accuracy', 'roc_auc', 'score']

    for split in splits:
        for label in labels:
            for m in label_metrics:
                paths.append((split, label, m))
        for m in global_metrics:
            paths.append((split, m))  # 注意：全局指标只有两层

    # 初始化结果字典（按第一个元素结构复制）
    result = {}
    for split in splits:
        result[split] = {}
        # 类别和 avg 指标
        for label in labels:
            result[split][label] = {}
            for m in label_metrics:
                result[split][label][m] = {}
        # 全局指标
        for m in global_metrics:
            result[split][m] = {}

    # 对每条路径收集数值并计算统计量
    for path in paths:
        values = []
        for metrics in metrics_list:
            if len(path) == 3:
                # (split, label, metric)
                val = metrics[path[0]][path[1]][path[2]]
            else:
                # (split, metric)
                val = metrics[path[0]][path[1]]
            values.append(val)

        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values)
        }

        # 写回 result
        if len(path) == 3:
            result[path[0]][path[1]][path[2]] = stats
        else:
            result[path[0]][path[1]] = stats

    return result


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
        y_pred: np.ndarray,
        task_type: TaskType,
        prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    assert task_type in ('binclass', 'multiclass')

    if prediction_type is None:
        return y_pred, None

    # 如果 y_pred 是 logits（未归一化的原始分数）
    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)   # 二分类：用 scipy.special.expit（即 sigmoid 函数）将其转为 [0,1] 的概率。
            if task_type == 'binclass'
            else scipy.special.softmax(y_pred, axis=1) # 多元分类：用 scipy.special.softmax 沿 axis=1（对每个样本的类别维度）做 softmax，得到概率分布。
        )
        # print("probs01: ", probs)
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
        # print("probs02: ", probs)
    else:
        util.raise_unknown('prediction_type', prediction_type)

    # print("probs 0: ", probs)
    assert probs is not None
    labels = np.round(probs) if task_type == 'binclass' else probs.argmax(axis=1)  # 如果是 binclass, 就对预测值四舍五入；如果是 multiclass，就取最大的

    # print("probs 1: ", probs)
    return labels, probs


def calculate_metrics(
    y_true: np.ndarray,  # y 的真实值
    y_pred: np.ndarray,  # y 的预测值
    task_type: TaskType,
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],    # y 的 info，例如mean, sta
) -> Dict[str, Any]:
    """
    Example: calculate_metrics(y_true, y_pred, 'binclass', 'probs', {})
    """
    # print("task_type: ", task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == 'regression':
        assert prediction_type is None
        # assert 'std' in y_info
        std = y_info.get("std", None)
        rmse = calculate_rmse(y_true, y_pred, std)
        r2 = skm.r2_score(y_true, y_pred)
        result = {'rmse': rmse, 'r2': r2}
    else:
        # print("y_pred: ", y_pred)
        # print("task_type: ", task_type)
        # print("prediction_type: ", prediction_type)

        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        labels = labels.astype(y_true.dtype)

        # 使用 skm.classification_report 计算指标， cast将结果转换为Dict类型
        result = cast(Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True))

        if task_type == 'binclass':
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result



from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from contextlib import redirect_stdout


""" 下面这两个函数，主要用于单个的评估结果 """
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


""" 下面的这两个函数，用于求多次结果的平均值 """
def evaluate_metrics(classifier, X, y):
    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)[:, 1]

    # 基本指标
    auc = roc_auc_score(y, y_score)
    mcc = matthews_corrcoef(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')

    # confusion matrix -> G-mean
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    # classification report（字典形式！）
    report = classification_report(
        y, y_pred, digits=4, output_dict=True
    )

    return {
        "acc": report["accuracy"],
        "auc": auc,
        "mcc": mcc,
        "g_mean": g_mean,
        "f1_macro": f1_macro,

        # 类别 0
        "p_0": report["0"]["precision"],
        "r_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],

        # 类别 1
        "p_1": report["1"]["precision"],
        "r_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
    }

def evaluate_multiple_seeds(
    X_train, y_train,
    X_val, y_val,
    seeds,
    catboost_params=None
):
    if catboost_params is None:
        catboost_params = {}

    all_results = defaultdict(list)

    for seed in seeds:
        clf = CatBoostClassifier(
            random_seed=seed,
            verbose=False,
            **catboost_params
        )
        clf.fit(X_train, y_train)

        metrics = evaluate_metrics(clf, X_val, y_val)
        for k, v in metrics.items():
            all_results[k].append(v)

    # 求均值
    avg_results = {k: np.mean(v) for k, v in all_results.items()}
    return avg_results

def write_avg_results_to_file(avg_results, help_str, log_file):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 20 + f" {help_str} (Average over seeds) " + "-" * 20 + "\n")

        f.write(f"ACC: {avg_results['acc']:.4f}\n")
        f.write(f"AUC: {avg_results['auc']:.4f}\n")
        f.write(f"MCC: {avg_results['mcc']:.4f}\n")
        f.write(f"G-mean: {avg_results['g_mean']:.4f}\n")
        f.write(f"F1 (macro): {avg_results['f1_macro']:.4f}\n\n")

        f.write("Class-wise metrics (averaged):\n")
        f.write("Class 0:\n")
        f.write(f"  Precision: {avg_results['p_0']:.4f}\n")
        f.write(f"  Recall:    {avg_results['r_0']:.4f}\n")
        f.write(f"  F1-score:  {avg_results['f1_0']:.4f}\n")

        f.write("Class 1:\n")
        f.write(f"  Precision: {avg_results['p_1']:.4f}\n")
        f.write(f"  Recall:    {avg_results['r_1']:.4f}\n")
        f.write(f"  F1-score:  {avg_results['f1_1']:.4f}\n")