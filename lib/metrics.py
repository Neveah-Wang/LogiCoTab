import enum
from typing import Any, Optional, Tuple, Dict, Union, cast, Literal
from functools import partial

import numpy as np
import scipy.special
import sklearn.metrics as skm

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
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }

        print("train: ", end=' ')
        print(res["train"])
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
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def get_mean_std_min_max(self) -> dict:
        res = {k: {} for k in ["train", "val", "test"]}
        splits = self._reports[0].get_splits_names()
        metrics = self._reports[0].get_metrics_names()

        for split in splits:
            for metric in metrics:
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
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
        res = {split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]} for split in ["train", "val", "test"]}
        print(f"\nEVAL RESULTS of {model_name}:")
        print("[train] ", end=' ')
        print(res["train"])
        print("[val] ", end=' ')
        print(res["val"])
        print("[test] ", end=' ')
        print(res["test"])
        return res



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

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == 'binclass'
            else scipy.special.softmax(y_pred, axis=1)
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
