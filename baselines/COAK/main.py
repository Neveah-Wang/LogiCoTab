# -*- coding: gbk -*-
"""
COAKB: Cost-Sensitive Online Adaptive Kernel Learning for Binary Classification
论文复现: Chen et al., "Cost-Sensitive Online Adaptive Kernel Learning for
Large-Scale Imbalanced Classification", IEEE TKDE, 2023.

核心算法说明:
1. 随机特征映射 (Random Feature Mapping): 将非线性核近似为有限维显式特征
2. 自适应误分类代价: c+ = (N + Tp) / (P + Tp), c- = 1
3. 在线自适应核更新: 同时对超平面向量 w 和频率分量 u 做梯度下降
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score,
    f1_score, confusion_matrix, classification_report
)

import lib
from lib.make_dataset import make_dataset_for_evaluation
from lib.metrics import evaluate, evaluate_metrics, write_avg_results_to_file


# ============================================================
# COAKB 核心算法
# ============================================================

class COAKB:
    """
    Cost-Sensitive Online Adaptive Kernel learning for Binary classification.

    参数
    ----
    D : int
        随机特征维度 (论文中从 {200,400,600,800,1000} 选取，默认400)
    eta1 : float
        超平面向量 w 的学习率 (论文推荐范围 1e-4 ~ 1e-5)
    eta2 : float
        频率分量 u 的学习率 (论文推荐范围 1e-1 ~ 1e-2)
    sigma : float
        初始 Gaussian RBF 核带宽, u0 ~ N(0, sigma^{-2} I)
    n_epochs : int
        在线训练轮数 (数据集较小时可多轮)
    random_state : int or None
        随机种子
    """

    def __init__(
        self,
        D: int = 400,
        eta1: float = 1e-4,
        eta2: float = 1e-1,
        sigma: float = 1.0,
        n_epochs: int = 1,
        random_state: int = 42,
    ):
        self.D = D
        self.eta1 = eta1
        self.eta2 = eta2
        self.sigma = sigma
        self.n_epochs = n_epochs
        self.random_state = random_state

        # 训练完成后可用
        self.w = None          # 超平面向量, shape (2D,)
        self.u = None          # 频率分量矩阵, shape (d, D)
        self.b = None          # 随机相位偏置, shape (D,)
        self.num_classes = 2

        # 内部统计 (用于自适应代价)
        self._P = 0    # 少数类样本总数
        self._N = 0    # 多数类样本总数
        self._Tp = 0   # 已正确分类的少数类样本数

    # ----------------------------------------------------------
    # 随机特征映射
    # ----------------------------------------------------------

    def _make_feature(self, X: np.ndarray) -> np.ndarray:
        """
        计算随机特征映射 z_tilde(x).

        z_tilde(x) = sqrt(2) * [cos(u[:,0]^T x + b[0]), ..., cos(u[:,D-1]^T x + b[D-1])]

        对应论文公式 (2) 和 (18).

        参数
        ----
        X : (n, d)

        返回
        ----
        Z : (n, 2D)  -- 论文用实部 cos，这里直接用 cos 拼 D 个分量
                        注: 论文公式 (18) 维度为 D，系数 sqrt(2) 已包含
        """
        # X @ u: (n, D)
        proj = X @ self.u + self.b  # (n, D)
        Z = np.sqrt(2.0 / self.D) * np.cos(proj)  # (n, D)
        return Z

    def _make_feature_grad_u(self, x: np.ndarray) -> np.ndarray:
        """
        计算单样本特征映射关于 u 的梯度系数矩阵.

        d z_tilde / d u[:, k] = -sqrt(2/D) * sin(u[:,k]^T x + b[k]) * x

        返回 shape (d, D): 每列是第 k 个频率分量的梯度方向向量
        对应论文公式 (22)-(23).
        """
        proj = x @ self.u + self.b   # (D,)
        sin_vals = np.sqrt(2.0 / self.D) * np.sin(proj)   # (D,)
        # grad_u[:, k] = -sin_vals[k] * x
        grad_u = -np.outer(x, sin_vals)   # (d, D)
        return grad_u

    # ----------------------------------------------------------
    # 训练
    # ----------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        在线训练 COAKB.

        标签约定: 多数类=0 -> 内部映射为 -1; 少数类=1 -> 内部映射为 +1.
        """
        rng = np.random.RandomState(self.random_state)

        X = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y_raw = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)

        # 标签映射: {0,1} -> {-1,+1}
        y = np.where(y_raw == 1, 1, -1).astype(np.float64)

        n, d = X.shape

        # 统计类别数量
        self._P = int((y == 1).sum())   # 少数类 (正类)
        self._N = int((y == -1).sum())  # 多数类 (负类)
        self._Tp = 0

        # 初始化参数
        # u0 ~ N(0, sigma^{-2} I), shape (d, D)
        self.u = rng.randn(d, self.D) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, size=self.D)
        self.w = np.zeros(self.D)

        eta1 = self.eta1
        eta2 = self.eta2

        for epoch in range(self.n_epochs):
            # 每轮打乱顺序 (在线学习常见做法)
            idx = rng.permutation(n)

            for i in idx:
                x_i = X[i]       # (d,)
                lb_i = y[i]      # +1 or -1
                is_minority = (lb_i == 1)

                # ---- 计算自适应代价 ρ_t (论文公式 13) ----
                # c+ = (N + Tp) / (P + Tp),  c- = 1
                if self._P + self._Tp > 0:
                    c_pos = (self._N + self._Tp) / (self._P + self._Tp)
                else:
                    c_pos = self._N / max(self._P, 1)

                rho = c_pos if is_minority else 1.0

                # ---- 计算随机特征映射 ----
                z_i = self._make_feature(x_i.reshape(1, -1)).ravel()  # (D,)

                # ---- 计算当前预测分数和铰链损失 ----
                score = self.w @ z_i          # 标量
                margin = lb_i * score         # lb_i * w^T z_i

                loss = rho * max(0.0, 1.0 - margin)

                # ---- 统计 Tp (在更新前根据当前模型判断) ----
                if is_minority and score > 0:  # 少数类预测正确
                    self._Tp += 1

                # ---- 参数更新 (仅当 loss > 0 时) ----
                if loss > 0:
                    # 更新 w (论文公式 19, 21)
                    # ?L/?w = -ρ * lb * z_tilde(x)
                    self.w += eta1 * rho * lb_i * z_i

                    # 更新 u (论文公式 20, 22-23)
                    # ?L/?u = sqrt(2) * ρ * lb * x * w^T ⊙ sin(...)
                    # 这里利用链式法则:
                    # ?L/?u = -ρ * lb * (?z_tilde/?u)^T w
                    # = ρ * lb * outer(x, sin_vals) * (w 投影)
                    grad_u_basis = self._make_feature_grad_u(x_i)  # (d, D)
                    # ?z_tilde/?u[:, k] = grad_u_basis[:, k]
                    # ?(w^T z)/?u[:, k] = w[k] * grad_u_basis[:, k]
                    # 综合: ?L/?u = -ρ * lb * grad_u_basis * w  (broadcast)
                    delta_u = -rho * lb_i * grad_u_basis * self.w  # (d, D)
                    self.u -= eta2 * delta_u

        return self

    # ----------------------------------------------------------
    # 预测
    # ----------------------------------------------------------

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        """返回 0/1 标签."""
        X = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
        Z = self._make_feature(X)          # (n, D)
        scores = Z @ self.w                # (n,)
        # 内部: +1 -> 少数类=1, -1 -> 多数类=0
        return (scores > 0).astype(int)

    def predict_proba(self, X_val: pd.DataFrame) -> np.ndarray:
        """
        返回概率估计, shape (n, 2).
        用 sigmoid 将得分映射到 [0,1].
        """
        X = X_val.values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
        Z = self._make_feature(X)
        scores = Z @ self.w
        prob_pos = 1.0 / (1.0 + np.exp(-scores))   # sigmoid
        return np.column_stack([1 - prob_pos, prob_pos])

    # ----------------------------------------------------------
    # 评估
    # ----------------------------------------------------------

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """计算并返回评估指标字典."""
        assert self.num_classes == 2

        y_true = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)

        auc = roc_auc_score(y_true, y_proba[:, 1])
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity)

        report = classification_report(
            y_true, y_pred, digits=4, output_dict=True
        )

        return {
            "acc":      report["accuracy"],
            "auc":      auc,
            "mcc":      mcc,
            "g_mean":   g_mean,
            "f1_macro": f1_macro,
            # 类别 0 (多数类)
            "p_0":  report["0"]["precision"],
            "r_0":  report["0"]["recall"],
            "f1_0": report["0"]["f1-score"],
            # 类别 1 (少数类)
            "p_1":  report["1"]["precision"],
            "r_1":  report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
        }


# ============================================================
# 超参数自动选择辅助函数 (可选)
# ============================================================

def auto_select_sigma(X_train: np.ndarray, n_sample: int = 2000) -> float:
    """
    用中位数启发式方法估计 RBF 核带宽 sigma.
    sigma = median(||x_i - x_j||) / sqrt(2)
    """
    rng = np.random.RandomState(0)
    n = X_train.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    X_sub = X_train[idx]
    # 采样部分对计算成对距离
    from sklearn.metrics.pairwise import euclidean_distances
    D_mat = euclidean_distances(X_sub)
    median_dist = np.median(D_mat[D_mat > 0])
    return max(median_dist / np.sqrt(2.0), 1e-3)


def main(raw_config):
    T_dict = raw_config['eval']['Transform']
    T_dict['normalization'] = "quantile"
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

    # ---- 自动估计 sigma ----
    sigma_est = auto_select_sigma(X_train.values)
    print(f"Auto sigma: {sigma_est:.4f}")

    all_results = defaultdict(list)

    for seed in list(range(10)):
        # ---- 训练 COAKB ----
        model = COAKB(
            D=400,
            eta1=1e-4,
            eta2=1e-1,
            sigma=sigma_est,
            n_epochs=3,  # 小数据集可多跑几轮
            random_state=seed,
        )
        model.fit(X_train, y_train)

        # ---- 评估 ----
        metrics = model.evaluate(X_val, y_val)
        for k, v in metrics.items():
            all_results[k].append(v)

    # 求均值
    avg_results = {k: np.mean(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"COAK, Dataset:{dataname}",
        log_file="eval_average.log"
    )

    # 求最大值
    avg_results = {k: np.max(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"COAK, Dataset:{dataname}",
        log_file="eval_max.log"
    )

    # 求中位数
    avg_results = {k: np.median(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"COAK, Dataset:{dataname}",
        log_file="eval_median.log"
    )

if __name__ == "__main__":
    raw_config_list = []
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp\churn\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/adult\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/shopper\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/Magic\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/bean\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/winequality\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/obesity\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/yeast_me2\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/page\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/buddy\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/mammography\CoTable\config.toml"))


    for raw_config in raw_config_list:
        main(raw_config)
