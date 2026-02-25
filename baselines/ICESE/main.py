# -*- coding: gbk -*-

"""
ICESE: Improved Contraction-Expansion Subspace Ensemble
for High-Dimensional Imbalanced Data Classification

论文复现：Xu et al., 2024, IEEE Transactions on Knowledge and Data Engineering

核心模块：
  - SMOTE 过采样
  - CESO (Contraction-Expansion Subspace Optimization)
  - 多层 CESO 优化结构
  - ICESE 集成分类器
"""
# 加入下面这四行，是为了解决无法找到自定义包路径的问题
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
sys.path.append(r'D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT')

import lib
from lib.make_dataset import make_dataset_for_evaluation
from lib.metrics import evaluate, evaluate_metrics, write_avg_results_to_file


import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SMOTE 过采样
# =============================================================================

def smote_oversample(X_min, n_synthetic, k_neighbors=5, random_state=None):
    """
    SMOTE：对少数类样本线性插值生成合成样本

    公式：X_syn = X_min + λ * (X_min_nn - X_min),  λ ~ Uniform(0, 1)
    """
    rng = check_random_state(random_state)
    n_min = len(X_min)

    if n_min <= 1:
        return X_min[rng.randint(0, n_min, n_synthetic)].copy()

    k = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X_min)
    _, indices = nn.kneighbors(X_min)
    nn_indices = indices[:, 1:]  # 去掉自身，shape: (n_min, k)

    X_syn = []
    for _ in range(n_synthetic):
        i = rng.randint(0, n_min)
        nn_idx = nn_indices[i, rng.randint(0, k)]
        lam = rng.uniform(0.0, 1.0)
        X_syn.append(X_min[i] + lam * (X_min[nn_idx] - X_min[i]))

    return np.array(X_syn)


# =============================================================================
# CESO：收缩-扩展子空间优化
# =============================================================================

class CESO:
    """
    Contraction-Expansion Subspace Optimization

    收缩阶段：加权随机森林计算特征重要性，选取 top-(L*mu) 个特征 → Ψ
    扩展阶段：在 Ψ 上做局部 PCA 旋转变换 → Φ = Ψ · M'
    输出：Γ = [Ψ, Φ]（拼接，公式9）

    参数
    ----
    mu          : 特征保留率，默认0.05（5%）
    n_trees     : 加权随机树数量，默认20
    omega       : 每个局部旋转子空间的特征数，默认3
    random_state
    """

    def __init__(self, mu=0.05, n_trees=20, omega=3, random_state=None):
        self.mu = mu
        self.n_trees = n_trees
        self.omega = omega
        self.random_state = random_state
        self.selected_features_ = None
        self.rotation_matrix_ = None

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _sample_weights(self, y):
        """
        W_i = N / (C * N_i)
        权重与类频率成反比，缓解类不平衡对特征选择的影响（公式1）
        """
        classes, counts = np.unique(y, return_counts=True)
        n_total, n_classes = len(y), len(classes)
        weight_map = {c: n_total / (n_classes * cnt)
                      for c, cnt in zip(classes, counts)}
        return np.array([weight_map[yi] for yi in y])

    def _contraction(self, X, y, rng):
        """
        收缩阶段（公式2-6）：
        T 棵加权随机树，每棵子空间大小 sqrt(L)，
        累加 Gini 重要性，选 top-(L*mu) 特征作为收缩子空间 Ψ
        """
        n_features = X.shape[1]
        sample_weights = self._sample_weights(y)
        max_features = max(1, int(np.sqrt(n_features)))
        importances = np.zeros(n_features)

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_features=max_features,
                random_state=rng.randint(0, 2**31)
            )
            tree.fit(X, y, sample_weight=sample_weights)
            importances += tree.feature_importances_

        n_selected = max(1, int(n_features * self.mu))
        return np.argsort(importances)[::-1][:n_selected]

    def _expansion(self, X_psi, rng):
        """
        扩展阶段（公式7-8）：
        将 Ψ 随机分成若干大小为 omega 的局部子空间，
        各自用 75% bootstrap 样本做 PCA，
        组装块对角旋转矩阵 M，计算 Φ = Ψ · M'
        """
        n_samples, n_contracted = X_psi.shape
        omega = min(self.omega, n_contracted)

        perm = rng.permutation(n_contracted)
        groups = [perm[s: min(s + omega, n_contracted)]
                  for s in range(0, n_contracted, omega)]

        M = np.zeros((n_contracted, n_contracted))
        for grp in groups:
            X_sub = X_psi[:, grp]
            n_boot = max(2, int(0.75 * n_samples))
            boot_idx = rng.choice(n_samples, size=n_boot, replace=True)
            n_comp = len(grp)
            try:
                pca = PCA(n_components=n_comp)
                pca.fit(X_sub[boot_idx])
                C = pca.components_.T          # shape: (n_comp, n_comp)
            except Exception:
                C = np.eye(n_comp)
            for li, gi in enumerate(grp):
                for lj, gj in enumerate(grp):
                    M[gi, gj] = C[li, lj]

        return M

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fit_transform(self, X, y):
        """训练并变换：返回 Γ = [Ψ, Φ]（公式9）"""
        rng = check_random_state(self.random_state)
        self.selected_features_ = self._contraction(X, y, rng)
        Psi = X[:, self.selected_features_]
        self.rotation_matrix_ = self._expansion(Psi, rng)
        Phi = Psi @ self.rotation_matrix_
        return np.concatenate([Psi, Phi], axis=1)

    def transform(self, X):
        """用已学习的参数变换新数据"""
        Psi = X[:, self.selected_features_]
        Phi = Psi @ self.rotation_matrix_
        return np.concatenate([Psi, Phi], axis=1)


# =============================================================================
# 多层 CESO 优化结构
# =============================================================================

def _multilayer_ceso_fit_transform(X, y, Delta, mu, omega, n_trees, rng):
    """
    多层 CESO（对应论文 Fig.3 / Algorithm 2 内层循环）

    第 1 层特征保留率 = mu
    第 2~Delta 层特征保留率 = 0.5（保持子空间规模一致）

    返回
    ----
    X_improved : 最终改进子空间（训练集），shape (n_samples, 2*n_contracted)
    chain      : 各层 CESO 对象列表，用于测试集变换
    """
    chain = []
    X_temp = X

    for j in range(Delta):
        mu_j = mu if j == 0 else 0.5
        ceso = CESO(mu=mu_j, n_trees=n_trees, omega=omega,
                    random_state=rng.randint(0, 2**31))
        X_temp = ceso.fit_transform(X_temp, y)
        chain.append(ceso)

    return X_temp, chain


def _multilayer_ceso_transform(X, chain):
    """将已训练的 CESO 链依次应用于测试集"""
    X_temp = X
    for ceso in chain:
        X_temp = ceso.transform(X_temp)
    return X_temp


# =============================================================================
# ICESE 主类
# =============================================================================

class ICESE(BaseEstimator, ClassifierMixin):
    """
    Improved Contraction-Expansion Subspace Ensemble (ICESE)

    论文：Xu et al., 2024, IEEE TKDE

    参数
    ----
    k            : 基分类器数量，默认10
    Delta        : 多层CESO优化层数，默认8
    mu           : 第1层CESO特征保留率，默认0.05
    omega        : 局部旋转子空间特征数，默认3
    n_trees      : CESO内加权随机树数量，默认20
    smote_k      : SMOTE中KNN邻居数，默认5
    random_state : 随机种子

    用法
    ----
    model = ICESE(k=10, Delta=8, mu=0.05)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_val)
    y_score = model.predict_proba(X_val)[:, 1]
    """

    def __init__(self, k=10, Delta=8, mu=0.05, omega=3,
                 n_trees=20, smote_k=5, random_state=None):
        self.k = k
        self.Delta = Delta
        self.mu = mu
        self.omega = omega
        self.n_trees = n_trees
        self.smote_k = smote_k
        self.random_state = random_state

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _preprocess_y(self, y):
        """
        统一标签为 int ndarray：少数类 → 1，多数类 → 0
        支持任意原始标签类型（字符串、浮点等）
        """
        y = np.asarray(y).ravel()
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)

        classes, counts = np.unique(y_enc, return_counts=True)
        min_cls = classes[np.argmin(counts)]

        # 确保少数类编码为 1
        if min_cls != 1:
            y_enc = (y_enc == min_cls).astype(int)

        return y_enc

    def _balance_smote(self, X, y, rng):
        """
        在改进子空间上执行 SMOTE，返回平衡训练集（公式10-12）

        q = (|maj| - |min|) / |min|
        """
        X_maj = X[y == 0]
        X_min = X[y == 1]
        n_syn = len(X_maj) - len(X_min)

        if n_syn <= 0 or len(X_min) == 0:
            return X, y

        X_syn = smote_oversample(X_min, n_syn,
                                  k_neighbors=self.smote_k,
                                  random_state=rng.randint(0, 2**31))
        X_bal = np.vstack([X, X_syn])
        y_bal = np.concatenate([y, np.ones(len(X_syn), dtype=y.dtype)])
        return X_bal, y_bal

    # ------------------------------------------------------------------
    # sklearn 接口
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        训练 ICESE（Algorithm 2）

        对每个基分类器 i = 1..k：
          1. 执行 Delta 层 CESO → 改进子空间 Γ^Δ
          2. 在 Γ^Δ 上 SMOTE → 平衡子集 B_i
          3. 用 B_i 训练决策树 D_i
        """
        X = np.asarray(X, dtype=float)
        y = self._preprocess_y(y)
        self.classes_ = np.array([0, 1])

        rng = check_random_state(self.random_state)
        self.estimators_ = []
        self.chains_ = []

        for _ in range(self.k):
            # Step 1：多层 CESO
            X_imp, chain = _multilayer_ceso_fit_transform(
                X, y,
                Delta=self.Delta, mu=self.mu,
                omega=self.omega, n_trees=self.n_trees,
                rng=rng
            )
            # Step 2：SMOTE 平衡
            X_bal, y_bal = self._balance_smote(X_imp, y, rng)

            # Step 3：训练决策树基分类器
            tree = DecisionTreeClassifier(random_state=rng.randint(0, 2**31))
            tree.fit(X_bal, y_bal)

            self.estimators_.append(tree)
            self.chains_.append(chain)

        return self

    def predict_proba(self, X):
        """
        各基分类器概率均值（公式13）
        Proc(π) = (1/k) Σ P^c_i(π_i)
        """
        X = np.asarray(X, dtype=float)
        proba_sum = None

        for tree, chain in zip(self.estimators_, self.chains_):
            X_imp = _multilayer_ceso_transform(X, chain)
            p = tree.predict_proba(X_imp)

            # 防御：若基分类器只见过一个类，补全为两列
            if p.shape[1] == 1:
                fill = 1 - p
                p = np.hstack([fill, p]) if tree.classes_[0] == 1 \
                    else np.hstack([p, fill])

            proba_sum = p if proba_sum is None else proba_sum + p

        return proba_sum / self.k

    def predict(self, X):
        """多数投票（公式14）：F(π) = argmax_c Proc(π)"""
        proba = self.predict_proba(X)
        pred_enc = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(pred_enc)


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
    # breakpoint()
    all_results = defaultdict(list)

    for seed in list(range(10)):
        model = ICESE(k=10, Delta=6, mu=0.5, random_state=seed)
        model.fit(X_train, y_train)

        # evaluate(model, X_val, y_val, help=f"ICESE, Dataset:{dataname}")
        metrics = evaluate_metrics(model, X_val, y_val)
        for k, v in metrics.items():
            all_results[k].append(v)

    # 求均值
    avg_results = {k: np.mean(v) for k, v in all_results.items()}

    write_avg_results_to_file(
        avg_results,
        help_str=f"ICESE, Dataset:{dataname}",
        log_file="eval_average.log"
    )

    # 求最大值
    avg_results = {k: np.max(v) for k, v in all_results.items()}

    write_avg_results_to_file(
        avg_results,
        help_str=f"ICESE, Dataset:{dataname}",
        log_file="eval_max.log"
    )
# =============================================================================
# 主程序入口（基于用户提供的代码）
# =============================================================================

if __name__ == "__main__":

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