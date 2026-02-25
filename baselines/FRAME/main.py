# -*- coding: gbk -*-

"""
FRAME: Feature Rectification for Class Imbalance Learning
复现自论文：Cheng et al., IEEE TKDE, Vol. 37, No. 3, March 2025

核心组件：
  1. EmbeddingNetwork      — 嵌入网络，将原始特征映射到隐空间
  2. SelfAttentiveCentroid — 自注意力中心学习模块（含分块策略）
  3. DistanceClassifier    — 基于距离的分类器（替代 Softmax）
  4. FRAMEModel            — 组合以上三个模块的完整模型
  5. FRAMETrainer          — 封装训练 / 推理 / 评估逻辑

用法示例（直接复制到主程序入口即可）：
    trainer = FRAMETrainer(input_dim=X_train.shape[1], num_classes=2)
    trainer.fit(X_train, y_train)
    metrics = evaluate(trainer, X_val, y_val)
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, classification_report
)


import lib
from lib.make_dataset import make_dataset_for_evaluation
from lib.metrics import evaluate, evaluate_metrics, write_avg_results_to_file

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor(X, y=None, device="cpu"):
    """将 DataFrame / ndarray / Series 转换为 FloatTensor。"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
    if y is not None:
        if isinstance(y, pd.Series):
            y = y.values
        y = torch.tensor(np.asarray(y, dtype=np.int64)).to(device)
        return X, y
    return X


def _gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 G-Mean（二分类 / 多分类均支持）。"""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-10)
        spec = tn / (tn + fp + 1e-10)
        return math.sqrt(sens * spec)
    recalls = [cm[i, i] / (cm[i].sum() + 1e-10) for i in range(n_classes)]
    gm = 1.0
    for r in recalls:
        gm *= r
    return gm ** (1.0 / n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 模块一：嵌入网络  Φ: X → F
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingNetwork(nn.Module):
    """
    论文 Section III-B, Eq.(23)
    单隐层 MLP，论文实验使用 128 个过滤器。
    """

    def __init__(self, input_dim: int, latent_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, D)  →  (N, F)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 模块二：自注意力中心学习
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttentiveCentroid(nn.Module):
    """
    论文 Section III-B2：Centroid Learning Via Self-Attentive Feature Interactions

    流程：
        H_k  →  Z_k  (MLP_z)
             →  q, k, v  (MLP_q / MLP_kv)
             →  分块点积注意力  (chunked self-attention)
             →  残差连接
             →  MLP_c  →  centroids c_k  ∈ R^{nc × F}

    超参（论文最优）：chunk_size = 256, num_centroids = 2
    """

    def __init__(self, latent_dim: int = 128,
                 num_centroids: int = 2,
                 chunk_size: int = 256):
        super().__init__()
        self.latent_dim    = latent_dim
        self.num_centroids = num_centroids
        self.chunk_size    = chunk_size

        # Step1：初始特征嵌入 Z_k = MLP_z(H_k)
        self.mlp_z = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

        # Step2：投影到 Q / K / V
        self.mlp_q  = nn.Linear(latent_dim, latent_dim)
        self.mlp_kv = nn.Linear(latent_dim, latent_dim * 2)   # concat(k, v)

        # Step5：从增强特征生成中心
        self.mlp_c = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim * num_centroids),
        )

    # ── 分块自注意力 ──────────────────────────────────────────────────────

    def _chunked_attention(self, q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
        """
        论文 chunking strategy：逐块累加避免 N?  内存。

        q, k, v : (N_k, F)
        返回    : (N_k, F)
        """
        N, F = q.shape
        scale = math.sqrt(F)

        # 样本量小时直接做全局注意力
        if N <= self.chunk_size:
            attn = torch.softmax(q @ k.T / scale, dim=-1)   # (N, N)
            return attn @ v                                   # (N, F)

        # 分块 running-sum（数值稳定）
        out     = torch.zeros_like(v)          # (N, F)
        exp_sum = torch.zeros(N, 1, device=q.device)

        for start in range(0, N, self.chunk_size):
            end  = min(start + self.chunk_size, N)
            k_c  = k[start:end]               # (chunk, F)
            v_c  = v[start:end]               # (chunk, F)

            scores     = q @ k_c.T / scale    # (N, chunk)
            max_scores = scores.max(dim=-1, keepdim=True).values
            exp_s      = torch.exp(scores - max_scores)
            exp_sum   += exp_s.sum(dim=-1, keepdim=True)
            out        += exp_s @ v_c         # (N, F)

        return out / (exp_sum + 1e-9)

    # ── 前向传播 ──────────────────────────────────────────────────────────

    def forward(self, H_k: torch.Tensor) -> torch.Tensor:
        """
        H_k : (N_k, F)  — 单类的隐特征矩阵
        返回 : (num_centroids, F)
        """
        # Step1：嵌入
        Z_k = self.mlp_z(H_k)                            # (N_k, F)

        # Step2：Q / K / V
        q  = self.mlp_q(Z_k)                             # (N_k, F)
        kv = self.mlp_kv(Z_k)                            # (N_k, 2F)
        k, v = kv.chunk(2, dim=-1)                       # 各 (N_k, F)

        # Step3-4：分块注意力 + 残差
        attended = self._chunked_attention(q, k, v)      # (N_k, F)
        enhanced = attended + Z_k                        # (N_k, F)  残差

        # 均值池化 → 全局代表向量
        pooled = enhanced.mean(dim=0, keepdim=True)      # (1, F)

        # Step5：生成 num_centroids 个中心
        c_flat = self.mlp_c(pooled)                      # (1, F * nc)
        c_k    = c_flat.view(self.num_centroids, self.latent_dim)  # (nc, F)

        return c_k


# ─────────────────────────────────────────────────────────────────────────────
# 模块三：基于距离的分类器
# ─────────────────────────────────────────────────────────────────────────────

class DistanceClassifier(nn.Module):
    """
    论文 Section III-B3，Eq.(3)：

        p(y=i | H_s) = exp(-||H_s - c_i||?)
                       ─────────────────────
                       Σ_k exp(-||H_s - c_k||?)

    每类可有多个中心；推断时取同类各中心中距离最近者作为该类距离。
    """

    def forward(self, H_s: torch.Tensor,
                centroids: torch.Tensor) -> torch.Tensor:
        """
        H_s       : (N, F)
        centroids : (C, nc, F)
        返回      : (N, C)  log 概率
        """
        # 广播计算每个样本到每类每个中心的平方欧氏距离
        # H_s      : (N, 1, 1, F)
        # centroids: (1, C, nc, F)
        H_exp = H_s.unsqueeze(1).unsqueeze(2)      # (N, 1,  1, F)
        c_exp = centroids.unsqueeze(0)             # (1, C, nc, F)
        dists = ((H_exp - c_exp) ** 2).sum(-1)     # (N, C, nc)

        # 取同类中最近中心的距离
        min_dists, _ = dists.min(dim=-1)           # (N, C)

        # 论文 Eq.(3)：log_softmax(-dist)
        return F.log_softmax(-min_dists, dim=-1)   # (N, C)


# ─────────────────────────────────────────────────────────────────────────────
# 完整 FRAME 模型
# ─────────────────────────────────────────────────────────────────────────────

class FRAMEModel(nn.Module):
    """
    全监督 FRAME（论文 Section III-C1）

    训练时传入 y → 内部计算中心并返回 NLL loss。
    推断时不传 y → 返回隐特征 H（由 FRAMETrainer 处理后续距离分类）。
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 latent_dim: int = 128, num_centroids: int = 2,
                 chunk_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_classes   = num_classes
        self.num_centroids = num_centroids
        self.latent_dim    = latent_dim

        self.embedding = EmbeddingNetwork(input_dim, latent_dim, dropout)

        # 每类独立的自注意力中心模块
        self.centroid_modules = nn.ModuleList([
            SelfAttentiveCentroid(latent_dim, num_centroids, chunk_size)
            for _ in range(num_classes)
        ])

        self.classifier = DistanceClassifier()

    def compute_centroids(self, H: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
        """
        H : (N, F),  y : (N,)
        返回 centroids : (C, nc, F)
        """
        centroids = []
        for c in range(self.num_classes):
            mask = (y == c)
            if mask.sum() == 0:
                # 极端情况：该类无样本，填零
                c_k = torch.zeros(self.num_centroids, self.latent_dim,
                                  device=H.device)
            else:
                c_k = self.centroid_modules[c](H[mask])   # (nc, F)
            centroids.append(c_k)
        return torch.stack(centroids, dim=0)               # (C, nc, F)

    def forward(self, x: torch.Tensor,
                y: torch.Tensor = None):

        H = self.embedding(x)                              # (N, F)
        if y is not None:
            centroids = self.compute_centroids(H, y)       # (C, nc, F)
            log_probs = self.classifier(H, centroids)      # (N, C)
            # 论文 Eq.(24)
            return F.nll_loss(log_probs, y)
        return H


# ─────────────────────────────────────────────────────────────────────────────
# 训练器：封装训练 / 预测 / 评估
# ─────────────────────────────────────────────────────────────────────────────

class FRAMETrainer:
    """
    封装 FRAME 的训练、推断与评估逻辑。

    参数
    ────
    input_dim     : 输入特征维度
    num_classes   : 类别数（默认 2，二分类）
    latent_dim    : 隐空间维度（论文默认 128）
    num_centroids : 每类中心数（论文最优 2）
    chunk_size    : 自注意力分块大小（论文最优 256）
    lr            : Adam 学习率（论文 1e-4）
    epochs        : 训练轮数（论文 1000）
    batch_size    : mini-batch 大小
    dropout       : Dropout 概率
    device        : 'cuda' 或 'cpu'（None = 自动检测）
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int = 2,
                 latent_dim: int = 128,
                 num_centroids: int = 2,
                 chunk_size: int = 256,
                 lr: float = 1e-4,
                 epochs: int = 1000,
                 batch_size: int = 256,
                 dropout: float = 0.1,
                 device: str = None):

        self.device      = "cuda"
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.num_classes = num_classes

        self.model = FRAMEModel(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            num_centroids=num_centroids,
            chunk_size=chunk_size,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 推断时使用的全局类中心（训练后构建）
        self._global_centroids: torch.Tensor = None

    # ── 训练 ──────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            verbose: bool = True, log_interval: int = 100):
        """
        端到端训练 FRAME。

        参数
        ────
        X_train      : 训练特征（DataFrame 或 ndarray）
        y_train      : 训练标签（Series 或 ndarray），整型类别
        verbose      : 是否打印训练日志
        log_interval : 每隔多少 epoch 打印一次损失
        """
        X_t, y_t = _to_tensor(X_train, y_train, self.device)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_b, y_b in loader:
                self.optimizer.zero_grad()
                loss = self.model(X_b, y_b)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(X_b)

            if verbose and epoch % log_interval == 0:
                print(f"[FRAME] Epoch {epoch:5d}/{self.epochs}"
                      f" | Loss: {epoch_loss / len(X_t):.6f}")

        # 训练完成后，用全量训练集构建并缓存全局类中心
        self._global_centroids = self._build_global_centroids(X_t, y_t)

    def _build_global_centroids(self, X_t: torch.Tensor,
                                y_t: torch.Tensor) -> torch.Tensor:
        """
        用全量训练数据构建最终的类中心（batch 方式避免显存溢出）。
        返回 : (C, nc, F)
        """
        self.model.eval()
        H_parts = []
        with torch.no_grad():
            for (x_b,) in DataLoader(TensorDataset(X_t),
                                     batch_size=self.batch_size):
                H_parts.append(self.model.embedding(x_b))
        H_all = torch.cat(H_parts, dim=0)                # (N_train, F)
        return self.model.compute_centroids(H_all, y_t)  # (C, nc, F)

    # ── 推断 ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回类概率矩阵，shape = (N, C)。"""
        if self._global_centroids is None:
            raise RuntimeError("请先调用 fit() 完成训练。")

        self.model.eval()
        X_t = _to_tensor(X, device=self.device)
        proba_parts = []
        with torch.no_grad():
            for (x_b,) in DataLoader(TensorDataset(X_t),
                                     batch_size=self.batch_size):
                H_b = self.model.embedding(x_b)
                log_p = self.model.classifier(H_b, self._global_centroids)
                proba_parts.append(torch.exp(log_p))
        return torch.cat(proba_parts, dim=0).cpu().numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测类别标签，shape = (N,)。"""
        return self.predict_proba(X).argmax(axis=1)

    # ── 评估 ──────────────────────────────────────────────────────────────

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        计算并返回评估指标字典：
            Accuracy | Macro-F1 | f1_class0 | f1_class1 | AUC | MCC | G-Mean
        """
        assert self.num_classes == 2

        y_true = y_val.values if isinstance(y_val, pd.Series) else np.array(y_val)
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)

        auc = roc_auc_score(y_true, y_proba[:, 1])
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # confusion matrix -> G-mean
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity)

        # classification report（字典形式！）
        report = classification_report(
            y_true, y_pred, digits=4, output_dict=True
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


# ─────────────────────────────────────────────────────────────────────────────
# 顶层 evaluate() — 适配用户提供的主程序模板
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(trainer: FRAMETrainer,
             X_val: pd.DataFrame,
             y_val: pd.Series) -> dict:
    """
    返回：Accuracy、Macro-F1、f1_class0、f1_class1、AUC、MCC、G-Mean

    参数
    ────
    trainer : 已训练好的 FRAMETrainer 实例
    X_val   : 验证集特征（DataFrame）
    y_val   : 验证集标签（Series）
    """
    return trainer.evaluate(X_val, y_val)


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
        sampling_method=None,
    )

    X_train = X['train']  # DataFrame
    y_train = pd.Series(dataset.y['train'].ravel())  # Series
    X_val = X['val']
    y_val = pd.Series(dataset.y['val'].ravel())

    all_results = defaultdict(list)

    for _ in list(range(10)):
        trainer = FRAMETrainer(
            input_dim=X_train.shape[1],
            num_classes=int(y_train.nunique()),
            latent_dim=128,  # 论文默认
            num_centroids=2, # 论文最优
            chunk_size=256,  # 论文最优
            lr=1e-4,         # 论文默认
            epochs=1000,     # 论文默认
            batch_size=1000,
            dropout=0.1,
        )
        trainer.fit(X_train, y_train, verbose=False, log_interval=100)

        metrics = evaluate(trainer, X_val, y_val)
        for k, v in metrics.items():
            all_results[k].append(v)

    # 求均值
    avg_results = {k: np.mean(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"FRAME, Dataset:{dataname}",
        log_file="eval_average.log"
    )

    # 求最大值
    avg_results = {k: np.max(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"FRAME, Dataset:{dataname}",
        log_file="eval_max.log"
    )

    # 求中位数
    avg_results = {k: np.median(v) for k, v in all_results.items()}
    write_avg_results_to_file(
        avg_results,
        help_str=f"FRAME, Dataset:{dataname}",
        log_file="eval_median.log"
    )


if __name__ == "__main__":
    raw_config_list = []
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp\churn\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/adult\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/shopper\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/Magic\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/bean\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/winequality\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/obesity\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/yeast_me2\CoTable\config.toml"))
    # raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/page\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/buddy\CoTable\config.toml"))
    raw_config_list.append(lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-LP-GAT\exp/mammography\CoTable\config.toml"))


    for raw_config in raw_config_list:
        main(raw_config)
