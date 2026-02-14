import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)

import lib
from lib.make_dataset import make_dataset


# -----------------------------
# 1) Label Propagation Soft Labels
# -----------------------------
class LabelPropagationSoftLabelGenerator:
    """
    在隐空间 Z 上构建 kNN 图，用 Label Propagation 得到软标签 p in [0,1]。
    关键点：只用“高置信 seeds”做强监督，其余点作为未标注点由传播决定。
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        alpha: float = 0.9,                 # propagation strength
        max_iter: int = 200,
        tol: float = 1e-6,
        seed_ratio: float = 0.2,            # per-class seed ratio
        min_seeds_per_class: int = 10,
        adaptive_bandwidth: bool = True,
        prior_from_data: bool = True,
        boundary_weight_lambda: float = 2.0, # sample weight strength for boundary points
        device: str = "cpu"
    ):
        self.n_neighbors = int(n_neighbors)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed_ratio = float(seed_ratio)
        self.min_seeds_per_class = int(min_seeds_per_class)
        self.adaptive_bandwidth = bool(adaptive_bandwidth)
        self.prior_from_data = bool(prior_from_data)
        self.boundary_weight_lambda = float(boundary_weight_lambda)
        self.device = device

    @staticmethod
    def _safe_minmax(x, eps=1e-12):
        x = np.asarray(x)
        return (x - x.min()) / (x.max() - x.min() + eps)

    def _build_knn_graph(self, Z: np.ndarray):
        """
        返回：
            neigh_idx: (n, k) int
            neigh_dist: (n, k) float
        """
        n = Z.shape[0]
        k = self.n_neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
        dist, idx = nbrs.kneighbors(Z)
        # drop self
        neigh_dist = dist[:, 1:]
        neigh_idx = idx[:, 1:]
        return neigh_idx, neigh_dist

    def _compute_weights(self, neigh_dist: np.ndarray):
        """
        RBF 权重：exp(-d^2 / (2*sigma_i^2))
        sigma_i 采用邻域平均距离（自适应带宽）
        """
        if self.adaptive_bandwidth:
            sigma = np.mean(neigh_dist, axis=1, keepdims=True)
            sigma = np.maximum(sigma, 1e-6)
        else:
            sigma = np.ones((neigh_dist.shape[0], 1)) * np.median(neigh_dist)

        w = np.exp(-(neigh_dist ** 2) / (2.0 * (sigma ** 2)))
        # 避免全 0
        w = np.maximum(w, 1e-12)
        return w

    def _select_seeds(self, y: np.ndarray, neigh_idx: np.ndarray):
        """
        按邻域纯度为每一类选择 seeds。
        purity(i) = mean( y[neighbors]==y[i] )
        每个类 选择 top n_seed 个 seeds，其中 n_seed = max(纯度为1的数量, 纯度最高的前seed_ratio个, 最小seeds数量)
        """
        n = y.shape[0]
        k = neigh_idx.shape[1]
        neigh_labels = y[neigh_idx]                  # (n, k)
        purity = (neigh_labels == y[:, None]).mean(axis=1)  # (n,)

        seeds = np.zeros(n, dtype=bool)
        classes = [0, 1]
        for c in classes:
            idx_c = np.where(y == c)[0]
            if idx_c.size == 0:
                continue

            # 选 top purity
            n_seed = max(int(np.ceil(idx_c.size * self.seed_ratio)), self.min_seeds_per_class)
            n_seed = max(n_seed, sum(purity[idx_c] == 1.0))
            n_seed = min(n_seed, idx_c.size)

            order = np.argsort(-purity[idx_c])   # descending
            chosen = idx_c[order[:n_seed]]
            seeds[chosen] = True

        return seeds, purity

    def _propagate_old(self, neigh_idx, weights, seeds_mask, y):
        """
        稀疏图上的迭代传播：F_{t+1} = alpha * S F_t + (1-alpha) * Y
        其中 S 为行归一化权重，Y 在 seeds 为 one-hot，否则为先验。
        """
        n, k = neigh_idx.shape
        eps = 1e-12
        """
        # prior
        if self.prior_from_data:
            pi = float((y == 1).mean())
        else:
            pi = 0.5
        prior = np.array([1.0 - pi, pi], dtype=np.float64)

        # Y: (n,2)
        Y = np.tile(prior[None, :], (n, 1))
        # clamp seeds to one-hot
        Y[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
        Y[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
        """
        Y = np.ones((n, 2), dtype=np.float64)
        Y[y == 0] = np.array([1.0, 0.0])
        Y[y == 1] = np.array([0.0, 1.0])


        # init F
        F_cur = Y.copy()   # (n,2) = (7828, 2)

        # row-normalized weights => S
        row_sum = weights.sum(axis=1, keepdims=True) + eps
        S_w = weights / row_sum   # (n,k) = (7828, 30)
        # breakpoint()
        # iterative
        for _ in range(self.max_iter):
            neigh_F = F_cur[neigh_idx]                     # (n,k,2) = (7828, 30, 2)
            SF = (S_w[:, :, None] * neigh_F).sum(axis=1)   # (n,2)

            F_new = self.alpha * SF + (1.0 - self.alpha) * Y

            # clamp seeds (hard constraint)
            F_new[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
            F_new[seeds_mask & (y == 1)] = np.array([0.0, 1.0])

            diff = np.mean(np.abs(F_new - F_cur))
            F_cur = F_new
            if diff < self.tol:
                break

        # return prob of class 1
        p = F_cur[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return p

    def _propagate(
        self, neigh_idx, weights, seeds_mask, y,
        # 超参数
        alpha_pos=0.80,
        alpha_neg=0.95,
        forbid_flip=True,
        flip_margin=1e-3
    ):
        """
        稀疏图上的迭代传播：F_{t+1} = alpha * S F_t + (1-alpha) * Y
        其中 S 为行归一化权重，Y 在 seeds 为 one-hot，否则为先验。
        """
        n, k = neigh_idx.shape
        eps = 1e-12
        """
        # prior
        if self.prior_from_data:
            pi = float((y == 1).mean())
        else:
            pi = 0.5
        prior = np.array([1.0 - pi, pi], dtype=np.float64)

        # Y: (n,2)
        Y = np.tile(prior[None, :], (n, 1))
        # clamp seeds to one-hot
        Y[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
        Y[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
        """
        Y = np.ones((n, 2), dtype=np.float64)
        Y[y == 0] = np.array([1.0, 0.0])
        Y[y == 1] = np.array([0.0, 1.0])

        # --------------------------------------------------
        # 2) Build asymmetric transition matrix S
        #    (boost positive neighbors)
        # --------------------------------------------------
        neigh_labels = y[neigh_idx]  # (n, k)
        boost = np.ones_like(weights)
        boost[neigh_labels == 1] *= 1

        W = weights * boost
        W = np.maximum(W, eps)
        W = W / (W.sum(axis=1, keepdims=True) + eps)  # row-normalize

        # --------------------------------------------------
        # 3) Iterative propagation (per-node alpha)
        # --------------------------------------------------
        # init F
        F_cur = Y.copy()  # (n,2) = (7828, 2)

        pos_mask = (y == 1)
        neg_mask = (y == 0)
        alpha_i = np.zeros(n)
        alpha_i[pos_mask] = alpha_pos
        alpha_i[neg_mask] = alpha_neg

        for _ in range(self.max_iter):
            neigh_F = F_cur[neigh_idx]                    # (n,k,2) = (7828, 30, 2)
            SF = (W[:, :, None] * neigh_F).sum(axis=1)    # (n,2)

            F_new = alpha_i[:, None] * SF + (1.0 - alpha_i[:, None]) * Y

            # clamp seeds (hard constraint)
            F_new[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
            F_new[seeds_mask & (y == 1)] = np.array([0.0, 1.0])
            # breakpoint()
            # optional: forbid label flipping
            if forbid_flip:
                p = F_new[:, 1]
                p[pos_mask] = np.maximum(p[pos_mask], 0.5 + flip_margin)
                p[neg_mask] = np.minimum(p[neg_mask], 0.5 - flip_margin)
                F_new[:, 1] = p
                F_new[:, 0] = 1.0 - p

            diff = np.mean(np.abs(F_new - F_cur))
            F_cur = F_new
            if diff < self.tol:
                break

        # return prob of class 1
        p = F_cur[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return p

    def generate(self, Z: np.ndarray, y: np.ndarray):
        """
        返回：
            soft_labels: (n,) in [0,1]
            sample_weights: (n,)  边界点权重（可用于训练加权）
            seeds_mask: (n,) bool
            purity: (n,) float
        """
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        neigh_idx, neigh_dist = self._build_knn_graph(Z)
        weights = self._compute_weights(neigh_dist)
        seeds_mask, purity = self._select_seeds(y, neigh_idx)
        soft_labels = self._propagate(neigh_idx, weights, seeds_mask, y)
        # soft_labels = self._propagate_old(neigh_idx, weights, seeds_mask, y)

        # boundary-aware sample weights: p near 0.5 => higher weight
        # w_i = 1 + lambda*(1 - |2p-1|)
        boundary_strength = 1.0 - np.abs(2.0 * soft_labels - 1.0)  # in [0,1]
        sample_weights = 1.0 + self.boundary_weight_lambda * boundary_strength

        return soft_labels, sample_weights, seeds_mask, purity


# -----------------------------
# 2) Classifier (MLP)
# -----------------------------
class SoftLabelClassifier(nn.Module):
    """基于隐空间 Z 的 MLP 二分类器"""

    def __init__(self, input_dim, hidden_dims=(256, 1024, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)


# -----------------------------
# 3) Cost-Sensitive Soft Label Loss
# -----------------------------
class CostSensitiveSoftLabelLoss(nn.Module):
    """
    代价敏感软标签损失：
    - 用 BCEWithLogits 直接对 soft target p
    - 用 pos_weight 强化少数类（正类）低分惩罚（FN 惩罚增强）
    - 可选 sample_weight（边界点更重要）
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits, soft_targets, sample_weight=None):
        # BCEWithLogits 支持 pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits,
            soft_targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


# -----------------------------
# 4) Full Pipeline
# -----------------------------
class ImbalancedLPSoftLabelClassifier:
    """
    完整方案：
    - Label Propagation 生成 soft labels
    - 代价敏感训练 MLP
    - 验证集自动选阈值
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims=(256, 1024, 256),
        dropout=0.3,
        # LP params
        lp_neighbors=30,
        lp_alpha=0.9,
        lp_seed_ratio=0.2,
        lp_max_iter=200,
        lp_tol=1e-6,
        boundary_weight_lambda=2.0,
        # train params
        learning_rate=1e-3,
        weight_decay=1e-5,
        device="cuda"
    ):
        self.device = device

        self.lp = LabelPropagationSoftLabelGenerator(
            n_neighbors=lp_neighbors,
            alpha=lp_alpha,
            seed_ratio=lp_seed_ratio,
            max_iter=lp_max_iter,
            tol=lp_tol,
            boundary_weight_lambda=boundary_weight_lambda,
            device="cpu"
        )

        self.classifier = SoftLabelClassifier(
            latent_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = None  # set in fit
        self.best_threshold_ = 0.5

    @staticmethod
    def _find_best_threshold(y_true, y_score, metric="macro_f1"):
        """
        在验证集扫描阈值，返回最优阈值和对应分数。
        metric: "macro_f1" or "gmean"
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).reshape(-1)

        best_t, best_s = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            y_pred = (y_score >= t).astype(int)
            if metric == "macro_f1":
                s = f1_score(y_true, y_pred, average="macro")
            elif metric == "gmean":
                cm = confusion_matrix(y_true, y_pred)
                if cm.size != 4:
                    continue
                tn, fp, fn, tp = cm.ravel()
                sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                s = float(np.sqrt(sen * spe))
            else:
                raise ValueError("metric must be 'macro_f1' or 'gmean'")

            if s > best_s:
                best_s = s
                best_t = float(t)

        return best_t, best_s

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        epochs=100,
        batch_size=256,
        Z_val=None,
        y_val=None,
        threshold_metric="macro_f1",
        verbose=True
    ):
        Z_train = np.asarray(Z_train)
        y_train = np.asarray(y_train).astype(int)

        # --------- 1) generate soft labels by LP ----------
        if verbose:
            print("[LP] Generating soft labels by label propagation ...")
        soft_y, sample_w, seeds_mask, purity = self.lp.generate(Z_train, y_train)

        # pos_weight for cost-sensitive loss
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        pos_weight = (n_neg / max(n_pos, 1))

        if verbose:
            print(f"[LP] pos_weight (neg/pos) = {pos_weight:.4f}")
            print(f"[LP] seeds selected: {int(seeds_mask.sum())}/{len(seeds_mask)}")
            print(f"[LP] purity mean={purity.mean():.4f}, median={np.median(purity):.4f}")

        self.criterion = CostSensitiveSoftLabelLoss(pos_weight=pos_weight).to(self.device)

        # tensors
        Zt = torch.tensor(Z_train, dtype=torch.float32, device=self.device)
        soft_t = torch.tensor(soft_y, dtype=torch.float32, device=self.device)
        sw_t = torch.tensor(sample_w, dtype=torch.float32, device=self.device)

        n = Z_train.shape[0]
        best_val = -1.0
        best_state = None
        best_threshold = 0.5

        # ------------- 2) train loop --------------
        for epoch in range(epochs):
            self.classifier.train()
            perm = torch.randperm(n, device=self.device)

            total_loss = 0.0
            nb = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]

                logits = self.classifier(Zt[idx])
                loss = self.criterion(
                    logits,
                    soft_t[idx],
                    sample_weight=sw_t[idx]
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                nb += 1

            avg_loss = total_loss / max(nb, 1)

            # --------- 3) validation + threshold selection ----------
            if Z_val is not None and y_val is not None:
                y_val_score = self.predict_proba(Z_val).reshape(-1)
                # choose threshold on val
                t_star, s_star = self._find_best_threshold(y_val, y_val_score, metric=threshold_metric)

                # you can also use AUC as a secondary indicator
                auc = roc_auc_score(y_val, y_val_score)

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1:03d}/{epochs} | loss={avg_loss:.4f} | "
                        f"val_{threshold_metric}={s_star:.4f} @t={t_star:.2f} | auc={auc:.4f}"
                    )

                # early best by the chosen metric
                if s_star > best_val:
                    best_val = s_star
                    best_threshold = t_star
                    best_state = {k: v.detach().cpu().clone() for k, v in self.classifier.state_dict().items()}
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:03d}/{epochs} | loss={avg_loss:.4f}")

        # restore best
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
            self.best_threshold_ = best_threshold
        else:
            self.best_threshold_ = 0.5

        if verbose and (Z_val is not None and y_val is not None):
            print(f"[Done] best_threshold_={self.best_threshold_:.2f}, best_{threshold_metric}={best_val:.4f}")

        return self

    def predict_proba(self, Z: np.ndarray):
        self.classifier.eval()
        Z = np.asarray(Z)
        with torch.no_grad():
            Zt = torch.tensor(Z, dtype=torch.float32, device=self.device)
            logits = self.classifier(Zt)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

    def predict(self, Z: np.ndarray, threshold=None):
        if threshold is None:
            threshold = self.best_threshold_
        probs = self.predict_proba(Z).reshape(-1)
        return (probs >= threshold).astype(int)

    def evaluate(self, Z, y, threshold=None, help="eval"):
        if threshold is None:
            threshold = self.best_threshold_

        y = np.asarray(y).astype(int)
        y_score = self.predict_proba(Z).reshape(-1)
        y_pred = (y_score >= threshold).astype(int)

        f1 = f1_score(y, y_pred, average="macro")
        auc = roc_auc_score(y, y_score)
        mcc = matthews_corrcoef(y, y_pred)
        report = classification_report(y, y_pred, digits=4)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = float(np.sqrt(sensitivity * specificity))

        print("")
        print("-" * 20, help, "-" * 20)
        print(f"threshold: {threshold:.3f}")
        print(f"F1 (macro): {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"G-mean: {gmean:.4f}")
        print("Report:")
        print(report)

        return {"auc": auc, "f1_macro": f1, "mcc": mcc, "gmean": gmean}


# -----------------------------
# 5) Minimal Example Usage
# -----------------------------
def example_usage_with_latent_arrays(Z_train, y_train, Z_val, y_val, Z_test, y_test, device="cuda"):
    """
    你把 VAE 的 latent_z 与标签传进来即可。
    """
    if Z_train.ndim == 3:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
        Z_val = Z_val.reshape(Z_val.shape[0], -1)
        Z_test = Z_test.reshape(Z_test.shape[0], -1)

    model = ImbalancedLPSoftLabelClassifier(
        latent_dim=Z_train.shape[1],
        lp_neighbors=30,
        lp_alpha=0.9,
        lp_seed_ratio=0.2,
        boundary_weight_lambda=2.0,
        hidden_dims=(256, 1024, 256),
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device=device
    )

    model.fit(
        Z_train, y_train,
        epochs=500,
        batch_size=1024,
        Z_val=Z_val,
        y_val=y_val,
        threshold_metric="macro_f1",
        verbose=True
    )

    model.evaluate(Z_val, y_val, help="VAL")
    model.evaluate(Z_test, y_test, help="TEST")
    model.evaluate(Z_train, y_train, help="Train")
    return model


if __name__ == "__main__":

    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/churn\CoTable\config.toml")
    raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/adult\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/bean\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/page\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/buddy\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/mammography\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/yeast_me2\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/obesity\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/shopper\CoTable\config.toml")
    # raw_config = lib.util.load_config("D:\Study\自学\表格数据生成\LogiCoTab-vae\exp/magic\CoTable\config.toml")
    parent_dir = raw_config['parent_dir']

    # 1. 加载VAE生成的隐向量
    Z_train = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize.npy'))
    Z_test = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_val.npy'))
    Z_val = np.load(os.path.join(parent_dir, 'latent_data/latent_z_after_reparameterize_test.npy'))

    # 2. 展平(如果是3D)
    if Z_train.ndim == 3:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
        Z_test = Z_test.reshape(Z_test.shape[0], -1)
        Z_val = Z_val.reshape(Z_val.shape[0], -1)

    # 3. 加载标签(从原始数据)
    real_data_path = raw_config['real_data_path']
    dataset = make_dataset(real_data_path, raw_config)
    y_train = dataset.y['train'].flatten()
    y_test = dataset.y['val'].flatten()
    y_val = dataset.y['test'].flatten()

    print(f"\n训练集: {Z_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {Z_test.shape}, 类别分布: {np.bincount(y_test)}")
    print(f"验证集: {Z_val.shape}, 类别分布: {np.bincount(y_val)}")

    example_usage_with_latent_arrays(Z_train, y_train, Z_val, y_val, Z_test, y_test, device="cuda")