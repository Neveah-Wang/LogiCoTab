import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# =========================================================
# 1) 软标签生成：复用你原实现（从 gat_soft_classifier_improved.py 拷贝）
# =========================================================
class LabelPropagationSoftLabelGenerator:
    """
    在隐空间 Z 上构建 kNN 图，用 Label Propagation 得到软标签 p in [0,1]。
    只用训练集做传播，避免任何 val/test 泄露。
    """
    def __init__(
        self,
        n_neighbors: int = 30,
        alpha: float = 0.9,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed_ratio: float = 0.2,
        min_seeds_per_class: int = 10,
        adaptive_bandwidth: bool = True,
        prior_from_data: bool = True,
        boundary_weight_lambda: float = 1.0,
        device: str = "cuda"
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

    def _build_knn_graph(self, Z: np.ndarray):
        """
        返回：
            neigh_idx: (n, k) int
            neigh_dist: (n, k) float
        """
        k = self.n_neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
        dist, idx = nbrs.kneighbors(Z)
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
        return np.maximum(w, 1e-12)

    def _select_seeds(self, y: np.ndarray, neigh_idx: np.ndarray):
        """
        按邻域纯度为每一类选择 seeds。
        purity(i) = mean( y[neighbors]==y[i] )
        """
        n = y.shape[0]
        neigh_labels = y[neigh_idx]
        purity = (neigh_labels == y[:, None]).mean(axis=1)

        seeds = np.zeros(n, dtype=bool)
        for c in [0, 1]:
            idx_c = np.where(y == c)[0]
            if idx_c.size == 0:
                continue

            n_seed = max(int(np.ceil(idx_c.size * self.seed_ratio)), self.min_seeds_per_class)
            n_seed = max(n_seed, int(np.sum(purity[idx_c] == 1.0)))
            n_seed = min(n_seed, idx_c.size)

            order = np.argsort(-purity[idx_c])
            chosen = idx_c[order[:n_seed]]
            seeds[chosen] = True

        return seeds, purity

    def _propagate(
        self, neigh_idx, weights, seeds_mask, y,
        # alpha_pos=0.80,
        # alpha_neg=0.95,
        alpha_pos=0.00,
        alpha_neg=0.85,
        forbid_flip=True,
        flip_margin=1e-2
    ):
        print("_propagate 的参数值:")
        for key, value in locals().items():
            if key in ['self', 'neigh_idx', 'weights', 'seeds_mask', 'y']:
                continue
            print(f"  {key} = {value}")

        n, k = neigh_idx.shape
        eps = 1e-12

        Y = np.ones((n, 2), dtype=np.float64)
        Y[y == 0] = np.array([1.0, 0.0])
        Y[y == 1] = np.array([0.0, 1.0])

        # --------------------------------------------------
        # 2) Build asymmetric transition matrix S
        #    (boost positive neighbors) 如果boost是1，就是对称的
        # --------------------------------------------------
        neigh_labels = y[neigh_idx]
        boost = np.ones_like(weights)
        boost[neigh_labels == 1] *= 1.0  # 如需增强可 >1

        W = weights * boost
        W = np.maximum(W, eps)
        W = W / (W.sum(axis=1, keepdims=True) + eps)

        # --------------------------------------------------
        # 3) Iterative propagation (per-node alpha)
        # --------------------------------------------------
        F_cur = Y.copy()
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        alpha_i = np.zeros(n)
        alpha_i[pos_mask] = alpha_pos
        alpha_i[neg_mask] = alpha_neg

        for _ in range(self.max_iter):
            neigh_F = F_cur[neigh_idx]                 # (n,k,2)
            SF = (W[:, :, None] * neigh_F).sum(axis=1) # (n,2)

            F_new = alpha_i[:, None] * SF + (1.0 - alpha_i[:, None]) * Y

            # clamp seeds
            F_new[seeds_mask & (y == 0)] = np.array([1.0, 0.0])
            F_new[seeds_mask & (y == 1)] = np.array([0.0, 1.0])

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

        F_cur = F_cur / (F_cur.sum(axis=1, keepdims=True) + eps)
        p = np.clip(F_cur[:, 1], 0.0, 1.0)
        return p

    def generate(self, Z: np.ndarray, y: np.ndarray):
        """
        返回：
            soft_p: (n,) 软标签
            sample_weight: (n,) 样本权重
            seeds_mask: (n,) bool
            purity: (n,) float
        """
        y = np.asarray(y, dtype=np.int64)
        Z = np.asarray(Z, dtype=np.float32)

        neigh_idx, neigh_dist = self._build_knn_graph(Z)
        weights = self._compute_weights(neigh_dist)
        seeds_mask, purity = self._select_seeds(y, neigh_idx)

        soft_p = self._propagate(neigh_idx, weights, seeds_mask, y)

        u = 1.0 - np.abs(2.0 * soft_p - 1.0)
        sample_weight = 1.0 + self.boundary_weight_lambda * u

        return soft_p, sample_weight, seeds_mask, purity


# =========================================================
# 2) 损失函数：Focal + SupCon + KL(二分类Bernoulli KL)
# =========================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with soft labels

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    参数:
        alpha: 正类权重 (默认根据类别比例自动计算)
        gamma: 聚焦参数，越大越关注难样本 (推荐2.0)
        reduction: 'mean' or 'sum' or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, soft_targets, sample_weight=None):
        """
        logits: (N,) 模型输出的logits
        soft_targets: (N,) 软标签 [0, 1]
        sample_weight: (N,) 可选的样本权重
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)

        soft_targets = torch.tensor(soft_targets, dtype=torch.float)

        # p_t: 如果target=1则p_t=p, 如果target=0则p_t=1-p
        # 对于软标签，我们使用加权组合
        p_t = soft_targets * probs + (1 - soft_targets) * (1 - probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, soft_targets, reduction='none'
        )

        # Focal loss
        focal_loss = focal_weight * bce_loss

        # Alpha weighting (类别平衡)
        if self.alpha is not None:
            alpha_t = soft_targets * self.alpha + (1 - soft_targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        # Sample weighting (边界样本加权)
        if sample_weight is not None:
            focal_loss = focal_loss * sample_weight

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftLabelFocalLoss(nn.Module):
    """
    修正版本的Focal Loss，适配软标签场景

    核心修改：
    - p_t 定义修改为 1 - |s - p_hat|，确保完全一致时损失为0
    - 保留调制因子 (1-p_t)^gamma 的难样本聚焦机制
    - 保留alpha类别平衡权重
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean',):
        """
        参数:
            alpha: 正类权重 (默认根据类别比例自动计算)
            gamma: 聚焦参数，越大越关注难样本 (推荐2.0)
            reduction: 'mean' or 'sum' or 'none'
            pt_definition: 'original' or 'modified'
                - 'original': p_t = s*p + (1-s)*(1-p) (原始定义)
                - 'modified': p_t = 1 - |s - p| (修正定义，推荐)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, soft_targets, y, sample_weight=None):
        """
        参数:
            logits: (N,) 模型输出的logits
            soft_targets: (N,) 软标签 [0, 1]
            sample_weight: (N,) 可选的样本权重
        """
        # 计算sigmoid概率
        probs = torch.sigmoid(logits)
        y = torch.tensor(y, dtype=torch.float)

        # 计算p_t,基于绝对误差
        p_t = 1 - torch.abs(soft_targets - probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        # focal_weight = 1.0

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, y, reduction='none'
        )

        # Focal loss
        focal_loss = focal_weight * bce_loss

        # Alpha weighting (类别平衡)
        if self.alpha is not None:
            alpha_t = soft_targets * self.alpha + (1 - soft_targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        # Sample weighting (边界样本加权)
        if sample_weight is not None:
            focal_loss = focal_loss * sample_weight

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, minority_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.minority_weight = minority_weight

    def forward(self, features, labels, mask=None):
        device = features.device
        if mask is not None:
            features = features[mask]
            labels = labels[mask]

        n = features.size(0)
        if n < 2:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T)  # (n,n)

        labels = labels.view(-1, 1)
        same = torch.eq(labels, labels.T).float()

        logits_mask = torch.ones_like(same)
        logits_mask.fill_diagonal_(0)
        same = same * logits_mask

        pos_count = same.sum(1)

        exp_logits = torch.exp(sim / self.temperature) * logits_mask
        log_prob = sim / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (same * log_prob).sum(1) / (pos_count + 1e-12)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # 少数类加权（假设1是少数类）
        w = torch.ones(n, device=device)
        w[(labels.squeeze() == 1)] = self.minority_weight
        loss = loss * w

        valid = (pos_count > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        return loss[valid].mean()


def bernoulli_kl(teacher_p, student_p, eps=1e-6):
    """
    KL( Bern(teacher_p) || Bern(student_p) )
    teacher_p, student_p: (N,) in [0,1]
    """
    t = torch.clamp(teacher_p, eps, 1 - eps)
    s = torch.clamp(student_p, eps, 1 - eps)
    return (t * torch.log(t / s) + (1 - t) * torch.log((1 - t) / (1 - s)))


# =========================================================
# 3) 概率预测网络：MLP + projection head（用于SupCon）
# =========================================================
"""
class ProbMLPWithProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=2, dropout=0.2, projection_dim=64):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, 1)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x, return_features=False):
        h = self.encoder(x)
        logits = self.classifier(h).squeeze(-1)
        if return_features:
            v = self.proj(h)
            return logits, v
        return logits
"""

class ProbMLPWithProjection(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 256, 128, 64], dropout=0.2):
        """
        参数:
            input_dim: 输入维度（隐空间维度）
            hidden_dims: 隐藏层维度列表
            dropout: dropout比例
        """
        super().__init__()

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_dims[:-2]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.proj = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-2], hidden_dims[-1])
        )

        # 输出层
        self.classifier = nn.Linear(hidden_dims[-1], 1)


    def forward(self, x):
        h = self.encoder(x)
        f = self.proj(h)
        logits = self.classifier(f).squeeze(-1)

        return logits, f


# =========================================================
# 4) 训练函数：计算 KL / Focal / SupCon，
# =========================================================
def train_prob_model(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    soft_p_all: np.ndarray,
    sample_w_all: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    # focal
    focal_gamma: float = 2.0,
    focal_alpha: float = None,
    # contrastive
    use_contrastive: bool = True,
    contrastive_weight: float = 0.5,
    contrastive_temp: float = 0.07,
    minority_weight: float = 2.0,
    # KL
    use_kl: bool = True,
    kl_weight: float = 1.0,
    device: str = "cuda",
):
    print("训练函数参数值:")
    for key, value in locals().items():
        if key in ['Z_train', 'y_train', 'Z_val', 'y_val', 'soft_p_all', 'sample_w_all']:
            continue
        print(f"  {key} = {value}")

    Z_all = np.asarray(Z_train, dtype=np.float32)
    y_all = np.asarray(y_train, dtype=np.int64)
    Z_val = np.asarray(Z_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    soft_p_all = np.asarray(soft_p_all, dtype=np.float32)
    sample_w_all = np.asarray(sample_w_all, dtype=np.float32)

    X = torch.tensor(Z_all, device=device)
    y_t = torch.tensor(y_all, device=device, dtype=torch.long)
    Z_val = torch.tensor(Z_val, device=device)
    y_val = torch.tensor(y_val, device=device, dtype=torch.long)

    soft_t = torch.tensor(soft_p_all, device=device)
    w_t = torch.tensor(sample_w_all, device=device)


    # focal_alpha auto（只用训练集统计）
    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    if focal_alpha is None:
        focal_alpha = n_neg / (n_neg + n_pos + 1e-12)

    model = ProbMLPWithProjection(
        in_dim=X.size(1),
        hidden_dims=[128, 256, 128, 64],
        dropout=0.2,
    ).to(device)

    # focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean").to(device)
    focal_fn = SoftLabelFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean").to(device)

    if use_contrastive:
        con_fn = SupervisedContrastiveLoss(
            temperature=contrastive_temp,
            minority_weight=minority_weight
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = -1.0

    def macro_f1_from_logits(logits, y_true, thr=0.5):
        from sklearn.metrics import f1_score
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        yy = y_true.detach().cpu().numpy().astype(int)
        y_pred = (probs >= thr).astype(int)
        return float(f1_score(yy, y_pred, average="macro"))

    for ep in range(1, epochs + 1):
        model.train()
        logits, feats = model(X)

        # (A) Focal on train
        loss_f = focal_fn(
            logits,
            soft_t,
            y_t,
            sample_weight=w_t
        )
        total = loss_f
        # total = 0

        # (B) KL on train: teacher=soft label, student=sigmoid(logit)
        if use_kl:
            student_p = torch.sigmoid(logits)
            teacher_p = soft_t.detach()  # stop-grad：教师分布固定
            loss_kl = bernoulli_kl(teacher_p, student_p).mean()
            total = total + kl_weight * loss_kl
        else:
            loss_kl = torch.tensor(0.0, device=device)

        # (C) SupCon on train (no leakage)
        if use_contrastive:
            loss_c = con_fn(feats, y_t, mask=None)
            total = total + contrastive_weight * loss_c
        else:
            loss_c = torch.tensor(0.0, device=device)

        opt.zero_grad()
        total.backward()
        opt.step()

        # validate
        if Z_val is not None and ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_val, _ = model(Z_val)
            val_f1 = macro_f1_from_logits(logits_val, y_val, thr=0.5)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"Ep {ep:03d} | Focal={loss_f.item():.4f} | KL={loss_kl.item():.4f} | "
                f"Con={loss_c.item():.4f} | Total={total.item():.4f} | ValF1@0.5={val_f1:.4f}"
            )
        elif ep % 20 == 0:
            print(
                f"Ep {ep:03d} | Focal={loss_f.item():.4f} | KL={loss_kl.item():.4f} | "
                f"Con={loss_c.item():.4f} | Total={total.item():.4f}"
            )

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[OK] Load best model by ValF1@0.5={best_val:.4f}")
    return model
