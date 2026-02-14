import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# ====================================================================
# 保持原有的 LabelPropagationSoftLabelGenerator 不变
# ====================================================================
class LabelPropagationSoftLabelGenerator:
    """
    在隐空间 Z 上构建 kNN 图，用 Label Propagation 得到软标签 p in [0,1]。
    关键点：只用"高置信 seeds"做强监督，其余点作为未标注点由传播决定。
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
        """
        n = y.shape[0]
        k = neigh_idx.shape[1]
        neigh_labels = y[neigh_idx]
        purity = (neigh_labels == y[:, None]).mean(axis=1)

        seeds = np.zeros(n, dtype=bool)
        classes = [0, 1]
        for c in classes:
            idx_c = np.where(y == c)[0]
            if idx_c.size == 0:
                continue

            n_seed = max(int(np.ceil(idx_c.size * self.seed_ratio)), self.min_seeds_per_class)
            n_seed = max(n_seed, sum(purity[idx_c] == 1.0))
            n_seed = min(n_seed, idx_c.size)

            order = np.argsort(-purity[idx_c])
            chosen = idx_c[order[:n_seed]]
            seeds[chosen] = True

        return seeds, purity

    def _propagate(
        self, neigh_idx, weights, seeds_mask, y,
        alpha_pos=0.80,
        alpha_neg=0.95,
        # alpha_pos=0.0,
        # alpha_neg=0.0,
        forbid_flip=True,
        flip_margin=1e-2
    ):
        """
        稀疏图上的迭代传播
        """
        n, k = neigh_idx.shape
        eps = 1e-12
        
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
            neigh_F = F_cur[neigh_idx]                    # (n,k,2) = (7828, 30, 2)
            SF = (W[:, :, None] * neigh_F).sum(axis=1)    # (n,2)

            F_new = alpha_i[:, None] * SF + (1.0 - alpha_i[:, None]) * Y

            # clamp seeds (hard constraint)
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

        # return prob of class 1
        p = F_cur[:, 1]
        p = np.clip(p, 0.0, 1.0)
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


# ====================================================================
# 工具函数
# ====================================================================
def build_knn_edges(Z: np.ndarray, k: int = 30, include_self: bool = False):
    """
    构建 kNN 图的边列表 (dst, src)
    """
    n = Z.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
    _, idx = nbrs.kneighbors(Z)

    if not include_self:
        idx = idx[:, 1:]
    else:
        k = k + 1

    dst = np.repeat(np.arange(n), k)
    src = idx.flatten()
    return dst, src


def segment_softmax(dst, e, num_nodes):
    """
    对边上的logit做分组softmax: dst相同的边共享softmax
    """
    if dst.numel() == 0:
        return torch.empty_like(e)

    if torch.cuda.is_available() and dst.is_cuda:
        max_per = torch.zeros((num_nodes,), device=e.device, dtype=e.dtype).fill_(-1e9)
        max_per.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
        e_exp = torch.exp(e - max_per[dst])

        sum_per = torch.zeros((num_nodes,), device=e.device, dtype=e.dtype)
        sum_per.scatter_add_(0, dst, e_exp)

        return e_exp / (sum_per[dst] + 1e-12)

    out = torch.empty_like(e)
    uniq = torch.unique(dst)
    for d in uniq:
        mask = (dst == d)
        out[mask] = torch.softmax(e[mask], dim=0)
    return out


# ====================================================================
# 新增: Focal Loss
# ====================================================================
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


# ====================================================================
# 新增: Supervised Contrastive Loss (基于硬标签)
# ====================================================================
class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失 - 使用硬标签进行类别对比
    
    将同类样本拉近，异类样本推远
    特别适合不平衡数据：给少数类更大的权重
    
    参数:
        temperature: 温度参数，控制分布的平滑度 (推荐0.07-0.1)
        base_temperature: 基础温度
        contrast_mode: 'all' 或 'one' (使用所有正样本还是随机一个)
        minority_weight: 少数类的权重倍数 (>1则增强少数类影响)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, 
                 contrast_mode='all', minority_weight=2.0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.minority_weight = minority_weight

    def forward(self, features, labels, mask=None):
        """
        features: (N, D) 特征向量 (已归一化或未归一化均可)
        labels: (N,) 硬标签 0/1
        mask: (N,) bool，指示哪些样本参与对比学习
        """
        device = features.device
        
        if mask is not None:
            features = features[mask]
            labels = labels[mask]
        
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # L2归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵: (N, N)
        similarity_matrix = torch.matmul(features, features.T)
        
        # 创建标签mask: (N, N) - 同类为1，异类为0
        labels = labels.contiguous().view(-1, 1)
        mask_same_class = torch.eq(labels, labels.T).float()
        
        # 移除对角线(自己和自己)
        logits_mask = torch.ones_like(mask_same_class)
        logits_mask.fill_diagonal_(0)
        mask_same_class = mask_same_class * logits_mask
        
        # 计算正样本数量
        pos_count = mask_same_class.sum(1)
        
        # 计算对比损失
        # exp(sim / temp)
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        
        # log sum exp (分母)
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 只对同类样本求和 (分子)
        mean_log_prob_pos = (mask_same_class * log_prob).sum(1) / (pos_count + 1e-12)
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # 【关键】根据样本类别加权：少数类权重更大
        # 假设label=1是少数类
        class_weights = torch.ones_like(labels.squeeze(), dtype=torch.float32)
        minority_mask = (labels.squeeze() == 1)
        class_weights[minority_mask] = self.minority_weight
        
        loss = loss * class_weights
        
        # 只计算有正样本的损失
        valid_mask = (pos_count > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        loss = loss[valid_mask].mean()
        
        return loss


# ====================================================================
# GAT Layer (保持不变)
# ====================================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, dst, src, num_nodes):
        N = x.size(0)
        h = self.W(x)
        h = h.view(N, self.heads, self.out_dim)

        h_dst = h[dst]
        h_src = h[src]

        e = (h_dst * self.a_dst[None, :, :]).sum(dim=-1) + (h_src * self.a_src[None, :, :]).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        alpha_list = []
        for head in range(self.heads):
            alpha_h = segment_softmax(dst, e[:, head], num_nodes)
            alpha_list.append(alpha_h)
        alpha = torch.stack(alpha_list, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((num_nodes, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        msg = alpha[:, :, None] * h_src
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


# ====================================================================
# 优化后的 GAT 分类器 (新增对比学习头)
# ====================================================================
class GATSoftClassifierWithContrastive(nn.Module):
    """
    改进的GAT分类器:
    1. 主任务: 软标签分类 (Focal Loss)
    2. 辅助任务: 对比学习 (基于硬标签)
    """
    def __init__(self, in_dim, hidden_dim=128, heads=4, dropout=0.2, projection_dim=64):
        super().__init__()
        
        # GAT layers for feature learning
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        
        # Classification head (主任务)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Projection head for contrastive learning (辅助任务)
        # 对比学习投影头: 将特征映射到对比学习空间
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x, dst, src, return_features=False):
        """
        x: (N, in_dim) 节点特征
        dst, src: 边索引
        return_features: 是否返回中间特征(用于对比学习)
        
        返回:
            logits: (N,) 分类logits
            features: (N, hidden_dim) 中间特征 (如果return_features=True)
        """
        num_nodes = x.size(0)
        
        # GAT feature learning
        h = self.gat1(x, dst, src, num_nodes)
        h = F.elu(h)
        h = self.gat2(h, dst, src, num_nodes)  # (N, hidden_dim)
        
        # Classification
        logits = self.classifier(h).squeeze(-1)  # (N,)
        
        if return_features:
            # Projection for contrastive learning
            proj_features = self.projection_head(h)  # (N, projection_dim)
            return logits, proj_features
        else:
            return logits


# ====================================================================
# 优化后的训练函数
# ====================================================================
def train_gat_soft_with_contrastive(
    Z_all: np.ndarray,
    y_all: np.ndarray,
    soft_p_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray = None,
    k: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    boundary_lambda: float = 2.0,
    # Focal Loss 参数
    focal_gamma: float = 2.0,
    focal_alpha: float = None,  # None则自动计算
    # 对比学习参数
    use_contrastive: bool = True,
    contrastive_weight: float = 0.5,  # 对比损失的权重
    contrastive_temp: float = 0.07,
    minority_weight: float = 2.0,  # 少数类在对比学习中的权重
    device: str = "cuda"
):
    """
    优化后的训练流程:
    1. 使用 Focal Loss 替代 BCE (更好处理难样本和不平衡)
    2. 加入对比学习辅助任务 (基于硬标签，增强特征判别性)
    3. 边界样本权重保留
    
    参数说明:
        focal_gamma: Focal Loss的gamma参数，推荐2.0-5.0，越大越关注难样本
        focal_alpha: 正类权重，None则自动根据类别比例计算
        use_contrastive: 是否使用对比学习
        contrastive_weight: 对比损失在总损失中的权重，推荐0.3-0.7
        contrastive_temp: 对比学习温度参数，推荐0.05-0.1
        minority_weight: 少数类在对比学习中的权重倍数，>1增强少数类影响
    """
    Z_all = np.asarray(Z_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)
    soft_p_all = np.asarray(soft_p_all, dtype=np.float32)

    # Build graph edges
    dst, src = build_knn_edges(Z_all, k=k, include_self=False)

    # Node features (可选择是否包含软标签信息)
    X_all = Z_all  # 仅使用原始特征，让GAT学习

    # 计算 focal_alpha (如果未指定)
    n_pos = int((y_all[train_mask] == 1).sum())
    n_neg = int((y_all[train_mask] == 0).sum())
    if focal_alpha is None:
        # alpha = n_neg / (n_neg + n_pos) 让正类权重更大
        focal_alpha = n_neg / (n_neg + n_pos)
    
    print(f"\n类别统计 (训练集):")
    print(f"  负类: {n_neg}, 正类: {n_pos}, 比例: {n_neg/n_pos:.2f}:1")
    print(f"  Focal Loss alpha: {focal_alpha:.4f}, gamma: {focal_gamma}")

    # Boundary sample weights
    u_all = 1.0 - np.abs(2.0 * soft_p_all - 1.0)
    sample_w = 1.0 + boundary_lambda * u_all

    # Torch tensors
    X = torch.tensor(X_all, device=device)
    soft_t = torch.tensor(soft_p_all, device=device)
    y_hard_t = torch.tensor(y_all, device=device, dtype=torch.long)
    w_t = torch.tensor(sample_w, device=device)
    dst_t = torch.tensor(dst, device=device, dtype=torch.long)
    src_t = torch.tensor(src, device=device, dtype=torch.long)
    train_mask_t = torch.tensor(train_mask, device=device, dtype=torch.bool)
    val_mask_t = None if val_mask is None else torch.tensor(val_mask, device=device, dtype=torch.bool)

    # Model
    model = GATSoftClassifierWithContrastive(
        in_dim=X.size(1), 
        hidden_dim=256, 
        heads=4, 
        dropout=0.2,
        projection_dim=64
    ).to(device)

    # Losses
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean').to(device)
    
    if use_contrastive:
        contrastive_loss_fn = SupervisedContrastiveLoss(
            temperature=contrastive_temp,
            minority_weight=minority_weight
        ).to(device)
        print(f"  对比学习: 启用 (weight={contrastive_weight}, temp={contrastive_temp}, minority_weight={minority_weight})")
    else:
        print(f"  对比学习: 禁用")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping
    best_state = None
    best_val = -1.0

    def macro_f1_from_logits(logits, y_true, mask, thr=0.5):
        probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        y = y_true[mask].cpu().numpy().astype(int)
        y_pred = (probs >= thr).astype(int)
        from sklearn.metrics import f1_score
        return float(f1_score(y, y_pred, average="macro"))

    print(f"\n开始训练 (epochs={epochs})...\n")

    for ep in range(1, epochs + 1):
        model.train()
        
        # Forward
        if use_contrastive:
            logits, proj_features = model(X, dst_t, src_t, return_features=True)
        else:
            logits = model(X, dst_t, src_t, return_features=False)

        # ===== 1. Focal Loss (主任务 - 软标签分类) =====
        loss_focal = focal_loss_fn(
            logits[train_mask_t],
            soft_t[train_mask_t],
            sample_weight=w_t[train_mask_t]
        )

        total_loss = loss_focal

        # ===== 2. Contrastive Loss (辅助任务 - 基于硬标签) =====
        if use_contrastive:
            loss_contrastive = contrastive_loss_fn(
                proj_features,
                y_hard_t,
                mask=train_mask_t
            )
            total_loss = total_loss + contrastive_weight * loss_contrastive

        # Backward
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Validation
        if val_mask_t is not None and ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(X, dst_t, src_t, return_features=False)

            val_f1 = macro_f1_from_logits(logits_val, y_hard_t, val_mask_t, thr=0.5)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if use_contrastive:
                print(f"Epoch {ep:03d} | Focal={loss_focal.item():.4f} | Contrastive={loss_contrastive.item():.4f} | Total={total_loss.item():.4f} | Val-F1={val_f1:.4f}")
            else:
                print(f"Epoch {ep:03d} | Focal={loss_focal.item():.4f} | Total={total_loss.item():.4f} | Val-F1={val_f1:.4f}")

        elif ep % 20 == 0:
            if use_contrastive:
                print(f"Epoch {ep:03d} | Focal={loss_focal.item():.4f} | Contrastive={loss_contrastive.item():.4f} | Total={total_loss.item():.4f}")
            else:
                print(f"Epoch {ep:03d} | Focal={loss_focal.item():.4f}")

    end_model = model
    # Load best model
    if best_state is not None:
        best_model = model.load_state_dict(best_state)
        print(f"\n加载最佳模型 (Val-F1={best_val:.4f})")

    return end_model, best_model


# ====================================================================
# 兼容性: 保留原有的训练函数接口
# ====================================================================
def train_gat_soft(
    Z_all: np.ndarray,
    y_all: np.ndarray,
    soft_p_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray = None,
    k: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    boundary_lambda: float = 2.0,
    device: str = "cuda"
):
    """
    兼容原有接口，默认启用 Focal Loss + Contrastive Learning
    """
    return train_gat_soft_with_contrastive(
        Z_all=Z_all,
        y_all=y_all,
        soft_p_all=soft_p_all,
        train_mask=train_mask,
        val_mask=val_mask,
        k=k,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        boundary_lambda=boundary_lambda,
        focal_gamma=2.0,
        focal_alpha=None,
        use_contrastive=True,
        contrastive_weight=0.5,
        contrastive_temp=0.07,
        minority_weight=2.0,
        device=device
    )
